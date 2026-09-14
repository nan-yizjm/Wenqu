"""产品 API：即使没有资料、索引或模型也能启动设置页面。"""

from contextlib import asynccontextmanager
from pathlib import Path
import platform
import sys

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import PRODUCT_NAME, PRODUCT_VERSION
from .credentials import CredentialStore
from .database import Database
from .paths import ProductPaths, bundle_root
from .retrieval_model import RetrievalModelManager


ALLOWED_SETTINGS = {
    "provider", "ollama_base_url", "ollama_model", "onboarding_complete",
    "display_name", "retrieval_mode",
}
DEFAULT_SETTINGS = {
    "provider": "ollama", "ollama_base_url": "http://127.0.0.1:11434",
    "ollama_model": "qwen2.5:7b", "onboarding_complete": False,
    "display_name": "我的知识工作台", "retrieval_mode": "bm25",
}


class SettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str | None = None
    ollama_base_url: str | None = Field(default=None, max_length=300)
    ollama_model: str | None = Field(default=None, max_length=100)
    onboarding_complete: bool | None = None
    display_name: str | None = Field(default=None, max_length=60)
    retrieval_mode: str | None = None

    @field_validator("provider")
    @classmethod
    def provider_allowed(cls, value):
        if value is not None and value not in ("ollama", "deepseek"):
            raise ValueError("provider 只能是 ollama 或 deepseek")
        return value

    @field_validator("retrieval_mode")
    @classmethod
    def retrieval_allowed(cls, value):
        if value is not None and value not in ("bm25", "hybrid"):
            raise ValueError("retrieval_mode 只能是 bm25 或 hybrid")
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def local_ollama_only(cls, value):
        if value is not None and not value.startswith(("http://127.0.0.1", "http://localhost")):
            raise ValueError("第一版本只允许连接本机 Ollama")
        return value.rstrip("/") if value else value


class SecretBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: str = Field(min_length=8, max_length=500)


def current_settings(database: Database):
    return {**DEFAULT_SETTINGS, **database.get_settings()}


def create_product_app(paths: ProductPaths | None = None, credential_store=None,
                       static_dir: Path | None = None, shutdown_callback=None,
                       retrieval_model_manager=None):
    paths = (paths or ProductPaths.default()).ensure()
    credentials = credential_store or CredentialStore()
    model_manager = retrieval_model_manager or RetrievalModelManager(paths.model_cache)

    @asynccontextmanager
    async def lifespan(app):
        app.state.database = Database(paths)
        app.state.paths = paths
        app.state.credentials = credentials
        app.state.retrieval_model = model_manager
        app.state.runtime_state = {"status": "not_configured", "detail": None}
        yield

    app = FastAPI(title=PRODUCT_NAME, version=PRODUCT_VERSION, lifespan=lifespan)

    @app.middleware("http")
    async def local_access_only(request: Request, call_next):
        host = request.headers.get("host", "").split(":", 1)[0].lower()
        origin = request.headers.get("origin")
        if host not in {"127.0.0.1", "localhost", "testserver"}:
            return JSONResponse({"error": "local_access_required"}, status_code=403)
        if origin and not origin.startswith(("http://127.0.0.1:", "http://localhost:")):
            return JSONResponse({"error": "local_origin_required"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, error):
        return JSONResponse({"error": "invalid_request", "fields": [
            {"path": ".".join(str(value) for value in item["loc"]), "type": item["type"]}
            for item in error.errors()
        ]}, status_code=422)

    @app.get("/api/v1/health")
    async def health(request: Request):
        database = request.app.state.database
        return {
            "status": "ready", "version": PRODUCT_VERSION,
            "database_schema": database.schema_version(),
            "runtime": request.app.state.runtime_state,
        }

    @app.get("/api/v1/setup")
    async def setup(request: Request):
        database = request.app.state.database
        settings = current_settings(database)
        model_state = request.app.state.retrieval_model.status()
        return {
            "settings": settings,
            "deepseek_key_configured": request.app.state.credentials.has_deepseek(),
            "data_root": str(paths.root),
            "steps": {
                "workspace": bool(settings.get("display_name")),
                "materials": False,
                "retrieval_model": model_state["status"] == "ready",
                "generation": (
                    settings["provider"] == "ollama"
                    or request.app.state.credentials.has_deepseek()
                ),
            }, "retrieval_model": model_state,
        }

    @app.get("/api/v1/setup/retrieval-model")
    async def retrieval_model_status(request: Request):
        return request.app.state.retrieval_model.status()

    @app.post("/api/v1/setup/retrieval-model")
    async def prepare_retrieval_model(request: Request):
        state = request.app.state.retrieval_model.start(allow_download=True)
        request.app.state.database.event("retrieval_model_prepare_started")
        return state

    @app.get("/api/v1/settings")
    async def get_settings(request: Request):
        return {"settings": current_settings(request.app.state.database),
                "deepseek_key_configured": request.app.state.credentials.has_deepseek()}

    @app.patch("/api/v1/settings")
    async def patch_settings(body: SettingsPatch, request: Request):
        values = body.model_dump(exclude_none=True)
        unknown = set(values) - ALLOWED_SETTINGS
        if unknown:
            return JSONResponse({"error": "unknown_setting"}, status_code=422)
        request.app.state.database.set_settings(values)
        request.app.state.database.event("settings_updated", {"keys": sorted(values)})
        return {"settings": current_settings(request.app.state.database)}

    @app.put("/api/v1/credentials/deepseek")
    async def save_deepseek(body: SecretBody, request: Request):
        request.app.state.credentials.set_deepseek(body.api_key)
        request.app.state.database.event("deepseek_key_saved")
        return {"configured": True}

    @app.delete("/api/v1/credentials/deepseek")
    async def delete_deepseek(request: Request):
        request.app.state.credentials.delete_deepseek()
        request.app.state.database.event("deepseek_key_deleted")
        return {"configured": False}

    @app.get("/api/v1/system/diagnostics")
    async def diagnostics(request: Request):
        return {
            "product_version": PRODUCT_VERSION,
            "python": platform.python_version(),
            "frozen": bool(getattr(sys, "frozen", False)),
            "database_schema": request.app.state.database.schema_version(),
            "data_directories": {
                "root_exists": paths.root.is_dir(), "database_exists": paths.database.is_file(),
                "model_cache_exists": paths.model_cache.is_dir(),
            },
            "credentials": {"deepseek_configured": request.app.state.credentials.has_deepseek()},
            "runtime": request.app.state.runtime_state,
            "retrieval_model": request.app.state.retrieval_model.status(),
        }

    @app.post("/api/v1/system/shutdown")
    async def shutdown():
        if shutdown_callback:
            shutdown_callback()
        return {"status": "shutting_down"}

    resolved_static = static_dir or bundle_root() / "web" / "dist"
    assets = resolved_static / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        candidate = (resolved_static / full_path).resolve()
        if (full_path and candidate.is_file()
                and candidate.is_relative_to(resolved_static.resolve())):
            return FileResponse(candidate)
        index = resolved_static / "index.html"
        if index.is_file():
            return FileResponse(index)
        return JSONResponse({
            "error": "frontend_not_built",
            "message": "后端已启动；请先构建 web 前端。",
        }, status_code=503)

    return app
