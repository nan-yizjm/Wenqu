"""产品 API：即使没有资料、索引或模型也能启动设置页面。"""

from contextlib import asynccontextmanager
import asyncio
import json
from pathlib import Path
import platform
import sys
import threading

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import iterate_in_threadpool

from . import PRODUCT_NAME, PRODUCT_VERSION
from .credentials import CredentialStore
from .database import Database
from .paths import ProductPaths, bundle_root
from .retrieval_model import RetrievalModelManager
from .materials import MAX_UPLOAD_BYTES, MaterialService
from .folder_picker import pick_folder
from .chat import ChatService


ALLOWED_SETTINGS = {
    "provider", "ollama_base_url", "ollama_model", "onboarding_complete",
    "display_name", "retrieval_mode", "deepseek_model",
}
DEFAULT_SETTINGS = {
    "provider": "ollama", "ollama_base_url": "http://127.0.0.1:11434",
    "ollama_model": "qwen2.5:7b", "onboarding_complete": False,
    "display_name": "我的知识工作台", "retrieval_mode": "bm25",
    "deepseek_model": "deepseek-chat",
}


class SettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str | None = None
    ollama_base_url: str | None = Field(default=None, max_length=300)
    ollama_model: str | None = Field(default=None, max_length=100)
    onboarding_complete: bool | None = None
    display_name: str | None = Field(default=None, max_length=60)
    retrieval_mode: str | None = None
    deepseek_model: str | None = Field(default=None, max_length=100)

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


class FolderBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=1000)
    name: str | None = Field(default=None, max_length=100)


class ConversationBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="新会话", max_length=80)


class ChatBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str | None = Field(default=None, max_length=4000)
    retry_message_id: str | None = Field(default=None, max_length=100)


def current_settings(database: Database):
    return {**DEFAULT_SETTINGS, **database.get_settings()}


def create_product_app(paths: ProductPaths | None = None, credential_store=None,
                       static_dir: Path | None = None, shutdown_callback=None,
                       retrieval_model_manager=None, material_run_inline=False,
                       chat_client_factory=None):
    paths = (paths or ProductPaths.default()).ensure()
    credentials = credential_store or CredentialStore()
    model_manager = retrieval_model_manager or RetrievalModelManager(paths.model_cache)

    @asynccontextmanager
    async def lifespan(app):
        app.state.database = Database(paths)
        app.state.paths = paths
        app.state.credentials = credentials
        app.state.retrieval_model = model_manager
        app.state.materials = MaterialService(
            app.state.database, paths, run_inline=material_run_inline)
        app.state.chat = ChatService(
            app.state.database, app.state.materials, credentials,
            lambda: current_settings(app.state.database), chat_client_factory)
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
        materials = request.app.state.materials.setup_summary()
        return {
            "settings": settings,
            "deepseek_key_configured": request.app.state.credentials.has_deepseek(),
            "data_root": str(paths.root),
            "steps": {
                "workspace": bool(settings.get("display_name")),
                "materials": materials["ready_documents"] > 0,
                "retrieval_model": model_state["status"] == "ready",
                "generation": (
                    settings["provider"] == "ollama"
                    or request.app.state.credentials.has_deepseek()
                ),
            }, "retrieval_model": model_state, "materials": materials,
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

    @app.get("/api/v1/libraries")
    async def libraries(request: Request):
        return {"libraries": request.app.state.materials.list_libraries()}

    @app.post("/api/v1/system/pick-folder")
    async def system_pick_folder():
        try:
            return {"path": await asyncio.to_thread(pick_folder)}
        except (RuntimeError, OSError) as error:
            return JSONResponse({"error": "folder_picker_failed", "message": str(error)},
                                status_code=500)

    @app.post("/api/v1/libraries/folders")
    async def connect_folder(body: FolderBody, request: Request):
        try:
            return request.app.state.materials.connect_folder(body.path, body.name)
        except ValueError as error:
            return JSONResponse({"error": "invalid_folder", "message": str(error)}, status_code=422)

    @app.post("/api/v1/libraries/{library_id}/refresh")
    async def refresh_library(library_id: str, request: Request):
        try:
            return request.app.state.materials.refresh_library(library_id)
        except KeyError as error:
            return JSONResponse({"error": "library_not_found", "message": str(error.args[0])},
                                status_code=404)

    @app.get("/api/v1/documents")
    async def documents(request: Request):
        return {"documents": request.app.state.materials.list_documents()}

    @app.post("/api/v1/documents/upload")
    async def upload_document(request: Request, file: UploadFile = File(...)):
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        try:
            return request.app.state.materials.upload(file.filename or "未命名文件", data)
        except ValueError as error:
            return JSONResponse({"error": "invalid_document", "message": str(error)}, status_code=422)
        finally:
            await file.close()

    @app.delete("/api/v1/documents/{document_id}")
    async def remove_document(document_id: str, request: Request):
        try:
            request.app.state.materials.remove_document(document_id)
            return {"removed": True}
        except KeyError as error:
            return JSONResponse({"error": "document_not_found", "message": str(error.args[0])},
                                status_code=404)

    @app.post("/api/v1/documents/{document_id}/retry")
    async def retry_document(document_id: str, request: Request):
        try:
            return request.app.state.materials.retry_document(document_id)
        except KeyError as error:
            return JSONResponse({"error": "document_not_found", "message": str(error.args[0])},
                                status_code=404)
        except ValueError as error:
            return JSONResponse({"error": "retry_unavailable", "message": str(error)},
                                status_code=409)

    @app.get("/api/v1/import-jobs")
    async def import_jobs(request: Request):
        return {"jobs": request.app.state.materials.list_jobs()}

    @app.get("/api/v1/search")
    async def search(request: Request, q: str = Query(min_length=1, max_length=1000),
                     top_k: int = Query(default=8, ge=1, le=20)):
        return request.app.state.materials.search(q, top_k)

    @app.get("/api/v1/conversations")
    async def conversations(request: Request):
        return {"conversations": request.app.state.chat.list_conversations()}

    @app.post("/api/v1/conversations")
    async def create_conversation(body: ConversationBody, request: Request):
        return request.app.state.chat.create_conversation(body.title)

    @app.get("/api/v1/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str, request: Request):
        try:
            return request.app.state.chat.get_conversation(conversation_id)
        except KeyError as error:
            return JSONResponse({"error": "conversation_not_found", "message": str(error.args[0])},
                                status_code=404)

    @app.post("/api/v1/conversations/{conversation_id}/messages/stream")
    async def stream_message(conversation_id: str, body: ChatBody, request: Request):
        if not body.question and not body.retry_message_id:
            return JSONResponse({"error": "question_required", "message": "请输入问题。"},
                                status_code=422)
        try:
            request.app.state.chat.validate_stream_request(conversation_id, body.retry_message_id)
        except KeyError as error:
            return JSONResponse({"error": "invalid_chat", "message": str(error.args[0])},
                                status_code=404)
        cancel = threading.Event()
        iterator = request.app.state.chat.stream(
            conversation_id, body.question, body.retry_message_id, cancel)

        async def events():
            try:
                async for event in iterate_in_threadpool(iterator):
                    if await request.is_disconnected():
                        cancel.set()
                        break
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            finally:
                cancel.set()
                close = getattr(iterator, "close", None)
                if close:
                    close()

        return StreamingResponse(events(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/v1/conversations/{conversation_id}/messages/{message_id}/stop")
    async def stop_message(conversation_id: str, message_id: str, request: Request):
        try:
            return request.app.state.chat.stop(conversation_id, message_id)
        except KeyError as error:
            return JSONResponse({"error": "message_not_found", "message": str(error.args[0])},
                                status_code=404)

    @app.get("/api/v1/documents/{document_id}/versions/{version_id}/source")
    async def document_source(document_id: str, version_id: str, request: Request):
        try:
            return request.app.state.materials.source(document_id, version_id)
        except KeyError as error:
            return JSONResponse({"error": "source_not_found", "message": str(error.args[0])},
                                status_code=404)
        except ValueError as error:
            return JSONResponse({"error": "source_unavailable", "message": str(error)},
                                status_code=409)

    @app.get("/api/v1/documents/{document_id}/versions/{version_id}/file")
    async def document_file(document_id: str, version_id: str, request: Request):
        try:
            path, media_type, name = request.app.state.materials.source_file(document_id, version_id)
            return FileResponse(path, media_type=("application/pdf" if media_type == "pdf"
                                                  else "text/markdown; charset=utf-8"),
                                filename=name, content_disposition_type="inline")
        except KeyError as error:
            return JSONResponse({"error": "source_not_found", "message": str(error.args[0])},
                                status_code=404)
        except ValueError as error:
            return JSONResponse({"error": "source_unavailable", "message": str(error)},
                                status_code=409)

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
