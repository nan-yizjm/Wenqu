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
from .retrieval_model import RetrievalModelManager, build_cpu_encoder
from .materials import (
    MAX_UPLOAD_BYTES, RETRIEVAL_PARAMETER_KEYS, RETRIEVAL_PARAMETERS, MaterialService,
)
from .resources import bundled_docs, bundled_examples, resolve_bundled
from .folder_browser import list_directory
from .chat import ChatService
from .organize import OrganizeService
from .studio import ARTIFACT_KINDS, StudioService
from .support import MAX_BACKUP_BYTES, SupportService


ALLOWED_SETTINGS = {
    "provider", "ollama_base_url", "ollama_model", "onboarding_complete",
    "display_name", "retrieval_mode", "deepseek_model", "theme",
    # 检索参数（见 materials.RETRIEVAL_PARAMETERS）。界面上没有开关，只为单变量
    # 实验开放：默认值就是产品一直以来的行为。
    "candidate_k", "rrf_k", "bm25_k1", "bm25_b", "heading_repeat",
    # 记忆开关。默认关；关着时产品**根本不会调用**记忆提供者（见 chat.py）。
    "memory_enabled",
    # 联网开关与"首次开启告知已看过"的确认。两者分开：`web_enabled` 是能力开关，
    # `web_disclosure_acknowledged` 是"用户已经被告知什么会离开这台机器"的凭据。
    # 没有后者的前者会被接口拒掉（见 patch_settings），这样"首次开启必须显式告知"
    # 就不只是一句界面文案——文案会被改掉，规则不会。
    "web_enabled", "web_disclosure_acknowledged",
}
DEFAULT_SETTINGS = {
    "provider": "ollama", "ollama_base_url": "http://127.0.0.1:11434",
    "ollama_model": "qwen2.5:7b", "onboarding_complete": False,
    "display_name": "我的知识工作台", "retrieval_mode": "bm25",
    "deepseek_model": "deepseek-chat", "theme": "system",
    "memory_enabled": False,
    "web_enabled": False, "web_disclosure_acknowledged": False,
    **{key: default for key, (default, _, _, _) in RETRIEVAL_PARAMETERS.items()},
}
SOURCE_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "markdown": "text/markdown; charset=utf-8",
    "notebook": "application/x-ipynb+json",
}
# 一次批量删除最多接受多少条。这不是分页：界面上"全选"选中几百条是正常用法，
# 但"一次删掉几千条"多半是调用方算错了 id 列表。界限写在接口上，越界时把实际
# 数字一起报出来——只说"太多了"等于让调用方自己去猜。
MAX_SELECTION = 500
# 单个 id 的长度上限。请求体本身没有大小限制，放开这一条就等于允许 500 个长字符串。
MAX_SELECTED_ID = 100


class SettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str | None = None
    ollama_base_url: str | None = Field(default=None, max_length=300)
    ollama_model: str | None = Field(default=None, max_length=100)
    onboarding_complete: bool | None = None
    display_name: str | None = Field(default=None, max_length=60)
    retrieval_mode: str | None = None
    deepseek_model: str | None = Field(default=None, max_length=100)
    theme: str | None = None
    candidate_k: int | None = Field(default=None, ge=1, le=200)
    rrf_k: int | None = Field(default=None, ge=1, le=1000)
    bm25_k1: float | None = Field(default=None, gt=0.0, le=10.0)
    bm25_b: float | None = Field(default=None, ge=0.0, le=1.0)
    heading_repeat: int | None = Field(default=None, ge=0, le=10)
    memory_enabled: bool | None = None
    web_enabled: bool | None = None
    web_disclosure_acknowledged: bool | None = None

    @field_validator("provider")
    @classmethod
    def provider_allowed(cls, value):
        if value is not None and value not in ("ollama", "deepseek"):
            raise ValueError("provider 只能是 ollama 或 deepseek")
        return value

    @field_validator("theme")
    @classmethod
    def theme_allowed(cls, value):
        if value is not None and value not in ("system", "light", "dark"):
            raise ValueError("theme 只能是 system、light 或 dark")
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
    skip_guard: bool = False


class FavoriteBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message_id: str = Field(min_length=1, max_length=100)


class FavoritePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, max_length=100)
    note: str | None = Field(default=None, max_length=4000)
    tags: list[str] | None = Field(default=None, max_length=20)


class FeedbackBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message_id: str = Field(min_length=1, max_length=100)
    kind: str = Field(min_length=1, max_length=40)
    note: str = Field(default="", max_length=2000)


class MigrationRecoveryBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backup_name: str = Field(min_length=1, max_length=300)


class ResourceImportBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)


class BatchDeleteBody(BaseModel):
    """批量删除的请求体。三个列表（资料 / 收藏 / 产出）共用这一个形状。

    字段就叫 `ids`，资源写在路径里（`/documents/delete` 等）。三个形状几乎相同
    的模型只会带来三处各写一遍的去重与限长，而那种地方迟早会有一处不一样。

    `min_length=1` 把"空选择"挡在业务层之前：空请求回 200 会让界面把一次根本
    没执行的删除显示成"已完成"。
    """

    model_config = ConfigDict(extra="forbid")
    ids: list[str] = Field(min_length=1, max_length=MAX_SELECTION)

    @field_validator("ids")
    @classmethod
    def ids_are_short_and_unique(cls, value):
        for item in value:
            if len(item) > MAX_SELECTED_ID:
                raise ValueError(f"id 过长（上限 {MAX_SELECTED_ID} 个字符）：{item[:20]}…")
        # 去重：同一个 id 第二次必然报"不存在"，那条 skipped 只是噪音。保持出现
        # 顺序，`deleted` 的数字才对得上界面上看到的那几条。
        return list(dict.fromkeys(value))


class ArtifactBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str = Field(min_length=1, max_length=200)
    # 只允许 guide / mindmap：产出的种类决定了它有没有模型参与，不能由前端随手
    # 传一个字符串进来。默认 guide 是"更贵、更可能有错"的那一种，让它做默认值
    # 反而更容易被看到。
    kind: str = "guide"

    @field_validator("kind")
    @classmethod
    def kind_allowed(cls, value):
        if value not in ARTIFACT_KINDS:
            raise ValueError("产出类型只能是 guide 或 mindmap")
        return value


def current_settings(database: Database):
    """只暴露可回写的键。

    内部状态（如 active_index_version）也放在 settings 表里，但它属于运行信息、
    PATCH 会以 extra_forbidden 拒绝；混进来会让设置页整份回写时必然 422。
    这类键统一走 /api/v1/setup 的 materials 等专门字段。
    """
    stored = database.get_settings()
    return {**DEFAULT_SETTINGS, **{key: stored[key] for key in stored if key in ALLOWED_SETTINGS}}


def memory_state(chat) -> dict:
    """记忆接缝的状态，供诊断上报。

    诊断接口不该被第三方记忆实现拖垮：条目数是调用提供者才拿得到的，所以整段
    包在 try 里，实现抛异常时只把类型名报出来，不让 `/api/v1/system/diagnostics`
    跟着失败。
    """
    if chat is None:
        return {"available": False}
    provider = chat.memory
    state = {"available": True, "provider": getattr(provider, "name", "unknown")}
    try:
        state["items"] = len(provider.list(limit=1000))
    except Exception as error:  # 第三方实现的问题不该让诊断接口失败
        state["items"] = None
        state["error"] = type(error).__name__
    return state


def studio_state(database) -> dict:
    """产出的状态，供诊断上报。

    只按状态数个数，不读正文：诊断页要能在"产出很多"时不变成一次全表扫描。
    查不动就报 `available: False`——诊断接口本身不该被产出拖垮。
    """
    try:
        rows = database.fetchall(
            "SELECT status, COUNT(*) count FROM artifacts GROUP BY status")
    except Exception:
        return {"available": False}
    return {"available": True,
            "by_status": {row["status"]: row["count"] for row in rows}}


def web_state(chat, settings: dict) -> dict:
    """联网接缝的状态，供诊断上报。
    与记忆不同，这里**不向后端要任何东西**：搜索协议没有"列出已有结果"这种便宜
    调用，诊断不该为了显示一个数字去发一次请求——那等于让"打开诊断页"变成一次
    出网。所以这个函数全程只读本机状态，不需要 try。
    """
    if chat is None:
        return {"available": False}
    provider = getattr(chat, "web", None)
    return {"available": True,
            "provider": getattr(provider, "name", "unknown"),
            "configured": bool(getattr(provider, "configured", False)),
            "enabled": bool(settings.get("web_enabled")),
            "disclosure_acknowledged": bool(settings.get("web_disclosure_acknowledged"))}


def create_product_app(paths: ProductPaths | None = None, credential_store=None,
                       static_dir: Path | None = None, shutdown_callback=None,
                       retrieval_model_manager=None, material_run_inline=False,
                       chat_client_factory=None, material_encoder_factory=None,
                       memory_provider_factory=None, search_provider_factory=None):
    paths = (paths or ProductPaths.default()).ensure()
    credentials = credential_store or CredentialStore()
    model_manager = retrieval_model_manager or RetrievalModelManager(paths.model_cache)

    def model_ready():
        """便宜的就绪判断：只读状态文件，不加载模型。"""
        return model_manager.status().get("status") == "ready"

    def encoder_factory():
        """在后台任务线程里加载编码器；这里绝不触发下载。"""
        if material_encoder_factory is not None:
            return material_encoder_factory()
        return build_cpu_encoder(paths.model_cache)

    @asynccontextmanager
    async def lifespan(app):
        app.state.database = Database(paths)
        app.state.paths = paths
        app.state.credentials = credentials
        app.state.retrieval_model = model_manager
        app.state.runtime_state = ({"status": "recovery_required", "detail": "数据库升级失败"}
                                   if app.state.database.migration_error else
                                   {"status": "not_configured", "detail": None})
        app.state.restore_pending = False
        if app.state.database.migration_error:
            app.state.materials = app.state.chat = app.state.organize = None
            app.state.support = app.state.studio = None
        else:
            app.state.materials = MaterialService(
                app.state.database, paths, run_inline=material_run_inline,
                settings_getter=lambda: current_settings(app.state.database),
                encoder_factory=encoder_factory, model_ready=model_ready)
            app.state.materials.ensure_vector_index()
            app.state.chat = ChatService(
                app.state.database, app.state.materials, credentials,
                lambda: current_settings(app.state.database), chat_client_factory,
                # 记忆提供者的注入点。不传就是没有记忆（`NullMemoryProvider`），
                # 出厂行为与"接缝不存在"逐字段一致。接入真实记忆系统时只改这里。
                memory=memory_provider_factory() if memory_provider_factory else None,
                # 搜索后端的注入点。不传就是没有联网（`NullSearchProvider`），
                # 此时即使把开关打开也不会发出任何请求。接入真实后端时只改这里。
                web=search_provider_factory() if search_provider_factory else None)
            app.state.organize = OrganizeService(app.state.database, paths)
            # 产出与问答共用同一个 `chat_client_factory`：用哪个模型只在一处决定，
            # 否则会出现"问答走 Ollama、产出偷偷走 DeepSeek"这种要翻配置才发现的事。
            app.state.studio = StudioService(
                app.state.database, paths, app.state.materials,
                lambda: current_settings(app.state.database), credentials, chat_client_factory)
            app.state.support = SupportService(
                app.state.database, paths, lambda: current_settings(app.state.database),
                credentials, lambda: app.state.runtime_state)
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
        if (getattr(request.app.state, "restore_pending", False)
                and request.url.path not in {"/api/v1/health", "/api/v1/system/shutdown"}):
            return JSONResponse({"error": "restart_required",
                                 "message": "数据已恢复，请退出并重新打开知识工作台。"},
                                status_code=409)
        database = getattr(request.app.state, "database", None)
        if (database and database.migration_error and request.url.path.startswith("/api/v1/")
                and request.url.path not in {"/api/v1/health", "/api/v1/setup",
                                             "/api/v1/system/recovery",
                                             "/api/v1/system/recovery/restore",
                                             "/api/v1/system/shutdown"}):
            return JSONResponse({"error": "database_recovery_required",
                                 "message": "数据库升级失败，请先从迁移备份恢复。"},
                                status_code=503)
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
        try:
            schema_version = database.schema_version()
        except Exception:
            schema_version = None
        return {
            "status": "recovery_required" if database.migration_error else "ready",
            "version": PRODUCT_VERSION,
            "database_schema": schema_version,
            "runtime": request.app.state.runtime_state,
            "restart_required": request.app.state.restore_pending,
        }

    @app.get("/api/v1/setup")
    async def setup(request: Request):
        database = request.app.state.database
        if database.migration_error:
            return {"recovery_required": True, "migration_error": "数据库升级未完成",
                    "recovery_backups": database.recovery_backups(), "data_root": str(paths.root),
                    "settings": DEFAULT_SETTINGS, "deepseek_key_configured": False,
                    "steps": {}, "retrieval_model": model_manager.status(),
                    "materials": {"total_documents": 0, "ready_documents": 0,
                                  "chunk_count": 0, "index_version": None}}
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
            "recovery_required": False,
        }

    @app.get("/api/v1/system/recovery")
    async def recovery_status(request: Request):
        database = request.app.state.database
        return {"required": bool(database.migration_error),
                "backups": database.recovery_backups()}

    @app.post("/api/v1/system/recovery/restore")
    async def restore_migration(body: MigrationRecoveryBody, request: Request):
        try:
            result = request.app.state.database.restore_migration_backup(body.backup_name)
            request.app.state.restore_pending = True
            return result
        except KeyError as error:
            return JSONResponse({"error": "backup_not_found", "message": str(error.args[0])},
                                status_code=404)
        except ValueError as error:
            return JSONResponse({"error": "invalid_backup", "message": str(error)},
                                status_code=422)

    @app.get("/api/v1/setup/retrieval-model")
    async def retrieval_model_status(request: Request):
        # 模型刚准备好时补建向量索引；已经建好或不是混合模式就什么都不做。
        request.app.state.materials.ensure_vector_index()
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
        # 首次开启联网必须先看过"什么会离开这台机器"。这条不做成界面文案而做成接口
        # 规则：文案会被改掉、会被跳过，规则不会。
        acknowledged = values.get("web_disclosure_acknowledged",
                                  current_settings(request.app.state.database)
                                  .get("web_disclosure_acknowledged"))
        if values.get("web_enabled") and not acknowledged:
            return JSONResponse({"error": "web_disclosure_required",
                                 "message": "开启联网前需先确认会发送的内容（只发送查询词）。"},
                                status_code=422)
        request.app.state.database.set_settings(values)
        request.app.state.database.event("settings_updated", {"keys": sorted(values)})
        # 检索参数只在建引擎时读一次，改完要就地换引擎；换的是同一份索引版本，
        # 已建好的向量索引不会因此作废。非检索参数的改动不碰引擎。
        if RETRIEVAL_PARAMETER_KEYS & set(values):
            request.app.state.materials.reload_retrieval_settings()
        # 切到混合检索时补建向量索引；其余情况这里什么都不做。
        request.app.state.materials.ensure_vector_index()
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

    @app.get("/api/v1/resources")
    async def resources_index():
        return {"docs": bundled_docs(), "examples": bundled_examples()}

    @app.get("/api/v1/resources/docs/{name}")
    async def bundled_document(name: str):
        path = resolve_bundled("docs", name)
        if path is None:
            return JSONResponse({"error": "resource_not_found", "message": "随包文档不存在。"},
                                status_code=404)
        return FileResponse(path, media_type="text/markdown; charset=utf-8",
                            filename=path.name, content_disposition_type="inline")

    @app.post("/api/v1/resources/import")
    async def import_bundled_example(body: ResourceImportBody, request: Request):
        try:
            return request.app.state.materials.import_bundled(body.name)
        except KeyError as error:
            return JSONResponse({"error": "resource_not_found", "message": str(error.args[0])},
                                status_code=404)
        except ValueError as error:
            return JSONResponse({"error": "invalid_document", "message": str(error)},
                                status_code=422)

    @app.get("/api/v1/system/folders")
    async def list_folders(path: str | None = Query(default=None, max_length=4096)):
        # 只列目录、只读，不写入也不删除；路径由前端逐级点选产生，
        # 不存在"用户提供的命令文本"这一类注入面。
        try:
            return await asyncio.to_thread(list_directory, path)
        except ValueError as error:
            return JSONResponse({"error": "invalid_folder", "message": str(error)},
                                status_code=422)

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

    @app.delete("/api/v1/libraries/{library_id}")
    async def remove_library(library_id: str, request: Request):
        try:
            return request.app.state.materials.remove_library(library_id)
        except KeyError as error:
            return JSONResponse({"error": "library_not_found", "message": str(error.args[0])},
                                status_code=404)
        except ValueError as error:
            return JSONResponse({"error": "library_not_removable", "message": str(error)},
                                status_code=422)

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

    @app.post("/api/v1/documents/delete")
    async def remove_documents(body: BatchDeleteBody, request: Request):
        """一次移除多篇资料。

        逐条调 `DELETE /api/v1/documents/{id}` 也能删掉同样多的东西，但每一条都会
        重建一次检索快照。走这一个入口，界面上的"选中 30 篇一起移除"才是一次
        事务、一次快照。
        """
        return request.app.state.materials.remove_documents(body.ids)

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
        try:
            return request.app.state.materials.search(q, top_k)
        except ValueError as error:
            # 候选窗口被实验性地调小到 top_k 以下时，这里会抛"要求 1 <= top_k
            # <= candidate_k"。那是设置问题不是服务故障，不能报成 500。
            return JSONResponse({"error": "invalid_search", "message": str(error)},
                                status_code=422)

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

    @app.delete("/api/v1/conversations/{conversation_id}")
    async def delete_conversation(conversation_id: str, request: Request):
        try:
            return request.app.state.chat.delete_conversation(conversation_id)
        except KeyError as error:
            return JSONResponse({"error": "conversation_not_found", "message": str(error.args[0])},
                                status_code=404)
        except RuntimeError as error:
            return JSONResponse({"error": "conversation_busy", "message": str(error)},
                                status_code=409)

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
            conversation_id, body.question, body.retry_message_id, cancel, body.skip_guard)

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

    @app.get("/api/v1/favorites")
    async def favorites(request: Request, library: str | None = Query(default=None, max_length=100),
                        tag: str | None = Query(default=None, max_length=100),
                        feedback: str | None = Query(default=None, max_length=40),
                        days: str | None = Query(default=None, max_length=10)):
        try:
            return request.app.state.organize.list_favorites(
                library=library, tag=tag, feedback=feedback, days=days)
        except ValueError as error:
            return JSONResponse({"error": "invalid_filter", "message": str(error)},
                                status_code=422)

    @app.post("/api/v1/favorites")
    async def create_favorite(body: FavoriteBody, request: Request):
        try:
            return request.app.state.organize.create_favorite(body.message_id)
        except KeyError as error:
            return JSONResponse({"error": "message_not_found", "message": str(error.args[0])},
                                status_code=404)
        except ValueError as error:
            return JSONResponse({"error": "favorite_unavailable", "message": str(error)},
                                status_code=409)

    @app.get("/api/v1/favorites/{favorite_id}")
    async def get_favorite(favorite_id: str, request: Request):
        try:
            return request.app.state.organize.get_favorite(favorite_id)
        except KeyError as error:
            return JSONResponse({"error": "favorite_not_found", "message": str(error.args[0])},
                                status_code=404)

    @app.patch("/api/v1/favorites/{favorite_id}")
    async def update_favorite(favorite_id: str, body: FavoritePatch, request: Request):
        try:
            return request.app.state.organize.update_favorite(
                favorite_id, body.title, body.note, body.tags)
        except KeyError as error:
            return JSONResponse({"error": "favorite_not_found", "message": str(error.args[0])},
                                status_code=404)
        except ValueError as error:
            return JSONResponse({"error": "invalid_favorite", "message": str(error)},
                                status_code=422)

    @app.post("/api/v1/favorites/delete")
    async def delete_favorites(body: BatchDeleteBody, request: Request):
        return request.app.state.organize.delete_favorites(body.ids)

    @app.delete("/api/v1/favorites/{favorite_id}")
    async def delete_favorite(favorite_id: str, request: Request):
        try:
            request.app.state.organize.delete_favorite(favorite_id)
            return {"deleted": True}
        except KeyError as error:
            return JSONResponse({"error": "favorite_not_found", "message": str(error.args[0])},
                                status_code=404)

    @app.get("/api/v1/favorites/collection/export")
    async def export_collection(request: Request, ids: str = Query(default="", max_length=8000),
                                title: str = Query(default="", max_length=100)):
        # 收哪些由界面决定（把当前筛出来的那批 id 原样传下来），服务端不重算筛选：
        # 这样"导出的就是眼前这些"是确定的，不必担心两次筛选之间资料又变了。
        try:
            path = request.app.state.organize.export_collection(
                title, [value for value in ids.split(",") if value])
            return FileResponse(path, media_type="text/markdown; charset=utf-8",
                                filename=path.name)
        except KeyError as error:
            return JSONResponse({"error": "favorite_not_found", "message": str(error.args[0])},
                                status_code=404)
        except ValueError as error:
            return JSONResponse({"error": "collection_unavailable", "message": str(error)},
                                status_code=409)

    @app.get("/api/v1/favorites/{favorite_id}/export")
    async def export_favorite(favorite_id: str, request: Request):
        try:
            path = request.app.state.organize.export_markdown(favorite_id)
            return FileResponse(path, media_type="text/markdown; charset=utf-8",
                                filename=path.name)
        except KeyError as error:
            return JSONResponse({"error": "favorite_not_found", "message": str(error.args[0])},
                                status_code=404)

    @app.post("/api/v1/feedback")
    async def save_feedback(body: FeedbackBody, request: Request):
        try:
            return request.app.state.organize.save_feedback(
                body.message_id, body.kind, body.note)
        except KeyError as error:
            return JSONResponse({"error": "message_not_found", "message": str(error.args[0])},
                                status_code=404)
        except ValueError as error:
            return JSONResponse({"error": "invalid_feedback", "message": str(error)},
                                status_code=422)

    @app.post("/api/v1/artifacts/stream")
    async def stream_artifact(body: ArtifactBody, request: Request):
        """产出一份指南或思维导图。

        复用问答那套流式通道（`application/x-ndjson`），所以前端只需要一套解析：
        `retrieval`（来源与编号）→ 若干 `token`（思维导图没有）→ `final`（带
        `backlink` 报告）。**不另起 job 机制**：`import_jobs` 是资料导入专用的，
        界面上的"正在处理"横幅读它，混进来会出现"正在导入 1 份资料"其实是在写指南。
        """
        service = request.app.state.studio
        try:
            service.validate_request(body.topic, body.kind)
        except ValueError as error:
            return JSONResponse({"error": "invalid_artifact", "message": str(error)},
                                status_code=422)
        cancel = threading.Event()
        iterator = service.stream(body.topic, body.kind, cancel)

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

    @app.get("/api/v1/artifacts")
    async def artifacts(request: Request):
        return {"artifacts": request.app.state.studio.list_artifacts()}

    @app.get("/api/v1/artifacts/{artifact_id}")
    async def get_artifact(artifact_id: str, request: Request):
        try:
            return request.app.state.studio.get_artifact(artifact_id)
        except KeyError as error:
            return JSONResponse({"error": "artifact_not_found", "message": str(error.args[0])},
                                status_code=404)

    @app.post("/api/v1/artifacts/delete")
    async def delete_artifacts(body: BatchDeleteBody, request: Request):
        """一次删掉多份产出，导出的图片跟着走。

        正在生成的那几份会被**跳过**并在 `skipped` 里说明原因，而不是连同删除：
        生成线程随后还会往来源表里写行，删掉主表会让它撞上外键约束、在流里抛
        异常（与单条删除拦住"还在生成"同一个理由）。
        """
        return request.app.state.studio.delete_artifacts(body.ids)

    @app.delete("/api/v1/artifacts/{artifact_id}")
    async def delete_artifact(artifact_id: str, request: Request):
        try:
            return request.app.state.studio.delete_artifact(artifact_id)
        except KeyError as error:
            return JSONResponse({"error": "artifact_not_found", "message": str(error.args[0])},
                                status_code=404)
        except RuntimeError as error:
            return JSONResponse({"error": "artifact_busy", "message": str(error)},
                                status_code=409)

    @app.post("/api/v1/artifacts/{artifact_id}/stop")
    async def stop_artifact(artifact_id: str, request: Request):
        try:
            return request.app.state.studio.stop(artifact_id)
        except KeyError as error:
            return JSONResponse({"error": "artifact_not_found", "message": str(error.args[0])},
                                status_code=404)

    @app.get("/api/v1/artifacts/{artifact_id}/infographic")
    async def infographic_export(artifact_id: str, request: Request):
        """上一次导出记录；从没导出过是 `export: null`，不是 404。"""
        try:
            return request.app.state.studio.infographic_export(artifact_id)
        except KeyError as error:
            return JSONResponse({"error": "artifact_not_found", "message": str(error.args[0])},
                                status_code=404)

    @app.get("/api/v1/artifacts/{artifact_id}/infographic.png")
    async def infographic_image(artifact_id: str, request: Request, download: int = 0):
        """已渲染的 PNG。默认**内联**返回（界面要拿它当 `<img>` 显示），`?download=1`
        才带 Content-Disposition——同一个文件的两个用法，不该靠两个接口表达。"""
        try:
            path = request.app.state.studio.infographic_file(artifact_id)
        except KeyError as error:
            return JSONResponse({"error": "infographic_missing", "message": str(error.args[0])},
                                status_code=404)
        if download:
            return FileResponse(path, media_type="image/png", filename=path.name)
        return FileResponse(path, media_type="image/png")

    @app.post("/api/v1/artifacts/{artifact_id}/infographic")
    async def export_infographic(artifact_id: str, request: Request):
        """把一份产出的来源画成 PNG（本机浏览器无头渲染）。

        **必须放线程池**：这里要起一个浏览器进程并等它出图（实测 0.8–2.3 秒），直接在
        事件循环里跑会把整个界面卡住。

        浏览器不可用**不是错误**：照样 200，带 `degraded: true` 与已导出的 HTML 路径
        ——图没出来，但用户还能自己打开那份 HTML。口径与 P2 的断网降级一致。
        """
        try:
            return await asyncio.to_thread(
                request.app.state.studio.export_infographic, artifact_id)
        except KeyError as error:
            return JSONResponse({"error": "artifact_not_found", "message": str(error.args[0])},
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
            return FileResponse(path, media_type=SOURCE_MEDIA_TYPES.get(
                                    media_type, "application/octet-stream"),
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
            "memory": memory_state(request.app.state.chat)
                      | {"enabled": bool(current_settings(request.app.state.database)
                                         .get("memory_enabled"))},
            "web": web_state(request.app.state.chat,
                             current_settings(request.app.state.database)),
            "studio": studio_state(request.app.state.database),
            "runtime": request.app.state.runtime_state,
            "retrieval_model": request.app.state.retrieval_model.status(),
        }

    @app.get("/api/v1/system/diagnostics/export")
    async def export_diagnostics(request: Request):
        path = request.app.state.support.write_diagnostic_report(
            request.app.state.retrieval_model.status(),
            request.app.state.materials.setup_summary())
        return FileResponse(path, media_type="application/json; charset=utf-8",
                            filename=path.name)

    @app.post("/api/v1/system/backup")
    async def create_backup(request: Request):
        path = await asyncio.to_thread(request.app.state.support.create_backup)
        return FileResponse(path, media_type="application/zip", filename=path.name)

    @app.post("/api/v1/system/restore")
    async def restore_backup(request: Request, file: UploadFile = File(...)):
        data = await file.read(MAX_BACKUP_BYTES + 1)
        try:
            result = await asyncio.to_thread(request.app.state.support.restore_backup, data)
            request.app.state.restore_pending = True
            return result
        except ValueError as error:
            return JSONResponse({"error": "invalid_backup", "message": str(error)},
                                status_code=422)
        finally:
            await file.close()

    @app.post("/api/v1/system/shutdown")
    async def shutdown(request: Request):
        """让启动器退出。

        **没有启动器时如实报错，而不是报成功。** 界面拿这个响应决定要不要显示"已退出"，
        所以在这里说谎会直接变成一句假话：用户看到"工作台已退出"，进程却还在跑。这与
        联网那一块"没有后端就说未配置、不说没搜到"是同一条线。

        退出**不删数据**，这句话由接口自己带出去而不是只写在前端文案里——它是这个动作
        唯一的对外承诺。
        """
        if not shutdown_callback:
            return JSONResponse(
                {"error": "shutdown_unavailable",
                 "message": "这个实例没有连接启动器，无法自行退出；请结束它的进程。"},
                status_code=503)
        # uvicorn 收到退出标志后会**等在跑的请求结束**才真正退出（流式生成的长流
        # 会拖住它）。这个等待是看不见的——不数清楚它，界面上的"已退出"就是一句
        # 假话。数据库起不来时服务根本不存在，等待清单自然是空的。
        state = request.app.state
        database = getattr(state, "database", None)
        waiting = {
            "imports": (state.materials.active_job_count()
                        if database is not None and not database.migration_error
                        and state.materials else 0),
            "answers": (state.chat.active_stream_count()
                        if database is not None and not database.migration_error
                        and state.chat else 0),
            "artifacts": (state.studio.active_count()
                          if database is not None and not database.migration_error
                          and state.studio else 0),
        }
        shutdown_callback()
        message = "本地服务正在退出。资料、索引和会话都会留在原处。"
        pending = sum(waiting.values())
        if pending:
            message += (f"正在等待 {pending} 个进行中的任务完成"
                        f"（导入 {waiting['imports']}、回答 {waiting['answers']}、"
                        f"产出 {waiting['artifacts']}），可能需要一点时间。")
        return {"status": "shutting_down", "waiting": waiting, "message": message}

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
