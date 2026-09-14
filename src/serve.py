"""本机单进程 HTTP 服务：版本固定、有界输入、单个计算任务、结构化错误。"""

import argparse
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import logging
import queue
import threading
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .runtime import RAGRuntime
from .settings import DEFAULT_CONFIG, Settings, load_settings

LOG = logging.getLogger("rag.service")


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(strict=True, min_length=1, max_length=20000)

    @field_validator("question")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("question 不能为空")
        return value.strip()


class BodyLimitMiddleware:
    """先限制实际收到的字节数，而非相信客户端的 Content-Length。"""
    def __init__(self, app, max_bytes):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.max_bytes:
                return await JSONResponse({"error": "body_too_large"}, status_code=413)(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


class RecentRequestMetrics:
    """仅保存最近的数字诊断，不保留问题、回答或证据正文。"""

    FIELDS = (
        "queue_wait_ms", "prepare_total_ms", "prepare_retrieval_ms",
        "prepare_context_pack_ms", "prepare_token_count_ms",
        "generation_wall_ms", "finalize_ms", "runtime_total_ms",
        "service_elapsed_ms",
    )

    def __init__(self, capacity=100):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("metrics capacity 必须是正整数")
        self.capacity = capacity
        self.records = deque(maxlen=capacity)
        self.outcomes = {}
        self.total = 0
        self.lock = threading.Lock()

    def observe(self, outcome, status, timings=None):
        timings = timings or {}
        record = {"outcome": str(outcome), "status": int(status)}
        for field in self.FIELDS:
            value = timings.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                record[field] = round(float(value), 2)
        with self.lock:
            self.total += 1
            self.outcomes[record["outcome"]] = self.outcomes.get(record["outcome"], 0) + 1
            self.records.append(record)

    @staticmethod
    def _summary(values):
        if not values:
            return None
        ordered = sorted(values)
        def percentile(ratio):
            # nearest-rank：小样本也返回真实观测值，不插值制造不存在的耗时。
            index = max(0, int((len(ordered) * ratio + 0.999999)) - 1)
            return ordered[min(index, len(ordered) - 1)]
        return {
            "count": len(ordered),
            "avg": round(sum(ordered) / len(ordered), 2),
            "p50": round(percentile(0.50), 2),
            "p95": round(percentile(0.95), 2),
            "max": round(ordered[-1], 2),
        }

    def snapshot(self):
        with self.lock:
            records = list(self.records)
            total = self.total
            outcomes = dict(self.outcomes)
        return {
            "observed_total": total,
            "retained": len(records),
            "capacity": self.capacity,
            "outcomes": outcomes,
            "latency_ms": {
                field: summary
                for field in self.FIELDS
                if (summary := self._summary([
                    record[field] for record in records if field in record
                ])) is not None
            },
            "privacy": "仅保留数字耗时、状态和结果类别；不保留问题、回答或证据正文",
        }


class SingleFlight:
    """一个执行席位 + 可选有界 FIFO；模型真正结束前不会转交席位。"""
    def __init__(self, max_waiting=0):
        if type(max_waiting) is not int or not 0 <= max_waiting <= 100:
            raise ValueError("max_waiting 必须位于 0..100")
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.waiters = deque()
        self.active = None
        self.max_waiting = max_waiting
        self.admitted_total = 0
        self.rejected_total = 0
        self.queue_timeout_total = 0
        self.cancelled_waiting_total = 0
        self.cancelled_after_admission_total = 0
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-answer")

    class Ticket:
        def __init__(self, request_id):
            self.request_id = request_id
            self.created_at = time.perf_counter()
            self.admitted_at = None
            self.initial_position = None
            self.state = "new"
            self.reason = None
            self.event = threading.Event()

        def admission_info(self):
            wait_ms = ((self.admitted_at or time.perf_counter()) - self.created_at) * 1000
            return {"queued": bool(self.initial_position),
                    "initial_position": self.initial_position or 0,
                    "queue_wait_ms": round(wait_ms, 2)}

    def request_slot(self, request_id):
        ticket = self.Ticket(request_id)
        with self.state_lock:
            if self.active is None and not self.waiters:
                if not self.lock.acquire(blocking=False):
                    raise RuntimeError("scheduler 状态与执行锁不一致")
                ticket.state = "admitted"
                ticket.admitted_at = time.perf_counter()
                self.active = ticket
                self.admitted_total += 1
            elif self.max_waiting == 0:
                ticket.state, ticket.reason = "rejected", "busy"
                self.rejected_total += 1
            elif len(self.waiters) >= self.max_waiting:
                ticket.state, ticket.reason = "rejected", "queue_full"
                self.rejected_total += 1
            else:
                ticket.state = "waiting"
                ticket.initial_position = len(self.waiters) + 1
                self.waiters.append(ticket)
        return ticket

    def wait_for_slot(self, ticket, timeout):
        if ticket.state == "admitted":
            return True
        if ticket.state != "waiting":
            return False
        ticket.event.wait(timeout)
        with self.state_lock:
            if ticket.state == "admitted":
                return True
            if ticket.state == "waiting":
                self.waiters.remove(ticket)
                ticket.state, ticket.reason = "timeout", "queue_wait_timeout"
                self.queue_timeout_total += 1
            return False

    def cancel_ticket(self, ticket):
        with self.state_lock:
            if ticket.state == "waiting":
                self.waiters.remove(ticket)
                ticket.state, ticket.reason = "cancelled", "client_disconnected_while_queued"
                self.cancelled_waiting_total += 1
                ticket.event.set()
            elif ticket.state == "admitted" and self.active is ticket:
                ticket.state, ticket.reason = "cancelled", "cancelled_before_execution"
                self.cancelled_after_admission_total += 1
                self._release_locked(ticket)

    def _release_locked(self, ticket):
        if self.active is not ticket:
            raise RuntimeError("尝试释放非当前 scheduler ticket")
        self.active = None
        self.lock.release()
        while self.waiters:
            next_ticket = self.waiters.popleft()
            if next_ticket.state != "waiting":
                continue
            if not self.lock.acquire(blocking=False):
                raise RuntimeError("scheduler 无法把空闲执行锁交给队首")
            next_ticket.state = "admitted"
            next_ticket.admitted_at = time.perf_counter()
            self.active = next_ticket
            self.admitted_total += 1
            next_ticket.event.set()
            break

    def release(self, ticket):
        with self.state_lock:
            self._release_locked(ticket)

    def submit_admitted(self, ticket, function, *args):
        with self.state_lock:
            if ticket.state != "admitted" or self.active is not ticket:
                raise RuntimeError("只有当前 admitted ticket 可以提交任务")
        def run():
            try:
                return function(*args)
            finally:
                self.release(ticket)
        try:
            return self.pool.submit(run)
        except BaseException:
            self.release(ticket)
            raise

    def submit(self, function, *args):
        """保留旧的立即提交语义；排队功能由 HTTP endpoint 显式使用。"""
        ticket = self.request_slot("legacy")
        if ticket.state != "admitted":
            if ticket.state == "waiting":
                self.cancel_ticket(ticket)
            return None
        return self.submit_admitted(ticket, function, *args)

    def snapshot(self):
        with self.state_lock:
            return {"busy": self.active is not None,
                    "waiting": len(self.waiters), "max_waiting": self.max_waiting,
                    "admitted_total": self.admitted_total,
                    "rejected_total": self.rejected_total,
                    "queue_timeout_total": self.queue_timeout_total,
                    "cancelled_waiting_total": self.cancelled_waiting_total,
                    "cancelled_after_admission_total": self.cancelled_after_admission_total}

    def close(self):
        with self.state_lock:
            for ticket in self.waiters:
                ticket.state, ticket.reason = "cancelled", "service_shutdown"
                ticket.event.set()
            self.waiters.clear()
        self.pool.shutdown(wait=True, cancel_futures=False)


def create_app(settings: Settings | None = None, *, runtime_factory=RAGRuntime,
               max_queue_size=None, queue_wait_timeout=None) -> FastAPI:
    settings = settings or load_settings()
    max_queue_size = settings.max_queue_size if max_queue_size is None else max_queue_size
    queue_wait_timeout = (settings.queue_wait_timeout if queue_wait_timeout is None
                          else queue_wait_timeout)
    if type(max_queue_size) is not int or not 0 <= max_queue_size <= 100:
        raise ValueError("max_queue_size 必须位于 0..100")
    if not isinstance(queue_wait_timeout, (int, float)) or queue_wait_timeout <= 0:
        raise ValueError("queue_wait_timeout 必须为正数")

    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.runtime = await asyncio.to_thread(runtime_factory, settings)
        except Exception as error:
            LOG.error(json.dumps({"event": "startup_failed", "error_type": type(error).__name__}))
            raise RuntimeError("启动失败；检查索引、模型缓存与配置") from None
        app.state.flight = SingleFlight(max_waiting=max_queue_size)
        app.state.metrics = RecentRequestMetrics(capacity=100)
        LOG.info(json.dumps({"event": "ready", **app.state.runtime.info()}, ensure_ascii=False))
        try:
            yield
        finally:
            await asyncio.to_thread(app.state.flight.close)

    app = FastAPI(title="Obsidian RAG 本机服务", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_question_chars * 4 + 1024)

    @app.middleware("http")
    async def request_metadata(request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        host = request.headers.get("host", "")
        # bind 地址与客户端 Host 不是同一个概念。容器内监听 0.0.0.0，
        # 但 Compose 只发布宿主机 loopback，健康检查也使用 127.0.0.1。
        allowed_hosts = {
            f"127.0.0.1:{settings.port}", f"localhost:{settings.port}",
            f"{settings.host}:{settings.port}",
        }
        origin = request.headers.get("origin")
        if host not in allowed_hosts or (origin is not None and origin not in {"http://" + h for h in allowed_hosts}):
            response = JSONResponse({"error": "local_origin_required"}, status_code=403)
        elif request.method == "POST" and request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
            response = JSONResponse({"error": "json_required"}, status_code=415)
        else:
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["Cache-Control"] = "no-store"
        runtime = getattr(request.app.state, "runtime", None)
        LOG.info(json.dumps({"event": "request", "request_id": request_id,
                             "index_version": runtime.version if runtime else None,
                             "status": response.status_code,
                             "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}))
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        # 默认验证错误会回显 input；这里只返回字段位置与错误类型。
        return JSONResponse({"error": "invalid_request", "request_id": request.state.request_id,
                             "fields": [{"loc": e["loc"], "type": e["type"]} for e in error.errors()]},
                            status_code=422)

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready():
        runtime = app.state.runtime
        available = await asyncio.to_thread(runtime.backend_ready)
        runtime_provider = runtime.info().get("provider")
        if runtime_provider == "ollama":
            backend_check = "ollama_model_installed"
        elif runtime_provider == "deepseek":
            backend_check = "configuration_only_not_remote_probe"
        else:
            # 实验运行时负责定义自己的 ready 语义，避免沿用 rag.toml 中的
            # 默认 provider 后把本地 Hugging Face adapter 错报成 Ollama。
            backend_check = "runtime_self_check"
        scheduler = app.state.flight.snapshot()
        return JSONResponse({"ready": available, "index_version": runtime.version,
                             "busy": scheduler["busy"], "scheduler": scheduler,
                             "backend_check": backend_check},
                            status_code=200 if available else 503)

    @app.get("/v1/index")
    async def index_info():
        return {**app.state.runtime.info(), "scheduler": app.state.flight.snapshot()}

    @app.get("/v1/metrics")
    async def request_metrics():
        return {
            "index_version": app.state.runtime.version,
            "scheduler": app.state.flight.snapshot(),
            "requests": app.state.metrics.snapshot(),
        }

    async def acquire_slot(request, request_id):
        ticket = app.state.flight.request_slot(request_id)
        if ticket.state == "rejected":
            app.state.metrics.observe(ticket.reason, 429, {"queue_wait_ms": 0.0})
            response = JSONResponse({"error": ticket.reason, "request_id": request_id,
                                     "index_version": app.state.runtime.version}, status_code=429)
            response.headers["Retry-After"] = "2"
            return None, response
        if ticket.state == "waiting":
            waiting = asyncio.create_task(asyncio.to_thread(
                app.state.flight.wait_for_slot, ticket, queue_wait_timeout))

            async def wait_for_disconnect():
                while True:
                    message = await request.receive()
                    if message["type"] == "http.disconnect":
                        return True

            disconnecting = asyncio.create_task(wait_for_disconnect())
            try:
                done, _ = await asyncio.wait(
                    {waiting, disconnecting}, return_when=asyncio.FIRST_COMPLETED)
                if disconnecting in done:
                    app.state.flight.cancel_ticket(ticket)
                    await waiting
                    app.state.metrics.observe(
                        "client_disconnected_while_queued", 408,
                        {"queue_wait_ms": ticket.admission_info()["queue_wait_ms"]},
                    )
                    return None, JSONResponse({"error": "client_disconnected_while_queued",
                                               "request_id": request_id,
                                               "index_version": app.state.runtime.version},
                                              status_code=408)
                disconnecting.cancel()
                try:
                    await disconnecting
                except asyncio.CancelledError:
                    pass
                admitted = waiting.result()
            except asyncio.CancelledError:
                app.state.flight.cancel_ticket(ticket)
                waiting.cancel()
                disconnecting.cancel()
                raise
            if not admitted:
                app.state.metrics.observe(
                    ticket.reason, 503,
                    {"queue_wait_ms": ticket.admission_info()["queue_wait_ms"]},
                )
                response = JSONResponse({"error": ticket.reason, "request_id": request_id,
                                         "index_version": app.state.runtime.version}, status_code=503)
                response.headers["Retry-After"] = "2"
                return None, response
        return ticket, None

    @app.post("/v1/answer")
    async def answer(body: AnswerRequest, request: Request):
        endpoint_started = time.perf_counter()
        request_id = request.state.request_id
        runtime = app.state.runtime
        def failure(code, status):
            return JSONResponse({"error": code, "request_id": request_id,
                                 "index_version": runtime.version}, status_code=status)
        if len(body.question) > settings.max_question_chars:
            app.state.metrics.observe("question_too_long", 422)
            return failure("question_too_long", 422)
        ticket, admission_error = await acquire_slot(request, request_id)
        if admission_error is not None:
            return admission_error
        job = app.state.flight.submit_admitted(ticket, runtime.answer, body.question)
        pending = asyncio.wrap_future(job)
        # 即使请求已经超时，也消费后来完成的异常，避免无人处理的 Future 异常日志。
        pending.add_done_callback(lambda completed: completed.exception() if not completed.cancelled() else None)
        try:
            result = await asyncio.wait_for(asyncio.shield(pending), timeout=settings.request_timeout)
            admission = ticket.admission_info()
            timings = dict(result.get("phase_timings_ms", {}))
            timings["queue_wait_ms"] = admission["queue_wait_ms"]
            timings["service_elapsed_ms"] = round(
                (time.perf_counter() - endpoint_started) * 1000, 2
            )
            aggregate_timings = dict(timings)
            if not result.get("generation_calls"):
                aggregate_timings.pop("generation_wall_ms", None)
                aggregate_timings.pop("finalize_ms", None)
            app.state.metrics.observe("completed", 200, aggregate_timings)
            return {"request_id": request_id, "admission": admission,
                    **result, "phase_timings_ms": timings}
        except TimeoutError:
            app.state.metrics.observe(
                "answer_timeout", 504,
                {"queue_wait_ms": ticket.admission_info()["queue_wait_ms"],
                 "service_elapsed_ms": round(
                     (time.perf_counter() - endpoint_started) * 1000, 2)},
            )
            return failure("answer_timeout_backend_may_still_run", 504)
        except Exception as error:
            LOG.warning(json.dumps({"event": "answer_failed", "request_id": request_id,
                                    "error_type": type(error).__name__}))
            app.state.metrics.observe(
                "answer_failed", 502,
                {"queue_wait_ms": ticket.admission_info()["queue_wait_ms"],
                 "service_elapsed_ms": round(
                     (time.perf_counter() - endpoint_started) * 1000, 2)},
            )
            return failure("answer_failed_check_local_backend", 502)

    @app.post("/v1/answer/stream")
    async def stream_answer(body: AnswerRequest, request: Request):
        """NDJSON 流：metadata → token* → final/cancelled；只供支持流式的运行时。"""
        endpoint_started = time.perf_counter()
        request_id = request.state.request_id
        runtime = app.state.runtime
        if not hasattr(runtime, "stream_answer"):
            return JSONResponse({"error": "streaming_not_supported", "request_id": request_id,
                                 "index_version": runtime.version}, status_code=501)
        if len(body.question) > settings.max_question_chars:
            app.state.metrics.observe("question_too_long", 422)
            return JSONResponse({"error": "question_too_long", "request_id": request_id,
                                 "index_version": runtime.version}, status_code=422)
        ticket, admission_error = await acquire_slot(request, request_id)
        if admission_error is not None:
            return admission_error

        cancel_event = threading.Event()
        stream_queue = queue.Queue()
        end_marker = object()
        cancel_state = {"reason": None}
        observed_timings = {}
        stream_started = time.perf_counter()

        def produce():
            try:
                for event in runtime.stream_answer(body.question, cancel_event):
                    stream_queue.put(event)
            except BaseException as error:
                LOG.warning(json.dumps({"event": "stream_failed", "request_id": request_id,
                                        "error_type": type(error).__name__}))
                stream_queue.put({"type": "error", "complete": False,
                                  "error": "stream_failed_check_local_backend"})
            finally:
                stream_queue.put(end_marker)

        producer = threading.Thread(target=produce, name="rag-stream-producer", daemon=False)
        producer.start()

        async def ndjson():
            done = asyncio.Event()

            async def watch_cancel():
                deadline = asyncio.get_running_loop().time() + settings.request_timeout
                while not done.is_set():
                    if await request.is_disconnected():
                        cancel_state["reason"] = "client_disconnected"
                        cancel_event.set()
                        return
                    if asyncio.get_running_loop().time() >= deadline:
                        cancel_state["reason"] = "request_timeout"
                        cancel_event.set()
                        return
                    try:
                        await asyncio.wait_for(done.wait(), timeout=0.05)
                    except TimeoutError:
                        pass

            watcher = asyncio.create_task(watch_cancel())
            try:
                first_event = True
                while True:
                    event = await asyncio.to_thread(stream_queue.get)
                    if event is end_marker:
                        break
                    if event.get("type") == "cancelled":
                        event["reason"] = cancel_state["reason"] or "cancel_requested"
                    if event.get("type") in {"final", "cancelled", "error"}:
                        timings = dict(event.get("phase_timings_ms", {}))
                        timings["queue_wait_ms"] = ticket.admission_info()["queue_wait_ms"]
                        timings["service_elapsed_ms"] = round(
                            (time.perf_counter() - endpoint_started) * 1000, 2
                        )
                        event["phase_timings_ms"] = timings
                        observed_timings.update(timings)
                    if first_event:
                        event["admission"] = ticket.admission_info()
                        first_event = False
                    event["request_id"] = request_id
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            finally:
                # 断线或响应生成器被关闭时也会到这里。生产线程拥有模型迭代器；
                # 这里只发事件并等待它自行退出，避免跨线程 close 正在执行的 generator。
                if producer.is_alive() and cancel_state["reason"] is None:
                    # ASGI 发送任务可能先因断线被取消，watcher 尚未来得及观察 disconnect。
                    cancel_state["reason"] = "response_closed"
                cancel_event.set()
                done.set()
                watcher.cancel()
                # 当前 ASGI task 正是因断线而被取消时，finally 中的 await 也会立刻
                # 再次抛 CancelledError，使锁永远无法释放。这里最多同步等待 30 秒；
                # 本机单生成服务短暂阻塞事件循环，比错误允许第二个 GPU 任务重入安全。
                producer.join(timeout=30)
                if producer.is_alive():
                    LOG.error(json.dumps({"event": "stream_producer_stuck",
                                          "request_id": request_id}))
                    # 线程仍可能占用 GPU，不释放席位。
                else:
                    app.state.flight.release(ticket)
                summary = runtime.info().get("last_request_summary")
                if not observed_timings and isinstance(summary, dict):
                    observed_timings.update(summary.get("phase_timings_ms") or {})
                    observed_timings["queue_wait_ms"] = ticket.admission_info()["queue_wait_ms"]
                    observed_timings["service_elapsed_ms"] = round(
                        (time.perf_counter() - endpoint_started) * 1000, 2
                    )
                if isinstance(summary, dict) and summary.get("mode") == "cancelled":
                    outcome = "stream_cancelled"
                elif isinstance(summary, dict) and summary.get("mode") == "complete":
                    outcome = "stream_completed"
                else:
                    outcome = "stream_failed"
                aggregate_timings = dict(observed_timings)
                if isinstance(summary, dict) and not summary.get("generation_calls"):
                    aggregate_timings.pop("generation_wall_ms", None)
                    aggregate_timings.pop("finalize_ms", None)
                app.state.metrics.observe(outcome, 200, aggregate_timings)
                LOG.info(json.dumps({
                    "event": "stream_finished",
                    "request_id": request_id,
                    "cancel_reason": cancel_state["reason"],
                    "runtime_summary": summary,
                    "body_elapsed_ms": round((time.perf_counter() - stream_started) * 1000, 2),
                }, ensure_ascii=False))

        return StreamingResponse(
            ndjson(),
            media_type="application/x-ndjson",
            headers={"X-Accel-Buffering": "no"},
        )

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--bind-host", choices=("127.0.0.1", "0.0.0.0"), default=None,
        help="仅覆盖 socket 监听地址；0.0.0.0 供受控容器端口映射使用",
    )
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import uvicorn
    # 不开启 reload / 多 workers：每个进程各加载一套模型，容易耗尽 8GB 显存。
    uvicorn.run(create_app(settings), host=args.bind_host or settings.host, port=settings.port,
                workers=1, access_log=False, proxy_headers=False)


if __name__ == "__main__":
    main()
