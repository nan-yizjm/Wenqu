"""验证本机 adapter NDJSON 流、客户端断开取消和取消后恢复。"""

import argparse
from datetime import datetime
import http.client
import json
from pathlib import Path
import time
from urllib.request import urlopen

from .config import PROJECT_ROOT
from .experiment_utils import sha256
from .settings import DEFAULT_CONFIG, load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    settings = load_settings(args.config)
    base_url = f"http://{settings.host}:{settings.port}"

    def index_info():
        with urlopen(base_url + "/v1/index", timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def open_stream(question):
        connection = http.client.HTTPConnection(
            settings.host, settings.port, timeout=settings.request_timeout + 10
        )
        body = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
        connection.request(
            "POST", "/v1/answer/stream", body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        response = connection.getresponse()
        if response.status != 200 or response.getheader("Content-Type") != "application/x-ndjson":
            error = response.read().decode("utf-8", errors="replace")
            connection.close()
            raise RuntimeError(f"流接口返回 HTTP {response.status}: {error}")
        return connection, response

    def read_complete(question):
        connection, response = open_stream(question)
        started = time.perf_counter()
        events = []
        first_token_ms = None
        try:
            while line := response.readline():
                event = json.loads(line.decode("utf-8"))
                events.append(event)
                if event.get("type") == "token" and first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
        finally:
            response.close()
            connection.close()
        return events, round((time.perf_counter() - started) * 1000, 2), first_token_ms

    info = index_info()
    if info.get("provider") != "hf-qlora-experimental":
        raise RuntimeError("目标不是本机 adapter 实验服务；停止以避免误调用其他后端")

    full_events, full_ms, full_first_token_ms = read_complete(
        "PagedAttention 是什么？它解决什么问题？"
    )
    full_final = full_events[-1]

    # 读到首个 token 后立即断开 TCP。取消状态无法再发给这个客户端，
    # 因此通过只读 /v1/index 的 last_request_summary 观察后台是否真实停止。
    connection, response = open_stream(
        "请详细说明一个基础 RAG 系统从问题到回答的全部步骤，并逐项解释。"
    )
    disconnected_events = []
    disconnect_started = time.perf_counter()
    try:
        while line := response.readline():
            event = json.loads(line.decode("utf-8"))
            disconnected_events.append(event)
            if event.get("type") == "token":
                break
    finally:
        response.close()
        connection.close()
    disconnected_after_ms = (time.perf_counter() - disconnect_started) * 1000

    cancel_summary = None
    ready_after_cancel = None
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        time.sleep(0.05)
        current = index_info()
        summary = current.get("last_request_summary") or {}
        if (summary.get("request_sequence", 0) > full_final.get("request_sequence", 0)
                and summary.get("mode") == "cancelled"):
            cancel_summary = summary
            with urlopen(base_url + "/health/ready", timeout=5) as ready_response:
                ready_after_cancel = json.loads(ready_response.read().decode("utf-8"))
            # 模型线程先记录 cancelled，ASGI 流的 finally 随后才释放 SingleFlight。
            # 两个条件都满足才可发送恢复请求，不能用固定 sleep 猜清理时机。
            if ready_after_cancel.get("busy") is False:
                break

    recovery_events, recovery_ms, recovery_first_token_ms = read_complete(
        "PagedAttention 是什么？它解决什么问题？"
    )
    recovery_final = recovery_events[-1]
    ood_events, ood_ms, _ = read_complete("北京明天天气怎么样？")
    ood_final = ood_events[-1]

    checks = {
        "full_protocol": (
            full_events[0].get("type") == "metadata"
            and any(event.get("type") == "token" for event in full_events)
            and full_final.get("type") == "final"
            and full_final.get("complete") is True
        ),
        "client_disconnected_after_token": (
            disconnected_events[-1].get("type") == "token"
        ),
        "backend_observed_cancel": cancel_summary is not None,
        "slot_released_after_cancel": (
            ready_after_cancel is not None and ready_after_cancel.get("busy") is False
        ),
        "recovery_complete": (
            recovery_final.get("type") == "final"
            and recovery_final.get("complete") is True
            and recovery_final.get("request_sequence", 0) > (cancel_summary or {}).get(
                "request_sequence", 0
            )
        ),
        "ood_final_without_tokens": (
            [event.get("type") for event in ood_events] == ["final"]
            and ood_final.get("rejected") is True
            and ood_final.get("generation_calls") == 0
            and ood_final.get("local_model_metrics") == {}
        ),
    }
    report = {
        "schema": "adapter-http-stream-probe-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "service": info,
        "measurements": {
            "full_stream_ms": full_ms,
            "full_first_token_ms": round(full_first_token_ms, 2),
            "disconnected_after_ms": round(disconnected_after_ms, 2),
            "cancel_summary": cancel_summary,
            "recovery_stream_ms": recovery_ms,
            "recovery_first_token_ms": round(recovery_first_token_ms, 2),
            "ood_stream_ms": ood_ms,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "events": {
            "full": full_events,
            "until_disconnect": disconnected_events,
            "recovery": recovery_events,
            "ood": ood_events,
        },
        "source_signature": {
            "src/adapter_stream_probe.py": sha256(Path(__file__)),
            "src/serve.py": sha256(PROJECT_ROOT / "src/serve.py"),
            "src/adapter_runtime.py": sha256(PROJECT_ROOT / "src/adapter_runtime.py"),
            "src/hf_qlora_client.py": sha256(PROJECT_ROOT / "src/hf_qlora_client.py"),
            "src/answer.py": sha256(PROJECT_ROOT / "src/answer.py"),
        },
        "interpretation": (
            "客户端断开后不能再收到 cancelled 事件；服务端 last_request_summary 记录模型实际"
            "观察取消并退出，ready.busy=false 证明 GPU 席位随后释放。"
        ),
    }
    output_dir = PROJECT_ROOT / "data/generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        "adapter_stream_probe_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
    )
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report["measurements"], ensure_ascii=False, indent=2))
    print(f"检查：{sum(checks.values())}/{len(checks)}；报告：{output}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
