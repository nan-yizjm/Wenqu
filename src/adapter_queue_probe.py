"""验证 adapter 服务的有界 FIFO、排队断线移除与队满背压。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import http.client
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .config import PROJECT_ROOT
from .experiment_utils import sha256
from .settings import DEFAULT_CONFIG, load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    settings = load_settings(args.config)
    base_url = f"http://{settings.host}:{settings.port}"

    def get(path):
        with urlopen(base_url + path, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def post_answer(name):
        data = json.dumps({"question": "PagedAttention 是什么？它解决什么问题？"},
                          ensure_ascii=False).encode("utf-8")
        started = time.perf_counter()
        request = Request(base_url + "/v1/answer", data=data,
                          headers={"Content-Type": "application/json"})
        try:
            response = urlopen(request, timeout=settings.request_timeout + 30)
        except HTTPError as error:
            response = error
        with response:
            body = json.loads(response.read().decode("utf-8"))
            status = response.status
        return {"name": name, "status": status,
                "client_elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "body": body}

    def wait_scheduler(predicate, timeout=5):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = get("/health/ready")["scheduler"]
            if predicate(last):
                return last
            time.sleep(0.02)
        raise RuntimeError(f"等待 scheduler 状态超时，最后状态：{last}")

    info = get("/v1/index")
    if info.get("provider") != "hf-qlora-experimental":
        raise RuntimeError("目标不是本机 adapter 实验服务")
    if info.get("scheduler", {}).get("max_waiting") != 2:
        raise RuntimeError("请使用 --queue-size 2 启动本实验")

    with ThreadPoolExecutor(max_workers=3) as pool:
        first_future = pool.submit(post_answer, "first")
        wait_scheduler(lambda state: state["busy"] and state["waiting"] == 0)

        # 建立一个请求但不等响应；确认入队后主动断开，服务应把它从 FIFO 移除。
        abandoned = http.client.HTTPConnection(settings.host, settings.port, timeout=5)
        raw = json.dumps({"question": "PagedAttention 是什么？"},
                         ensure_ascii=False).encode("utf-8")
        abandoned.request("POST", "/v1/answer", body=raw,
                          headers={"Content-Type": "application/json",
                                   "Content-Length": str(len(raw))})
        wait_scheduler(lambda state: state["waiting"] == 1)
        abandoned.close()
        after_abandon = wait_scheduler(
            lambda state: (state["busy"] and state["waiting"] == 0
                           and state["admitted_total"] == 1
                           and state["cancelled_waiting_total"] == 1)
        )

        second_future = pool.submit(post_answer, "second")
        wait_scheduler(lambda state: state["waiting"] == 1)
        third_future = pool.submit(post_answer, "third")
        full_state = wait_scheduler(lambda state: state["waiting"] == 2)
        rejected = post_answer("fourth")

        first = first_future.result(timeout=settings.request_timeout + 10)
        second = second_future.result(timeout=settings.request_timeout + 20)
        third = third_future.result(timeout=settings.request_timeout + 30)

    final_state = get("/health/ready")["scheduler"]
    admitted = [first, second, third]
    checks = {
        "abandoned_waiter_removed": (
            after_abandon["waiting"] == 0
            and after_abandon["cancelled_waiting_total"] == 1
            and after_abandon["admitted_total"] == 1
        ),
        "queue_reached_capacity": full_state["waiting"] == 2,
        "fourth_rejected_queue_full": (
            rejected["status"] == 429 and rejected["body"].get("error") == "queue_full"
        ),
        "three_requests_completed": all(record["status"] == 200 for record in admitted),
        "fifo_positions": [record["body"]["admission"]["initial_position"]
                           for record in admitted] == [0, 1, 2],
        "runtime_sequence_is_fifo": [record["body"].get("request_sequence")
                                     for record in admitted] == [1, 2, 3],
        "wait_time_increases": (
            first["body"]["admission"]["queue_wait_ms"]
            < second["body"]["admission"]["queue_wait_ms"]
            < third["body"]["admission"]["queue_wait_ms"]
        ),
        "scheduler_returns_idle": not final_state["busy"] and final_state["waiting"] == 0,
    }
    report = {
        "schema": "adapter-bounded-queue-probe-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "service": info,
        "states": {
            "after_abandoned_disconnect": after_abandon,
            "at_capacity": full_state,
            "final": final_state,
        },
        "records": {record["name"]: record for record in [*admitted, rejected]},
        "checks": checks,
        "passed": all(checks.values()),
        "source_signature": {
            "src/adapter_queue_probe.py": sha256(Path(__file__)),
            "src/serve.py": sha256(PROJECT_ROOT / "src/serve.py"),
            "src/adapter_runtime.py": sha256(PROJECT_ROOT / "src/adapter_runtime.py"),
        },
        "interpretation": (
            "max_waiting=2 表示一个任务运行、最多两个请求等待。队满立即 429；"
            "等待连接断开会删除 ticket；执行完成后按 FIFO 把唯一 GPU 席位转交队首。"
        ),
    }
    output_dir = PROJECT_ROOT / "data/generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        "adapter_queue_probe_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
    )
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    summary = [{"name": record["name"], "status": record["status"],
                "client_elapsed_ms": record["client_elapsed_ms"],
                "admission": record["body"].get("admission"),
                "error": record["body"].get("error")} for record in [*admitted, rejected]]
    print(json.dumps({"requests": summary, "final_scheduler": final_state,
                      "checks": checks, "report": str(output)}, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
