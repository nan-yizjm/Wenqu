"""检查已启动的本机 adapter 服务，并比较首次与热请求；不调用付费 API。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

from .config import PROJECT_ROOT
from .experiment_utils import sha256
from .settings import DEFAULT_CONFIG, load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    settings = load_settings(config_path)
    base_url = f"http://{settings.host}:{settings.port}"
    records = []

    def call(name, path, payload=None):
        data = (json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if payload is not None else None)
        headers = {"Content-Type": "application/json"} if data is not None else {}
        started = time.perf_counter()
        with urlopen(Request(base_url + path, data=data, headers=headers),
                     timeout=settings.request_timeout + 10) as response:
            body = json.loads(response.read().decode("utf-8"))
        record = {
            "name": name,
            "http_elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "response": body,
        }
        records.append(record)
        print(f"{name}: {record['http_elapsed_ms']} ms", flush=True)
        return body

    info = call("index", "/v1/index")
    if info.get("provider") != "hf-qlora-experimental":
        raise RuntimeError("目标不是本机 adapter 实验服务；停止检查以避免误调用其他后端")
    ready = call("ready", "/health/ready")
    if not ready.get("ready") or ready.get("backend_check") != "runtime_self_check":
        raise RuntimeError("adapter 运行时未就绪")

    question = "PagedAttention 是什么？它解决什么问题？"
    first = call("first_answer", "/v1/answer", {"question": question})
    warm = call("warm_answer", "/v1/answer", {"question": question})
    ood = call("ood_guard", "/v1/answer", {"question": "北京明天天气怎么样？"})

    checks = {
        "same_index": all(item.get("index_version") == info.get("index_version")
                          for item in (first, warm, ood)),
        "sequence_increases": warm.get("request_sequence") == first.get("request_sequence", 0) + 1,
        "first_generated": first.get("generation_calls") == 1,
        "warm_generated": warm.get("generation_calls") == 1,
        "ood_rejected_without_generation": (
            ood.get("rejected") is True
            and ood.get("generation_calls") == 0
            and ood.get("local_model_metrics") == {}
        ),
    }
    report = {
        "schema": "adapter-service-probe-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "service": info,
        "measurements": {
            "startup_ms": info.get("startup_ms"),
            "first_request_ms": first.get("elapsed_ms"),
            "warm_request_ms": warm.get("elapsed_ms"),
            "first_generation_ms": first.get("local_model_metrics", {}).get("generation_ms"),
            "warm_generation_ms": warm.get("local_model_metrics", {}).get("generation_ms"),
            "ood_request_ms": ood.get("elapsed_ms"),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "records": records,
        "reproducibility": {
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "index_manifest_sha256": info.get("index_manifest_sha256"),
            "adapter_artifacts": info.get("adapter_artifacts"),
            "source_signature": {
                "src/adapter_runtime.py": sha256(PROJECT_ROOT / "src/adapter_runtime.py"),
                "src/adapter_service_probe.py": sha256(Path(__file__)),
                "src/serve_adapter.py": sha256(PROJECT_ROOT / "src/serve_adapter.py"),
                "src/serve.py": sha256(PROJECT_ROOT / "src/serve.py"),
                "src/hf_qlora_client.py": sha256(PROJECT_ROOT / "src/hf_qlora_client.py"),
            },
        },
        "interpretation": (
            "startup_ms 包含模型、adapter 与向量索引加载；两个回答请求都复用同一运行时。"
            "first_request 仍可能包含 CUDA kernel 首次执行开销，warm_request 更接近稳定热路径。"
        ),
    }
    output_dir = PROJECT_ROOT / "data/generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        "adapter_service_probe_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
    )
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report["measurements"], ensure_ascii=False, indent=2))
    print(f"检查：{sum(checks.values())}/{len(checks)}；报告：{output}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
