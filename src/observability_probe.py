"""验证 adapter 服务的分阶段耗时、聚合统计和不记录正文约束。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

from .config import PROJECT_ROOT
from .experiment_utils import sha256
from .settings import DEFAULT_CONFIG, load_settings


CASES = (
    ("ood", "北京明天天气怎么样？", True),
    ("single", "PagedAttention 是什么？它解决什么问题？", False),
    ("synthesis", "一个基础 RAG 系统从用户问题到最终回答通常经历哪些步骤？", False),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    settings = load_settings(args.config)
    base_url = f"http://{settings.host}:{settings.port}"

    def get(path):
        with urlopen(base_url + path, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def ask(name, question, expected_rejected):
        raw = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
        request = Request(base_url + "/v1/answer", data=raw,
                          headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        with urlopen(request, timeout=settings.request_timeout + 30) as response:
            body = json.loads(response.read().decode("utf-8"))
        return {
            "name": name,
            "question": question,
            "expected_rejected": expected_rejected,
            "status": response.status,
            "client_elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "body": body,
        }

    info = get("/v1/index")
    if info.get("provider") != "hf-qlora-experimental":
        raise RuntimeError("目标不是本机 adapter 实验服务")
    before = get("/v1/metrics")
    records = [ask(*case) for case in CASES]
    after = get("/v1/metrics")
    new_observations = (
        after["requests"]["observed_total"]
        - before["requests"]["observed_total"]
    )

    required_timing_fields = {
        "prepare_guard_ms", "prepare_total_ms", "generation_wall_ms",
        "finalize_ms", "runtime_total_ms", "queue_wait_ms", "service_elapsed_ms",
    }
    answers_match_behavior = all(
        record["status"] == 200
        and record["body"].get("rejected") == record["expected_rejected"]
        for record in records
    )
    timing_fields_complete = all(
        required_timing_fields <= set(record["body"].get("phase_timings_ms", {}))
        for record in records
    )
    hierarchy_consistent = all(
        timing["runtime_total_ms"] + 0.05 >= timing["prepare_total_ms"]
        and timing["service_elapsed_ms"] + 0.05 >= timing["runtime_total_ms"]
        for timing in (record["body"]["phase_timings_ms"] for record in records)
    )
    ood_zero_generation = (
        records[0]["body"]["generation_calls"] == 0
        and records[0]["body"]["phase_timings_ms"]["generation_wall_ms"] == 0.0
        and records[0]["body"].get("local_model_metrics") == {}
    )
    knowledge_generated = all(record["body"]["generation_calls"] == 1
                              for record in records[1:])
    metrics_text = json.dumps(after["requests"], ensure_ascii=False)
    metrics_omit_content = all(
        marker not in metrics_text
        for marker in ("北京", "PagedAttention", "基础 RAG", "结论：", "知识库资料")
    )
    latency_fields_aggregated = {
        "queue_wait_ms", "prepare_total_ms", "generation_wall_ms",
        "runtime_total_ms", "service_elapsed_ms",
    } <= set(after["requests"]["latency_ms"])
    generation_aggregate_excludes_ood = (
        after["requests"]["latency_ms"]["generation_wall_ms"]["count"] == 2
    )

    checks = {
        "answers_match_behavior": answers_match_behavior,
        "timing_fields_complete": timing_fields_complete,
        "hierarchy_consistent": hierarchy_consistent,
        "ood_zero_generation": ood_zero_generation,
        "knowledge_questions_generated": knowledge_generated,
        "three_new_observations": new_observations == len(CASES),
        "latency_fields_aggregated": latency_fields_aggregated,
        "generation_aggregate_excludes_ood": generation_aggregate_excludes_ood,
        "metrics_omit_question_answer_and_evidence": metrics_omit_content,
    }
    report = {
        "schema": "adapter-observability-probe-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "service": info,
        "records": records,
        "metrics_before": before,
        "metrics_after": after,
        "checks": checks,
        "passed": all(checks.values()),
        "source_signature": {
            "src/observability_probe.py": sha256(Path(__file__)),
            "src/answer.py": sha256(PROJECT_ROOT / "src/answer.py"),
            "src/adapter_runtime.py": sha256(PROJECT_ROOT / "src/adapter_runtime.py"),
            "src/serve.py": sha256(PROJECT_ROOT / "src/serve.py"),
        },
        "interpretation": (
            "prepare_total/generation/finalize 是可相加顶层阶段；prepare_* 子项属于"
            " prepare_total，不能再次与 total 相加。聚合端只保留最近 100 条数字诊断。"
        ),
    }
    output_dir = PROJECT_ROOT / "data/generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        "adapter_observability_probe_"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
    )
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    compact_records = [{
        "name": record["name"],
        "status": record["status"],
        "rejected": record["body"]["rejected"],
        "generation_calls": record["body"]["generation_calls"],
        "timings_ms": record["body"]["phase_timings_ms"],
    } for record in records]
    print(json.dumps({
        "records": compact_records,
        "aggregate": after["requests"],
        "checks": checks,
        "report": str(output),
    }, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
