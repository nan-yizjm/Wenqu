"""验证本地 QLoRA 的 token 流与模型级取消，并检查取消后可继续生成。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import threading
import time

from .adapter_rag import sft_messages
from .config import PROJECT_ROOT
from .experiment_utils import sha256
from .hf_qlora_client import HFQLoRAClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--cancel-after-ms", type=int, default=250)
    args = parser.parse_args()
    if not 10 <= args.cancel_after_ms <= 10000:
        raise ValueError("cancel-after-ms 必须位于 10..10000")

    client = HFQLoRAClient(args.adapter, max_new_tokens=args.max_new_tokens)
    evidence = (
        "[S1] 压测规范：输出应包含从 1 到 100 的编号项目，每项写一个完整句子。\n"
        "[S2] 取消测试：调用方可以在生成尚未完成时发出停止信号。"
    )
    messages = sft_messages("请严格依据证据输出从 1 到 100 的全部编号项目。", evidence)
    cancel = threading.Event()
    timer = threading.Timer(args.cancel_after_ms / 1000, cancel.set)
    started = time.perf_counter()
    timer.start()
    pieces = []
    try:
        pieces.extend(client.stream_chat(messages, cancel_event=cancel))
    finally:
        timer.cancel()
    cancel_elapsed_ms = (time.perf_counter() - started) * 1000
    cancel_metrics = dict(client.last_metrics)

    recovery_messages = sft_messages(
        "PagedAttention 是什么？",
        "[S1] PagedAttention 像操作系统分页一样管理 KV Cache，减少显存碎片。",
    )
    recovery_started = time.perf_counter()
    recovery_answer = client.chat(recovery_messages)
    recovery_elapsed_ms = (time.perf_counter() - recovery_started) * 1000
    recovery_metrics = dict(client.last_metrics)

    checks = {
        "cancel_event_set": cancel.is_set(),
        "model_observed_cancel": cancel_metrics.get("cancel_observed_by_model") is True,
        "worker_stopped": cancel_metrics.get("worker_alive_after_join") is False,
        "cancelled_before_max_tokens": (
            0 < cancel_metrics.get("generated_tokens", 0) < args.max_new_tokens
        ),
        "recovery_generated": bool(recovery_answer) and recovery_metrics.get("generated_tokens", 0) > 0,
        "recovery_not_streamed": recovery_metrics.get("streamed") is False,
    }
    report = {
        "schema": "hf-qlora-cancellation-lab-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "model": client.model,
        "adapter": str(client.adapter),
        "adapter_artifacts": client.adapter_artifacts,
        "parameters": {
            "max_new_tokens": args.max_new_tokens,
            "cancel_after_ms": args.cancel_after_ms,
        },
        "cancelled_run": {
            "visible_text": "".join(pieces),
            "visible_chunks": pieces,
            "elapsed_ms": round(cancel_elapsed_ms, 2),
            "metrics": cancel_metrics,
        },
        "recovery_run": {
            "answer": recovery_answer,
            "elapsed_ms": round(recovery_elapsed_ms, 2),
            "metrics": recovery_metrics,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "source_signature": {
            "src/cancellation_lab.py": sha256(Path(__file__)),
            "src/hf_qlora_client.py": sha256(PROJECT_ROOT / "src/hf_qlora_client.py"),
        },
        "interpretation": (
            "cancel_observed_by_model 只有在 StoppingCriteria 的 decode 检查实际读到事件时才为 true；"
            "随后恢复生成成功，说明生成线程和客户端锁均已真实释放。"
        ),
    }
    output_dir = PROJECT_ROOT / "data/generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        "cancellation_lab_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
    )
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        "cancelled_run": report["cancelled_run"],
        "recovery_run": report["recovery_run"],
        "checks": checks,
        "report": str(output),
    }, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
