"""在同一个 Qwen2.5 1.5B 基座上比较 QLoRA 前后的冻结挑战结果。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import torch

from .config import PROJECT_ROOT
from .evaluate_sft_challenge import DEFAULT_CHALLENGE, load_challenge, messages_for, score, summarize
from .experiment_utils import save_report, sha256, torch_memory
from .qlora_model import load_for_inference
from .sft_tokenization import MODEL_NAME, MODEL_REVISION


def adapter_artifacts(adapter):
    if not adapter:
        return None
    adapter = Path(adapter).resolve()
    names = ("adapter_config.json", "adapter_model.safetensors")
    missing = [name for name in names if not (adapter / name).is_file()]
    if missing:
        raise ValueError(f"adapter 缺少固定评估所需文件：{missing}")
    return {name: {"bytes": (adapter / name).stat().st_size, "sha256": sha256(adapter / name)}
            for name in names}


def run(challenge_path=DEFAULT_CHALLENGE, adapter=None, allow_download=False, output=None):
    challenge_path = Path(challenge_path).resolve()
    challenge = load_challenge(challenge_path)
    adapter = Path(adapter).resolve() if adapter else None
    adapter_hashes = adapter_artifacts(adapter)
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    model, tokenizer = load_for_inference(adapter=adapter, allow_download=allow_download)
    torch.cuda.reset_peak_memory_stats()
    records = []
    for index, case in enumerate(challenge["cases"], 1):
        started = time.perf_counter()
        try:
            inputs = tokenizer.apply_chat_template(
                messages_for(challenge, case), add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            input_length = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs, max_new_tokens=160, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
            answer = tokenizer.decode(output_ids[0, input_length:], skip_special_tokens=True).strip()
            checks, cited = score(case, answer)
            record = {
                "id": case["id"], "category": case["category"],
                "expected_behavior": case["expected_behavior"], "answer": answer,
                "target_response": case["target_response"], "cited_labels": cited,
                "checks": checks, "passed": checks["passed"],
                "input_tokens": input_length,
                "generated_tokens": output_ids.shape[-1] - input_length,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }
        except Exception as error:
            record = {
                "id": case["id"], "category": case["category"],
                "expected_behavior": case["expected_behavior"], "answer": "",
                "target_response": case["target_response"], "cited_labels": [], "checks": {},
                "passed": False, "elapsed_ms": (time.perf_counter() - started) * 1000,
                "error": f"{type(error).__name__}: {error}",
            }
        records.append(record)
        print(f"[{index}/{len(challenge['cases'])}] {case['id']}: {'PASS' if record['passed'] else 'FAIL'}", flush=True)
    summary = summarize(records)
    result = {
        "schema": "hf-rag-sft-challenge-report-v1",
        "role": "post_qlora" if adapter else "pre_qlora_same_base",
        "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
        "quantization": "bitsandbytes NF4 double quantization, BF16 compute",
        "adapter": str(adapter) if adapter else None,
        "adapter_artifacts": adapter_hashes,
        "challenge": str(challenge_path), "challenge_sha256": sha256(challenge_path),
        "evaluator_sha256": sha256(Path(__file__)),
        "summary": summary, "records": records, "torch_cuda_memory": torch_memory(),
        "generation": {"do_sample": False, "max_new_tokens": 160},
    }
    label = "adapter" if adapter else "base"
    output = Path(output).resolve() if output else PROJECT_ROOT / "data/generated" / (
        f"hf_sft_{label}_challenge_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    save_report(output, result)
    print("=" * 64)
    print(f"通过：{summary['passed']}/{summary['total']} ({summary['pass_rate']:.1%})")
    print(f"分类：{json.dumps(summary['by_category'], ensure_ascii=False)}")
    print(f"峰值 PyTorch 分配：{result['torch_cuda_memory']['peak_allocated_mib']:.1f} MiB")
    print(f"报告：{output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge", type=Path, default=DEFAULT_CHALLENGE)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.challenge, args.adapter, args.allow_download, args.output)


if __name__ == "__main__":
    main()
