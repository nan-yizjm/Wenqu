"""核对完整训练与恢复训练的最终状态；只读本项目自己生成的 last.pt。"""

import argparse
from datetime import datetime
from pathlib import Path

import torch

from .config import PROJECT_ROOT
from .experiment_utils import save_report, sha256


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_state(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


def compare(left_path, right_path):
    left = torch.load(left_path, map_location="cpu", weights_only=True)
    right = torch.load(right_path, map_location="cpu", weights_only=True)
    if left.get("schema") != "tiny-lm-training-v1" or right.get("schema") != "tiny-lm-training-v1":
        raise ValueError("需要两个本项目生成的训练状态检查点 last.pt")
    checks = {key: equal_state(left[key], right[key]) for key in (
        "identity", "step", "model_state", "optimizer_state", "sample_rng", "cpu_rng", "cuda_rng")}
    comparable = (left["model_state"].keys() == right["model_state"].keys()
                  and all(left["model_state"][key].shape == right["model_state"][key].shape for key in left["model_state"]))
    maximum = max(float((left["model_state"][key] - right["model_state"][key]).abs().max())
                  for key in left["model_state"]) if comparable else None
    return {"left": str(Path(left_path).resolve()), "right": str(Path(right_path).resolve()),
            "left_sha256": sha256(Path(left_path)), "right_sha256": sha256(Path(right_path)),
            "left_step": left["step"], "right_step": right["step"], "checks": checks,
            "training_state_exact": all(checks.values()), "maximum_parameter_difference": maximum,
            "history_equal": equal_state(left["history"], right["history"]),
            "note": "这是同环境的实际核对，不是跨硬件/版本逐位一致性的保证"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    path = PROJECT_ROOT / "data/generated" / ("training_resume_comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    save_report(path, result)
    print("状态逐位一致：", result["training_state_exact"])
    print("各项检查：", result["checks"])
    print("最大参数差异：", result["maximum_parameter_difference"])
    print("验证记录相同：", result["history_equal"])
    print("报告：", path)
    if not result["training_state_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
