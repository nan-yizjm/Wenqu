"""离线阅读生成报告，无需再次调用模型。"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="生成评测 JSON 文件的路径")
    parser.add_argument("--case", help="例如 gen-010；默认只列失败题")
    parser.add_argument("--context", action="store_true", help="显示本次实际上下文和检索轨迹")
    args = parser.parse_args()
    path = Path(args.report)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    report = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(report.get("summary", {}), ensure_ascii=False, indent=2))
    cases = [record for record in report["records"] if (
        record["id"] == args.case if args.case else not record["passed"]
    )]
    if not cases:
        print("没有匹配记录。")
    for case in cases:
        print(f"\n{'=' * 60}\n{case['id']} | {case['question']}\n\n{case['answer']}")
        print("检查：", json.dumps(case.get("checks", {}), ensure_ascii=False))
        print("模型调用次数：", case.get("generation_calls", "旧报告未记录"))
        print("逐项覆盖：", case.get("separate_coverage_ok"))
        if case.get("error"):
            print("运行错误：", case["error"])
        if args.context:
            print("\n上下文省略/去重诊断：\n", json.dumps(case.get("context_diagnostics", []), ensure_ascii=False, indent=2))
            print("\n实际上下文：\n", case.get("context_text", "旧报告未记录"))
            print("\n检索轨迹：\n", json.dumps(case.get("retrieval_trace", []), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
