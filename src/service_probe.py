"""真实 HTTP 冒烟检查，只针对已启动的本机 Ollama 服务；报告不等于质量评测。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .config import PROJECT_ROOT
from .settings import DEFAULT_CONFIG, load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    settings = load_settings(args.config)
    base_url = f"http://{settings.host}:{settings.port}"
    records = []

    def call(name, path, expected, payload=None, headers=None, raw=None):
        data = raw if raw is not None else (json.dumps(payload, ensure_ascii=False).encode("utf-8")
                                           if payload is not None else None)
        request_headers = {"Content-Type": "application/json"} if data is not None else {}
        request_headers.update(headers or {})
        request = Request(base_url + path, data=data, headers=request_headers)
        started = time.perf_counter()
        try:
            response = urlopen(request, timeout=settings.request_timeout + 10)
        except HTTPError as error:
            response = error
        with response:
            body = json.loads(response.read().decode("utf-8"))
            status = response.status
            request_id = response.headers.get("X-Request-ID")
        record = {"name": name, "status": status, "expected": expected,
                  "passed": status == expected, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                  "request_id": request_id, "response": body}
        records.append(record)
        print(f"{name}: HTTP {status}，{record['elapsed_ms']} ms", flush=True)
        return body

    info = call("index", "/v1/index", 200)
    if info.get("provider") != "ollama":
        raise RuntimeError("此自动检查只允许本地 Ollama，避免误触发付费 API")
    call("live", "/health/live", 200)
    call("ready", "/health/ready", 200)
    for name, question in (("rag_question", "一个基础 RAG 系统从用户问题到最终回答通常经历哪些步骤？"),
                           ("pagedattention_question", "PagedAttention 是什么？它解决什么问题？"),
                           ("ood_guard", "北京明天天气怎么样？")):
        body = call(name, "/v1/answer", 200, {"question": question})
        records[-1]["passed"] &= body.get("index_version") == info.get("index_version")
        if name == "ood_guard":
            records[-1]["passed"] &= body.get("rejected") is True and body.get("generation_calls") == 0
    call("blank_question", "/v1/answer", 422, {"question": " "})
    call("unknown_parameter", "/v1/answer", 422, {"question": "RAG", "unexpected": 1})
    call("body_limit", "/v1/answer", 413, raw=b"x" * (settings.max_question_chars * 4 + 1025))
    call("cross_origin", "/v1/answer", 403, {"question": "RAG"}, headers={"Origin": "https://example.invalid"})
    report = {"created_at": datetime.now().astimezone().isoformat(), "service": info,
              "passed": sum(record["passed"] for record in records), "total": len(records), "records": records}
    output_dir = PROJECT_ROOT / "data/generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / ("service_probe_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"HTTP 合约检查：{report['passed']}/{report['total']}；报告：{output}")
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
