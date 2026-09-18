"""量一量 Studio 产出的回链命中率：真模型、真语料、走产品真实的接口。

为什么需要这个入口：产出类功能最容易变成"看起来像那么回事"。`backlink_report`
给出了一个可数的数字，但**一个只有夹具在跑的指标没有意义**——夹具是我写的，
它当然会过。所以这里用本机模型（默认 Ollama `qwen2.5:7b`）在真实语料上跑一遍，
把数字落盘，供文档引用。

**这个数字是什么，不是什么**（写在最前面，因为它最容易被误读）：

- 它量的是**引用覆盖率**：一篇指南里有多少句陈述句挂上了 `[Sn]`。
- 它**不是**事实正确性，也**不是**"引用挂对了地方"。模型可以把 `[S1]` 挂在一句
  与之无关的话后面，覆盖率照样是 100%。本脚本无法发现这种情况，所以报告里
  不能把它写成"准确率"。
- 它依赖模型。"某模型 × 某语料 × N 个主题"下的一个数，不能外推。

统计口径：

- 聚合值是 `Σ带来源的句子 / Σ陈述句`（**按句子数加权**），不是"各主题命中率的
  平均值"。后者会让只写了 2 句的主题和写了 40 句的主题等权，把整体拉偏。
- 一句断言都没有的产出**不参与聚合**，单独记成 `topics_without_assertions`——
  空产出没有"全部带来源"这回事，混进分母会让结果虚高或虚低。
- 思维导图的覆盖是**构造性**的（节点取自片段标题层级），本脚本只记录它并标注
  `constructional`，不把它和指南的命中率并列比较。

用法（先按 `docs/检索评测复现指南.md` 的方式把产品跑起来，或直接用下面的内嵌方式）：

    .\\.venv\\Scripts\\python.exe -X utf8 -m src.evaluate_studio_backlinks ^
        --vault "C:\\Users\\zjm\\Desktop\\file\\项目\\obsidian-rag\\docs" ^
        --limit 4

不带 `--topics` 时主题取语料里的文件名（去掉扩展名）。那量的是"整理某一篇笔记"，
不是"综述某个概念"——文件名不等于主题，这个区别要在报告里写清楚。

索引落在被 Git 忽略的 `data/generated/studio_eval/`，重跑复用，不重新编码。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from src.product.app import create_product_app
from src.product.credentials import MemoryCredentialStore
from src.product.paths import ProductPaths
from src.product.retrieval_model import MemoryRetrievalModelManager

DEFAULT_DATA_DIR = "data/generated/studio_eval"
DEFAULT_OUT = "data/generated/studio_backlinks_{date}.json"


def read_topics(vault: Path, limit: int | None) -> list[str]:
    """没给主题时就用文件名（去掉扩展名）。

    这样跑出来的数字解释得清楚："拿这 N 篇笔记当主题，各写一篇指南"。它**不是**
    "概念综述"的成绩——文件名常常是编号或日期，不含概念本身。
    """
    stems = sorted(path.stem for path in vault.glob("*.md"))
    if not stems:
        raise SystemExit(f"目录里没有 Markdown：{vault}")
    return stems[:limit] if limit else stems


def measure(client, topic: str, kind: str) -> dict:
    """产出一份，返回这一份的回链记录。

    走的是产品真实的路由（`POST /api/v1/artifacts/stream`），所以量到的包含
    提示词、证据截断、模型、以及落库再读回来这一整条链路，而不是某个内部函数。
    """
    started = time.perf_counter()
    response = client.post("/api/v1/artifacts/stream", json={"topic": topic, "kind": kind})
    elapsed = round(time.perf_counter() - started, 2)
    if response.status_code != 200:
        return {"topic": topic, "kind": kind, "status": "route_failed",
                "http_status": response.status_code, "seconds": elapsed}
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    final = events[-1] if events else {}
    artifact_id = final.get("artifact_id")
    record = {"topic": topic, "kind": kind, "status": final.get("status", "unknown"),
              "artifact_id": artifact_id, "seconds": elapsed,
              "events": [event.get("type") for event in events]}
    if kind == "mindmap":
        mindmap = final.get("mindmap") or {}
        record["constructional"] = True
        record["node_count"] = mindmap.get("node_count")
        record["linked_chunks"] = mindmap.get("linked_chunks")
        record["note"] = "节点取自片段标题层级，覆盖率是构造结果，不是得分。"
        return record
    # 落库再读回来：流里的报告与读回来的报告**必须一致**，否则"刚生成"和"翻旧的"
    # 会显示两个不同的分数，而这种不一致比分数本身低更糟。
    stored = client.get(f"/api/v1/artifacts/{artifact_id}").json() if artifact_id else {}
    from_stream = final.get("backlink") or {}
    from_store = stored.get("backlink") or {}
    record.update({
        "assertions": from_stream.get("assertions", 0),
        "with_source": from_stream.get("with_source", 0),
        "hit_rate": from_stream.get("hit_rate"),
        "missing_count": from_stream.get("missing_count", 0),
        "invalid_labels": from_stream.get("invalid_labels", []),
        "report_matches_stored": from_stream == from_store,
        "sources": len(stored.get("sources", [])),
        "content_chars": len(stored.get("content", "")),
    })
    if not record["report_matches_stored"]:
        # 只记录差异，不抛异常：一次不一致说明有 bug，但不该让整轮评测中断。
        record["stored_hit_rate"] = from_store.get("hit_rate")
    return record


def summarize(records: list[dict]) -> dict:
    """按**句子数**加权聚合，并如实标出无法参与聚合的那些。"""
    guides = [item for item in records if item.get("kind") == "guide"]
    graded = [item for item in guides if item.get("assertions", 0) > 0]
    assertions = sum(item["assertions"] for item in graded)
    with_source = sum(item["with_source"] for item in graded)
    rates = sorted(item["with_source"] / item["assertions"] for item in graded)
    middle = len(rates) // 2
    median = (None if not rates else
              rates[middle] if len(rates) % 2 else (rates[middle - 1] + rates[middle]) / 2)
    return {
        "topics": len(records),
        "guides_graded": len(graded),
        # 空产出独立计数：它们既不进分子也不进分母。
        "topics_without_assertions": len(guides) - len(graded),
        "failed": sum(1 for item in records if item.get("status") in {"failed", "route_failed"}),
        "assertions": assertions,
        "with_source": with_source,
        "weighted_hit_rate": (with_source / assertions) if assertions else None,
        "median_hit_rate": median,
        "guides_at_full_coverage": sum(1 for item in graded if item["with_source"] == item["assertions"]),
        "invalid_labels_total": sum(len(item.get("invalid_labels", [])) for item in guides),
        "report_mismatches": sum(1 for item in graded if not item.get("report_matches_stored")),
    }


def run(vault: Path, topics: list[str], data_dir: Path, kinds: tuple[str, ...],
        topics_source: str = "vault filenames (stems)") -> dict:
    paths = ProductPaths(data_dir).ensure()
    app = create_product_app(
        paths, MemoryCredentialStore(),
        # 默认状态即可：检索走 bm25（产品默认），所以这一轮**不需要**任何编码器，
        # 也就不会在评测里触发模型下载。
        retrieval_model_manager=MemoryRetrievalModelManager(),
        material_run_inline=True,
    )
    report = {"vault": str(vault), "data_dir": str(data_dir),
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "kinds": list(kinds), "records": []}
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        connected = client.post("/api/v1/libraries/folders", json={"path": str(vault)})
        connected.raise_for_status()
        documents = client.get("/api/v1/documents").json()["documents"]
        setup = client.get("/api/v1/setup").json()
        settings = setup["settings"]
        report["corpus"] = {
            "documents": len(documents),
            "ready_documents": sum(1 for item in documents if item["status"] == "ready"),
            "chunks": setup["materials"]["chunk_count"],
        }
        report["model"] = {"provider": settings["provider"],
                           "model": (settings["ollama_model"] if settings["provider"] == "ollama"
                                     else settings.get("deepseek_model")),
                           "retrieval_mode": settings["retrieval_mode"],
                           # 主题从哪来的要如实记：文件名当主题量的是"整理某一篇"，与
                           # 手工挑的概念不是一回事，混起来会让数字没法解释。
                           "topics_source": topics_source}
        for topic in topics:
            for kind in kinds:
                record = measure(client, topic, kind)
                report["records"].append(record)
                print(f"  {kind:7s} {topic[:38]:38s} {record.get('status')}", flush=True)
        report["summary"] = summarize(report["records"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="量 Studio 产出的回链命中率")
    parser.add_argument("--vault", required=True, help="Markdown 资料目录")
    parser.add_argument("--topics", nargs="*", default=None, help="主题；不给则取文件名")
    parser.add_argument("--limit", type=int, default=4, help="不给主题时最多取几篇（默认 4）")
    parser.add_argument("--kind", choices=("guide", "mindmap", "both"), default="guide")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    vault = Path(arguments.vault).expanduser()
    if not vault.is_dir():
        raise SystemExit(f"资料目录不存在：{vault}")
    topics = arguments.topics or read_topics(vault, arguments.limit)
    kinds = ("guide", "mindmap") if arguments.kind == "both" else (arguments.kind,)
    print(f"语料：{vault}")
    print(f"主题 {len(topics)} 个 × {len(kinds)} 种产出：{topics}")
    report = run(vault, topics, Path(arguments.data_dir), kinds,
                 topics_source=("command line" if arguments.topics
                                else "vault filenames (stems)"))

    summary = report["summary"]
    print(f"\n语料：{report['corpus']}　模型：{report['model']}")
    print(f"参与统计的指南：{summary['guides_graded']} 篇"
          f"（另有 {summary['topics_without_assertions']} 篇没有可统计的句子、"
          f"{summary['failed']} 篇失败）")
    rate = summary["weighted_hit_rate"]
    median = summary["median_hit_rate"]
    print("回链命中率（Σ带来源 / Σ陈述句）："
          + ("无可统计" if rate is None else f"{rate:.1%}")
          + "　各主题命中率中位数：" + ("—" if median is None else f"{median:.1%}"))
    print(f"陈述句 {summary['assertions']} 句，其中 {summary['with_source']} 句带来源；"
          f"满覆盖率指南 {summary['guides_at_full_coverage']} 篇；"
          f"不存在的编号 {summary['invalid_labels_total']} 个；"
          f"流与落库不一致 {summary['report_mismatches']} 篇")

    out = Path(arguments.out) if arguments.out else Path(
        DEFAULT_OUT.format(date=datetime.now(timezone.utc).strftime("%Y%m%d")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告：{out}")


if __name__ == "__main__":
    main()
