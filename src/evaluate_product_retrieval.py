"""在产品自己的检索路径上评测检索质量：两套判对口径 × BM25 / 混合 / 可加重排。

为什么需要这个入口：产品此前没有任何检索质量评测集。实验线有 dev 30 题和
holdout 12 题，但它们跑在 `src/serve.py` 那套编排上，量不到产品真正使用的那条
路径（`product/materials.py` 的快照 + 按版向量索引）。所以"混合检索要不要当默认"
一直缺一把尺子。

这个入口不调用生成模型，只测检索：把每个问题送进 `GET /api/v1/search`，拿回来的
片段按文档去重后排序，再与评测集比对。检索层能答对的问题，生成层才谈得上答对。

**两套口径**（同时记录，`--rubric` 只决定打印哪一张表）：

- **严格**：只认 `expected_source_files` 指向的那份文档，即"权威笔记有没有被顶到
  前面"。这贴近"点开引用要看到权威笔记"的产品目标。
- **宽松**：认**任何**一份全文覆盖该题全部 `expected_keywords` 的文档——它同样能
  回答这个问题，只是 ground truth 没收录它。这可以区分"检索排错了"与"排上来的
  是另一份合法资料"。

口径会改变结论，所以不能只留一套：实测 56 篇纯笔记语料上严格口径 hybrid 明显赢
（84.6% vs 69.2%），宽松口径却是平手（88.5% vs 88.5%）；101 篇完整语料上宽松口径
甚至 bm25 反超（76.9% vs 73.1%）。细节见 `docs/产品检索评测-2026-09-16.md` §11。

已知定义陷阱：部分题目的关键词是中英混排（如 `Prompt Injection` + `泄露`），
一份纯英文 PDF 永远覆盖不了中文关键词，所以宽松口径在跨语言对上天然偏严。

`--rerank` 会在同一批候选上追加一次本地 CrossEncoder 重排，用来回答"产品当前
没有重排，加回来能挽回多少名次"。注意这是**实验档**：产品本身不暴露重排开关，
这一档只在评测里存在。

用法：

    .\\.venv\\Scripts\\python.exe -X utf8 -m src.evaluate_product_retrieval ^
        --vault "C:\\Users\\zjm\\Documents\\Obsidian Vault\\10_技术学习\\概念笔记\\大模型"

首次运行要建索引并编码（当前 56 篇 / 911 片段约需几十秒），索引落在被 Git 忽略的
`data/generated/product_eval/`，之后重跑只走增量。全部离线：编码器与重排器都只读
本机缓存。

JSON 报告里逐题保存了每一路命中的原始 `channels` / `channel_scores` /
`matched_tokens` / `quality_reason`，因此"某道题为什么排成这样"可以离线复盘，
不必再跑一次索引。

holdout 集只做一次性对比记录，不据它调参——这是实验线既有的纪律，这里同样遵守。
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
from src.product.materials import RETRIEVAL_PARAMETERS
from src.product.retrieval_model import MODEL_NAME, MODEL_REVISION, MemoryRetrievalModelManager

MODES = ("bm25", "hybrid")
DEFAULT_SETS = ("data/eval_set.json", "data/eval_holdout.json")
# 每条条件开始前把五个检索参数显式写回默认值，保证条件之间互不污染。
DEFAULT_RETRIEVAL_PARAMETERS = {
    key: default for key, (default, _, _, _) in RETRIEVAL_PARAMETERS.items()}
# `/api/v1/search` 的 top_k 上限是 20；重排要在同一批候选上比，所以取满。
MAX_CANDIDATES = 20
# 片段命中要还原成"文档相对路径"才能和 expected_source_files 比；检索结果里带了
# document_id，这里由 /api/v1/documents 建一次映射。反查用于宽松口径：拿到路径后
# 要回到文档，才能判断"这份文档有没有覆盖全部期望关键词"。
DOCUMENT_PATHS: dict[str, str] = {}
DOCUMENT_IDS: dict[str, str] = {}


def build_eval_encoder(cache_dir: Path):
    """只读本机快照装编码器；不联网、不下载。"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL_NAME, revision=MODEL_REVISION, device="cpu",
        cache_folder=str(cache_dir), local_files_only=True,
    )


def build_eval_reranker():
    from src.reranker import LocalReranker

    return LocalReranker(device="cpu", allow_download=False)


def relative_path(hit: dict) -> str | None:
    return hit.get("relative_path") or DOCUMENT_PATHS.get(hit["document_id"])


def dedupe_paths(hits: list[dict]) -> list[str]:
    """按文档去重、保留首次出现的位置——名次就是"正确文档最早出现在第几位"。"""
    paths, seen = [], set()
    for hit in hits:
        path = relative_path(hit)
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def search_candidates(client, question: str, candidate_k: int) -> tuple[list[dict], str | None]:
    response = client.get("/api/v1/search", params={"q": question, "top_k": candidate_k})
    response.raise_for_status()
    payload = response.json()
    return payload["results"], payload.get("effective_mode")


def rerank_candidates(reranker, question: str, hits: list[dict], chunk_text: dict[str, str]) -> list[dict]:
    """用本地 CrossEncoder 对同一批候选重新排序。

    `search()` 会剥掉片段正文（`materials.py:813`），接口只给 360 字 `preview`。
    重排要读完整片段才有意义，所以正文从库里按 `chunk_id` 取——这也说明重排
    没法做成接口之外的一层，必须进服务内部。
    """
    chunks = [{"heading_path": hit["heading_path"],
               "text": chunk_text.get(hit["chunk_id"], hit.get("preview", ""))} for hit in hits]
    scored = reranker.score(question, chunks)
    order = sorted(range(len(hits)), key=lambda index: -scored[index][0])
    return [hits[index] for index in order]


def _announce(first: int | None) -> dict:
    """把"第一份合格文档排第几"折算成各项指标；两套口径共用同一套字段名。"""
    return {
        "hit": first is not None,
        "first_rank": first,
        "recall_at_1": bool(first == 1),
        "recall_at_5": bool(first is not None and first <= 5),
        "recall_at_8": bool(first is not None and first <= 8),
        "reciprocal_rank": 0.0 if first is None else 1.0 / first,
    }


def score_strict(expected: list[str], ranked: list[str]) -> dict:
    """严格口径：只认 `expected_source_files` 那份文档最早出现在第几位。

    字段名与改动前完全一致，好让 §3/§11 已记录的数字可以直接当回归锚点。
    """
    ranks = {path: index + 1 for index, path in enumerate(ranked)}
    hit_ranks = [ranks[path] for path in expected if path in ranks]
    return _announce(min(hit_ranks) if hit_ranks else None)


def covers_all(text_lower: str | None, keywords: list[str]) -> bool:
    """该文档全文是否覆盖全部期望关键词。

    用子串匹配：英文关键词（`Prompt Injection`、`Model Router`）要求整串出现，
    中文关键词（`可验证奖励`）要求连续出现。关键词里混排中英时，纯英文文档
    注定覆盖不全——这是口径本身的限制，不是实现缺陷。

    `text_lower` **必须是已转小写的全文**（`load_document_text` 的返回值就是这么
    给的）。这里不自己转：一篇文档最多会被拿来比对 8 次，语料规模下重复转小写
    是白白的几十 MB 拷贝。
    """
    if not text_lower:
        return False
    return all(keyword.lower() in text_lower for keyword in keywords)


def score_lenient(keywords: list[str], ranked: list[str],
                  document_text: dict[str, str]) -> dict:
    """宽松口径：不论来源，最早一份"全文能回答这道题"的文档排第几位。

    与严格口径唯一的差别是"哪份文档算对"：严格只认 ground truth 里那一份，
    宽松认任何覆盖了全部期望关键词的文档（Notebook、PDF、总览页都算）。
    """
    if not keywords:
        return _announce(None)
    for index, path in enumerate(ranked, 1):
        if covers_all(document_text.get(DOCUMENT_IDS.get(path, "")), keywords):
            return _announce(index)
    return _announce(None)


def summarize(rows: list[dict], key: str, require: str = "expected") -> dict:
    """`require` 决定拿哪些题当分母：严格口径按 expected，宽松口径按 keywords。"""
    scored = [row for row in rows if row[require]]
    if not scored:
        return {}
    count = len(scored)
    scores = [row[key] for row in scored]
    return {
        "questions": count,
        "source_recall@1": round(sum(score["recall_at_1"] for score in scores) / count, 4),
        "source_recall@5": round(sum(score["recall_at_5"] for score in scores) / count, 4),
        "source_recall@8": round(sum(score["recall_at_8"] for score in scores) / count, 4),
        "mrr": round(sum(score["reciprocal_rank"] for score in scores) / count, 4),
    }


def _better(base, other) -> bool:
    if base is None:
        return other is not None
    if other is None:
        return False
    return other < base


def compare(rows: list[dict], left_key: str, right_key: str, label: str) -> dict:
    changes = []
    for row in rows:
        base, other = row[left_key]["first_rank"], row[right_key]["first_rank"]
        if base == other:
            continue
        changes.append({"id": row["id"], "question": row["question"],
                        "before_rank": base, "after_rank": other,
                        "verdict": "better" if _better(base, other) else "worse"})
    return {"comparison": label, "improved": sum(1 for i in changes if i["verdict"] == "better"),
            "degraded": sum(1 for i in changes if i["verdict"] == "worse"),
            "changed_questions": changes}


def compact_hits(hits: list[dict]) -> list[dict]:
    """把接口返回的命中压成诊断用的瘦身版，随报告落盘，供离线复盘。"""
    return [{
        "rank": index,
        "relative_path": relative_path(hit),
        "chunk_id": hit["chunk_id"],
        "media_type": hit.get("media_type"),
        "heading_path": hit.get("heading_path"),
        "score": hit.get("score"),
        "channels": hit.get("channels"),
        "channel_scores": hit.get("channel_scores"),
        "matched_tokens": hit.get("matched_tokens"),
        "quality_reason": hit.get("quality_reason"),
        "preview": (hit.get("preview") or "")[:200],
    } for index, hit in enumerate(hits, 1)]


def parse_sweep(specs: list[str]) -> list[tuple[str, str, dict]]:
    """把 `方式:键=值` 解析成 (标签, 检索方式, 设置补丁)。

    写成 `hybrid:rrf_k=10` 而不是 `rrf_k=10` 是刻意的：同一个参数在不同检索方式下
    影响不同（`rrf_k` 只有混合用得上，`b`/`heading_repeat` 主要作用在 BM25 那一路），
    让每一项自己声明方式，报告里就不会出现"这个数是在哪条路径上得的"这种含糊。
    """
    entries = []
    for spec in specs:
        mode, separator, assignment = spec.partition(":")
        key, equals, raw = assignment.partition("=")
        if not separator or not equals or mode not in MODES:
            raise SystemExit(f"无法解析的实验项：{spec}；写法应为 方式:键=值，方式取 {MODES}")
        # 计划书里就写作 b 和 k1，这里认这两个短名，报告里仍用设置的全名。
        key = {"b": "bm25_b", "k1": "bm25_k1"}.get(key, key)
        if key not in RETRIEVAL_PARAMETERS:
            raise SystemExit(f"不是可实验的检索参数：{key}；可选 {sorted(RETRIEVAL_PARAMETERS)}")
        try:
            value = int(raw) if raw.lstrip("-").isdigit() else float(raw)
        except ValueError:
            raise SystemExit(f"实验项的值不是数字：{spec}")
        if any(existing[0] == spec for existing in entries):
            raise SystemExit(f"实验项重复：{spec}")
        entries.append((spec, mode, {key: value}))
    return entries


def build_conditions(sweep: list[tuple[str, str, dict]], rerank: bool) -> list[tuple[str, str, dict, bool]]:
    """条件表：(标签, 检索方式, 设置补丁, 是否重排)。

    基线是 bm25 / hybrid 两条；sweep 里的每一项各成一条；`--rerank` 时每条再各加
    一份重排档。标签直接带改动内容，报告里一眼能对上。
    """
    base = [(mode, mode, {}, False) for mode in MODES]
    base += [(label, mode, patch, False) for label, mode, patch in sweep]
    if rerank:
        base += [(f"{label}+rerank", mode, patch, True) for label, mode, patch, _ in base]
    return base


def evaluate_set(client, materials, name: str, items: list[dict],
                 top_k: int, candidate_k: int, reranker=None,
                 chunk_text: dict[str, str] | None = None,
                 document_text: dict[str, str] | None = None,
                 sweep: list[tuple[str, str, dict]] | None = None) -> dict:
    chunk_text = chunk_text or {}
    document_text = document_text or {}
    conditions = build_conditions(sweep or [], reranker is not None)
    labels = [label for label, _, _, _ in conditions]

    rows = []
    for item in items:
        expected = item.get("expected_source_files") or []
        keywords = item.get("expected_keywords") or []
        row = {"id": item["id"], "category": item["category"], "question": item["question"],
               "expected": expected, "keywords": keywords}
        for label, mode, patch, wants_rerank in conditions:
            # 每条条件都先回到默认参数再打补丁：否则上一条 sweep 改过的值会留到
            # 下一条，单变量实验就悄悄变成多变量了。
            client.patch("/api/v1/settings",
                         json={"retrieval_mode": mode, **DEFAULT_RETRIEVAL_PARAMETERS, **patch})
            if mode == "hybrid":
                prepare_hybrid(client, materials)
            hits, effective = search_candidates(client, item["question"], candidate_k)
            row[f"{label}_effective"] = effective
            row[f"{label}_hits"] = compact_hits(hits)
            ranked = dedupe_paths(hits)[:top_k]
            row[f"{label}_ranked"] = ranked
            row[f"{label}"] = score_strict(expected, ranked)
            row[f"{label}_lenient"] = score_lenient(keywords, ranked, document_text)
            if wants_rerank and reranker is not None:
                reordered = dedupe_paths(
                    rerank_candidates(reranker, item["question"], hits, chunk_text))[:top_k]
                row[f"{label}_ranked"] = reordered
                row[f"{label}"] = score_strict(expected, reordered)
        rows.append(row)

    entry = {"set": name, "top_k": top_k, "candidate_k": candidate_k, "conditions": {}}
    for label in labels:
        base_mode = label.split(":")[0].split("+")[0]
        entry["conditions"][label] = {
            "effective_mode": rows[0].get(f"{label}_effective") if rows else None,
            "summary": summarize(rows, label),
            "lenient_summary": summarize(rows, f"{label}_lenient", require="keywords"),
            "out_of_domain_zero_hit": sum(
                1 for row in rows if not row["expected"] and not row[f"{label}_ranked"]),
            "out_of_domain_total": sum(1 for row in rows if not row["expected"]),
        }
    entry["comparisons"] = [compare(rows, "bm25", "hybrid", "bm25 → hybrid")]
    if reranker is not None:
        entry["comparisons"] += [
            compare(rows, "bm25", "bm25+rerank", "bm25 → bm25+rerank"),
            compare(rows, "hybrid", "hybrid+rerank", "hybrid → hybrid+rerank"),
            compare(rows, "bm25", "hybrid+rerank", "bm25 → hybrid+rerank"),
        ]
    entry["rows"] = rows
    return entry


def prepare_hybrid(client, materials) -> None:
    """触发一次向量索引构建并等它完成。run_inline 下会在本线程直接跑完。"""
    for _ in range(600):
        state = materials._vector_state()
        if state["status"] == "ready":
            return
        materials.ensure_vector_index()
        time.sleep(0.5)
    raise RuntimeError("向量索引未能在预期时间内就绪，混合检索无法评测。")


def run(vault: Path, sets: list[Path], data_dir: Path, model_cache: Path,
        top_k: int, candidate_k: int, rerank: bool, rubric: str,
        sweep: list[tuple[str, str, dict]] | None = None) -> dict:
    paths = ProductPaths(data_dir).ensure()
    app = create_product_app(
        paths, MemoryCredentialStore(),
        retrieval_model_manager=MemoryRetrievalModelManager(status="ready"),
        material_run_inline=True,
        material_encoder_factory=lambda: build_eval_encoder(model_cache),
    )
    reranker = build_eval_reranker() if rerank else None
    result = {"vault": str(vault), "data_dir": str(data_dir), "top_k": top_k,
              "candidate_k": candidate_k, "rerank": rerank, "rubric": rubric,
              "sweep": [label for label, _, _ in sweep or []],
              "generated_at": datetime.now(timezone.utc).isoformat(), "sets": []}
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        started = time.perf_counter()
        connected = client.post("/api/v1/libraries/folders", json={"path": str(vault)})
        connected.raise_for_status()
        documents = client.get("/api/v1/documents").json()["documents"]
        DOCUMENT_PATHS.clear()
        DOCUMENT_IDS.clear()
        DOCUMENT_PATHS.update({item["id"]: item["relative_path"] for item in documents})
        DOCUMENT_IDS.update({item["relative_path"]: item["id"] for item in documents})
        result["documents"] = len(documents)
        result["ready_documents"] = sum(1 for item in documents if item["status"] == "ready")
        result["ingest_seconds"] = round(time.perf_counter() - started, 2)

        # 两份都必须在入库完成之后再读：这时 document_chunks 才写好。
        chunk_text = load_chunk_text(paths) if reranker is not None else {}
        document_text = load_document_text(paths)
        result["documents_with_text"] = len(document_text)
        for path in sets:
            items = json.loads(path.read_text(encoding="utf-8"))
            result["sets"].append(evaluate_set(
                client, app.state.materials, path.name, items, top_k, candidate_k,
                reranker, chunk_text, document_text, sweep))
    return result


def load_document_text(paths) -> dict[str, str]:
    """只读打开产品库，取回 `document_id` → 当前版本全文（小写）。

    只取 `documents.current_version_id` 指向的那一版：产品不回收旧版本的
    `document_chunks`（软删无清理），若把历史片段也拼进来，宽松口径会因为
    "某句已经删掉的话里出现过关键词"而虚高。
    """
    import sqlite3

    connection = sqlite3.connect(f"file:{paths.database}?mode=ro", uri=True)
    try:
        rows = connection.execute("""
            SELECT c.document_id, c.text
            FROM document_chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.version_id = d.current_version_id
            ORDER BY c.document_id, c.position
        """)
        parts: dict[str, list[str]] = {}
        for document_id, text in rows:
            parts.setdefault(document_id, []).append(text)
        return {document_id: "\n".join(texts).lower() for document_id, texts in parts.items()}
    finally:
        connection.close()


def load_chunk_text(paths) -> dict[str, str]:
    """只读打开产品库，取回 chunk_id → 完整片段正文（重排需要）。"""
    import sqlite3

    connection = sqlite3.connect(f"file:{paths.database}?mode=ro", uri=True)
    try:
        return {row[0]: row[1] for row in
                connection.execute("SELECT id, text FROM document_chunks")}
    finally:
        connection.close()


def _table(entry: dict, field: str, title: str) -> None:
    print(f"\n  [{title}]")
    print(f"  {'条件':<16}{'题数':>5}{'R@1':>9}{'R@5':>9}{'R@8':>9}{'MRR':>9}")
    for condition, block in entry["conditions"].items():
        summary = block[field]
        if not summary:
            continue
        print(f"  {condition:<16}{summary['questions']:>5}"
              f"{summary['source_recall@1']:>9.1%}{summary['source_recall@5']:>9.1%}"
              f"{summary['source_recall@8']:>9.1%}{summary['mrr']:>9.3f}")


def report(result: dict) -> None:
    rubric = result.get("rubric", "both")
    print(f"\n语料：{result['documents']} 篇（就绪 {result['ready_documents']}）"
          f"，入库 {result['ingest_seconds']}s，top_k={result['top_k']}"
          f"，候选 {result['candidate_k']}，重排={'开' if result['rerank'] else '关'}"
          f"，口径={rubric}")
    for entry in result["sets"]:
        print(f"\n=== {entry['set']} ===")
        if rubric in ("strict", "both"):
            _table(entry, "summary", "严格口径：只认 expected_source_files")
        if rubric in ("lenient", "both"):
            _table(entry, "lenient_summary", "宽松口径：认任何覆盖全部关键词的文档")
        for item in entry["comparisons"]:
            print(f"  {item['comparison']}：改善 {item['improved']}，退化 {item['degraded']}")
        for condition, block in entry["conditions"].items():
            total = block["out_of_domain_total"]
            if total:
                print(f"  知识库外未召回任何片段：{condition} "
                      f"{block['out_of_domain_zero_hit']}/{total}（仅记录，拒答由守卫负责）")


def main() -> None:
    parser = argparse.ArgumentParser(description="在产品检索路径上对比 BM25 / 混合 / 重排")
    parser.add_argument("--vault", required=True, help="Markdown 资料目录")
    parser.add_argument("--sets", nargs="*", default=list(DEFAULT_SETS), help="评测集 JSON 路径")
    parser.add_argument("--data-dir", default="data/generated/product_eval",
                        help="产品数据目录（被 Git 忽略；复用可走增量）")
    parser.add_argument("--model-cache", default=str(Path.home() / ".cache" / "huggingface" / "hub"),
                        help="E5 快照所在目录，只读不下载")
    parser.add_argument("--top-k", type=int, default=8, help="计入指标的名次深度")
    parser.add_argument("--candidate-k", type=int, default=MAX_CANDIDATES,
                        help=f"取回并参与重排的候选数（上限 {MAX_CANDIDATES}）")
    parser.add_argument("--rerank", action="store_true", help="追加本地 CrossEncoder 重排档")
    parser.add_argument("--sweep", nargs="*", default=[],
                        help="单变量实验项，写法 方式:键=值，可给多个；"
                             "如 hybrid:rrf_k=10  bm25:b=0.3  bm25:heading_repeat=0")
    parser.add_argument("--rubric", choices=("strict", "lenient", "both"), default="both",
                        help="打印哪一套判对口径；两套都会算好并存进报告，这里只控输出")
    parser.add_argument("--output", default=None, help="报告落点，默认 data/generated/product_retrieval_eval_<时间戳>.json")
    arguments = parser.parse_args()

    vault = Path(arguments.vault).expanduser()
    if not vault.is_dir():
        raise SystemExit(f"资料目录不存在：{vault}")
    candidate_k = min(arguments.candidate_k, MAX_CANDIDATES)

    result = run(vault, [Path(item) for item in arguments.sets], Path(arguments.data_dir),
                 Path(arguments.model_cache), arguments.top_k, candidate_k,
                 arguments.rerank, arguments.rubric, parse_sweep(arguments.sweep))
    report(result)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(arguments.output or f"data/generated/product_retrieval_eval_{stamp}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入：{output}")


if __name__ == "__main__":
    main()
