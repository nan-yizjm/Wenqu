"""混合检索在产品侧的接线：异步建索引、双通道命中、不可用时诚实降级。"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from fastapi.testclient import TestClient

from src.product.app import create_product_app
from src.product.credentials import MemoryCredentialStore
from src.product.paths import ProductPaths
from src.product.retrieval_model import MemoryRetrievalModelManager

# 维度取 2 就够了：这里要验的是"哪一路把片段捞回来的"，不是语义质量。
DIMENSION = 2
# 只在查询里出现的词。它和下面那个只在正文里出现的词指向同一个方向，
# 所以词法一路完全搜不到，向量一路必须能命中——两条通道的贡献才分得开。
QUERY_ONLY = "同义改写"
PASSAGE_ONLY = "近义表达"
# 让两段文档落在同一个方向但强弱不同，好验证名次而不是只验证"出现了"。
STRONG, WEAK = np.array([1.0, 0.0], dtype=np.float32), np.array([0.6, 0.8], dtype=np.float32)


class FakeEncoder:
    """把词面换成向量的内存编码器，不下载也不加载任何模型。"""

    def __init__(self):
        self.batches = []
        # 只触发一次的回调，用来在"编码进行中"插进一个真实的资料变更。
        self.on_encode = None

    def get_embedding_dimension(self):
        return DIMENSION

    def tokenizer(self, texts, **kwargs):
        # 截断计数只关心 token 数，这里按字符数粗略代替。
        return {"input_ids": [list(range(len(text))) for text in texts]}

    def encode(self, texts, **kwargs):
        single = isinstance(texts, str)
        self.batches.append([texts] if single else list(texts))
        hook, self.on_encode = self.on_encode, None
        if hook is not None:
            hook()
        rows = [self._vector(text) for text in ([texts] if single else texts)]
        return np.asarray(rows if not single else rows[0], dtype=np.float32)

    def encoded_passages(self):
        return [text for batch in self.batches for text in batch if "passage:" in text]

    def _vector(self, text):
        if QUERY_ONLY in text or PASSAGE_ONLY in text:
            return STRONG
        if "弱相关" in text:
            return WEAK
        return np.array([0.0, 1.0], dtype=np.float32)


class ProductMaterialsTestCase(unittest.TestCase):
    """产品资料链路的公共夹具。本身不含测试，供下面的用例类继承。"""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.paths = ProductPaths(self.root / "产品数据")
        self.encoder = FakeEncoder()
        app = create_product_app(
            self.paths, MemoryCredentialStore(),
            retrieval_model_manager=MemoryRetrievalModelManager(status="ready"),
            material_run_inline=True, material_encoder_factory=lambda: self.encoder,
        )
        self.app = app
        self.client = TestClient(app, base_url="http://127.0.0.1:8765")
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def upload(self, name, text):
        response = self.client.post(
            "/api/v1/documents/upload",
            files={"file": (name, text.encode("utf-8"), "text/markdown")})
        self.assertEqual(response.status_code, 200)
        return response.json()["document_id"]

    def search(self, query, **params):
        response = self.client.get("/api/v1/search", params={"q": query, **params})
        self.assertEqual(response.status_code, 200)
        return response.json()

    def mode(self, value):
        response = self.client.patch("/api/v1/settings", json={"retrieval_mode": value})
        self.assertEqual(response.status_code, 200)


class ProductHybridTests(ProductMaterialsTestCase):
    def test_hybrid_recalls_a_chunk_that_keyword_search_misses(self):
        """这是接通混合检索的全部意义：换个说法问，也能找到那段资料。"""
        self.upload("词法.md", "# 词法\n\n这里写的是检索词的原始表述。")
        self.upload("语义.md", f"# 语义\n\n这段话讲的是{PASSAGE_ONLY}，一个词都没重合。")

        keyword = self.search(QUERY_ONLY)
        self.assertEqual(keyword["effective_mode"], "bm25")
        self.assertEqual(keyword["results"], [])

        self.mode("hybrid")
        hybrid = self.search(QUERY_ONLY)
        self.assertEqual(hybrid["retrieval_mode"], "hybrid")
        self.assertEqual(hybrid["effective_mode"], "hybrid")
        hit, *rest = hybrid["results"]
        self.assertEqual(hit["title"], "语义.md")
        self.assertEqual(hit["channels"], {"vector": 1})
        self.assertAlmostEqual(hit["channel_scores"]["vector"], 1.0, places=5)
        # 向量一路没有分数下限，余弦为 0 的片段照样带着一个名次回来。这不是
        # 缺陷而是 RRF 的用法：无关片段的贡献只有 1/(60+名次)，排不到前面；
        # 真实模型上 E5 的余弦普遍在 0.7 附近，靠阈值裁剪反而会误杀。
        self.assertTrue(rest)
        self.assertTrue(all(item["score"] < hit["score"] for item in rest))

    def test_fused_ranking_reports_both_channels_for_a_shared_hit(self):
        """同一片段被两路都捞到时，界面要能看到"BM25 第几 / 向量 第几"。"""
        self.upload("命中.md", f"# 命中\n\n{QUERY_ONLY} 与 {PASSAGE_ONLY} 都在这一段里。")
        self.upload("弱相关.md", "# 弱相关\n\n只有一点点关系。")
        self.mode("hybrid")

        result = self.search(QUERY_ONLY)
        self.assertEqual(result["effective_mode"], "hybrid")
        top = result["results"][0]
        self.assertEqual(top["title"], "命中.md")
        self.assertEqual(top["channels"], {"bm25": 1, "vector": 1})
        self.assertEqual(sorted(top["channel_scores"]), ["bm25", "vector"])
        # 融合分是名次的函数，不该等于任何一路的原始分——量级不可比。
        self.assertNotIn(top["score"], top["channel_scores"].values())

    def test_hybrid_without_a_ready_model_degrades_and_says_so(self):
        """模型没准备好时不能假装开了混合检索，也不能让搜索直接不可用。"""
        app = create_product_app(
            self.paths, MemoryCredentialStore(),
            retrieval_model_manager=MemoryRetrievalModelManager(),
            material_run_inline=True, material_encoder_factory=lambda: self.encoder,
        )
        with TestClient(app, base_url="http://127.0.0.1:8765") as client:
            self.assertEqual(client.post("/api/v1/documents/upload", files={
                "file": ("笔记.md", "# 笔记\n\nPagedAttention 管理 KV Cache。".encode("utf-8"),
                         "text/markdown")}).status_code, 200)
            client.patch("/api/v1/settings", json={"retrieval_mode": "hybrid"})
            result = client.get("/api/v1/search", params={"q": "PagedAttention"}).json()
            self.assertEqual(result["retrieval_mode"], "hybrid")
            self.assertEqual(result["effective_mode"], "bm25")
            self.assertTrue(result["results"])
            self.assertEqual(client.get("/api/v1/setup").json()["materials"]["vector_index"],
                             {"status": "off", "version_id": result["index_version"]})

    def test_a_new_index_version_drops_the_old_vector_index(self):
        """资料一变就换了索引版本；旧向量对不上新片段，必须重建而不是硬用。"""
        self.upload("一.md", "# 一\n\n一开始只有这一份资料。")
        self.mode("hybrid")
        first = self.search(QUERY_ONLY)
        self.assertEqual(first["effective_mode"], "hybrid")
        self.assertNotEqual(first["results"][0]["title"], "二.md")

        self.upload("二.md", f"# 二\n\n后来又补了一段讲{PASSAGE_ONLY}的资料。")
        second = self.search(QUERY_ONLY)
        self.assertEqual(second["effective_mode"], "hybrid")
        # 新片段必须在新的向量矩阵里找得到；还在用旧矩阵的话，它一个名次都拿不到。
        self.assertEqual(second["results"][0]["title"], "二.md")
        self.assertNotEqual(second["index_version"], first["index_version"])
        # 旧版本的缓存留在原处（增量复用还要读它），当前版本也必须有一份自己的。
        cached = {path.parent.name for path in self.paths.indexes.glob("*/vector_index.npz")}
        self.assertEqual(cached, {first["index_version"], second["index_version"]})
        # 增量复用真的生效：整库指纹一变就全量失效，只有按文本命中的复用能
        # 把编码量压到新增片段数。这里两版共编两次，不是三次。
        self.assertEqual(len(self.encoder.encoded_passages()), 2)

    def test_vector_index_job_is_recorded_and_does_not_leak_into_source_text(self):
        """建索引是后台任务，用户在任务列表里能看到它，而不是让页面卡住。"""
        self.upload("笔记.md", "# 笔记\n\nPagedAttention 管理 KV Cache。")
        self.mode("hybrid")
        self.search("PagedAttention")

        jobs = self.client.get("/api/v1/import-jobs").json()["jobs"]
        build = next(job for job in jobs if job["job_type"] == "build_vector_index")
        self.assertEqual(build["status"], "completed")
        self.assertEqual(json.loads(build["payload_json"])["chunks"], 1)

        hit = self.search("PagedAttention")["results"][0]
        source = self.client.get(
            f"/api/v1/documents/{hit['document_id']}/versions/{hit['version_id']}/source").json()
        self.assertEqual(source["text"].strip(), "# 笔记\n\nPagedAttention 管理 KV Cache。")

    def test_vector_cache_lives_under_the_index_version(self):
        """缓存按索引版本分目录；混用一个文件会让增量复用认错旧矩阵。"""
        self.upload("笔记.md", "# 笔记\n\nPagedAttention 管理 KV Cache。")
        self.mode("hybrid")
        version_id = self.search("PagedAttention")["index_version"]
        cached = self.paths.indexes / version_id / "vector_index.npz"
        self.assertTrue(cached.is_file())
        self.assertEqual(json.loads(
            str(np.load(cached, allow_pickle=False)["metadata"].item()))["dimension"], DIMENSION)
        self.assertEqual(len(list(self.paths.indexes.glob("*/vector_index.npz"))), 1)

    def test_a_build_superseded_midway_reports_success_and_redoes_the_current_version(self):
        """编码要几分钟，期间用户又加了资料是常事——那不是失败，是改做当前版本。

        把它记成失败，任务列表里就会出现一排红色记录，让人以为建索引本身出了
        问题；真正该做的是收尾时为新版本重排一次。这里让编码器在编码途中插进
        一份新资料，把"建到一半资料变了"这个竞态直接构造出来。
        """
        self.mode("hybrid")
        self.upload("一.md", "# 一\n\n最早的资料。")
        # 走服务方法而不是 HTTP：编码就在请求线程里跑，从那里再发一次同步请求
        # 会被 TestClient 挡下来（"cannot be called from the event loop thread"）。
        materials = self.client.app.state.materials
        self.encoder.on_encode = lambda: materials.upload(
            "三.md", f"# 三\n\n后来的资料，讲到{PASSAGE_ONLY}。".encode("utf-8"))
        self.upload("二.md", "# 二\n\n中间的资料。")

        builds = [job for job in self.client.get("/api/v1/import-jobs").json()["jobs"]
                  if job["job_type"] == "build_vector_index"]
        superseded = [job for job in builds if "作废" in (job["message"] or "")]
        self.assertEqual(len(superseded), 1)
        self.assertEqual(superseded[0]["status"], "completed")
        self.assertEqual(superseded[0]["failed"], 0)
        # 作废之后必须自动补上当前版本，而不是留下一个没有语义索引的工作台。
        current = self.search(QUERY_ONLY)["index_version"]
        self.assertEqual(self.client.get(
            "/api/v1/setup").json()["materials"]["vector_index"],
            {"status": "ready", "version_id": current, "truncated_chunks": 0})
        self.assertEqual(self.search(QUERY_ONLY)["effective_mode"], "hybrid")
        # 补建出来的那一版必须包含编码途中新加的资料，否则等于白建。
        self.assertEqual(self.search(QUERY_ONLY)["results"][0]["title"], "三.md")

    def test_over_long_chunks_are_reported_instead_of_silently_truncated(self):
        """片段超出模型长度上限时，语义那一路只看得到前半段，名次会失真。

        实验线遇到这种情况直接拒绝发布索引；产品这边只记录并说明——向量索引
        是辅助通道，为它拒绝服务会把关键词检索一起挡掉。所以这里验证两件事：
        状态里报得出这个数量，任务消息里也说得出原因。
        """
        # 段落超过 512 个字符就会超出 FakeEncoder 的 max_seq_length（按 token 数
        # 恒等于字符数），但仍在 800 字符的切片上限之内，所以它还是单个片段。
        self.upload("长片段.md", f"# 长片段\n\n{'长' * 520}\n\nPagedAttention 管 KV Cache。")
        self.mode("hybrid")
        result = self.search("PagedAttention")
        self.assertEqual(result["effective_mode"], "hybrid")

        state = self.client.get("/api/v1/setup").json()["materials"]["vector_index"]
        self.assertEqual(state["status"], "ready")
        self.assertEqual(state["truncated_chunks"], 1)

        build = next(job for job in self.client.get("/api/v1/import-jobs").json()["jobs"]
                     if job["job_type"] == "build_vector_index")
        self.assertEqual(build["status"], "completed")
        self.assertIn("长度上限", build["message"] or "")
        # 超限只是被记下来，不该让这次构建失败——否则用户会以为索引坏了。
        self.assertEqual(build["failed"], 0)

    def test_a_clean_index_reports_zero_truncated_chunks(self):
        """没有超限片段时也要明确报 0，别让界面把"没报"当成"没检查"。"""
        self.upload("短片段.md", "# 短片段\n\nPagedAttention 管理 KV Cache。")
        self.mode("hybrid")
        self.search("PagedAttention")

        state = self.client.get("/api/v1/setup").json()["materials"]["vector_index"]
        # 就绪状态固定是这三个键：`off`/`building` 不带计数，因为"没有索引"和
        # "索引干净"是两回事，报成 0 会被读成"检查过了，没问题"。
        self.assertEqual(sorted(state), ["status", "truncated_chunks", "version_id"])
        self.assertEqual(state["status"], "ready")
        self.assertEqual(state["truncated_chunks"], 0)
        build = next(job for job in self.client.get("/api/v1/import-jobs").json()["jobs"]
                     if job["job_type"] == "build_vector_index")
        self.assertIsNone(build["message"])


class ProductRetrievalParameterTests(ProductMaterialsTestCase):
    """检索参数只用于单变量实验：改得动，但默认值必须还是历史行为。"""

    def engine(self):
        return self.app.state.materials._snapshot.engine

    def build_jobs(self):
        return [job for job in self.client.get("/api/v1/import-jobs").json()["jobs"]
                if job["job_type"] == "build_vector_index"]

    def test_defaults_are_the_historical_values(self):
        settings = self.client.get("/api/v1/settings").json()["settings"]
        for key, expected in (("candidate_k", 20), ("rrf_k", 60), ("bm25_k1", 1.5),
                              ("bm25_b", 0.75), ("heading_repeat", 1)):
            self.assertEqual(settings[key], expected, key)

        engine = self.engine()
        self.assertEqual(engine.candidate_k, 20)
        self.assertEqual(engine.rrf_k, 60)
        self.assertEqual(engine.bm25.k1, 1.5)
        self.assertEqual(engine.bm25.b, 0.75)
        self.assertEqual(engine.bm25.heading_repeat, 1)

    def test_a_parameter_change_reaches_the_engine(self):
        response = self.client.patch("/api/v1/settings", json={
            "rrf_k": 10, "bm25_b": 0.3, "bm25_k1": 2.0, "heading_repeat": 0})
        self.assertEqual(response.status_code, 200)

        engine = self.engine()
        self.assertEqual(engine.rrf_k, 10)
        self.assertEqual(engine.bm25.b, 0.3)
        self.assertEqual(engine.bm25.k1, 2.0)
        self.assertEqual(engine.bm25.heading_repeat, 0)

    def test_changing_a_parameter_does_not_throw_away_the_vector_index(self):
        """就地换引擎的目的：换参数不该让已经编码好的向量索引作废重来。"""
        self.upload("词法.md", "# 词法\n\n这里写的是检索词的原始表述。")
        self.upload("语义.md", f"# 语义\n\n这段话讲的是{PASSAGE_ONLY}，一个词都没重合。")
        self.mode("hybrid")
        self.assertEqual(self.search(QUERY_ONLY)["effective_mode"], "hybrid")
        before = len(self.build_jobs())

        self.client.patch("/api/v1/settings", json={"rrf_k": 10})

        # 混合检索仍然生效，说明向量索引还挂在新的那份引擎上。
        self.assertEqual(self.search(QUERY_ONLY)["effective_mode"], "hybrid")
        self.assertEqual(len(self.build_jobs()), before)

    def test_impossible_values_are_rejected_at_the_boundary(self):
        for payload in ({"candidate_k": 0}, {"rrf_k": 0}, {"bm25_k1": 0},
                        {"bm25_b": 1.5}, {"bm25_b": -0.1}, {"heading_repeat": -1}):
            with self.subTest(**payload):
                self.assertEqual(
                    self.client.patch("/api/v1/settings", json=payload).status_code, 422)

    def test_a_value_stored_behind_the_api_still_falls_back_to_the_default(self):
        """有人直接改库（或旧版本写脏了）时，检索不能因此崩掉。"""
        self.app.state.database.set_settings({"bm25_b": 5, "heading_repeat": -3})
        self.app.state.materials.reload_retrieval_settings()

        engine = self.engine()
        self.assertEqual(engine.bm25.b, 0.75)
        self.assertEqual(engine.bm25.heading_repeat, 1)

    def test_a_small_candidate_window_says_so_instead_of_failing(self):
        self.client.patch("/api/v1/settings", json={"candidate_k": 10})

        response = self.client.get("/api/v1/search", params={"q": "PagedAttention", "top_k": 20})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"], "invalid_search")


if __name__ == "__main__":
    unittest.main()