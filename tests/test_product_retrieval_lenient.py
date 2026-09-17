import unittest

from src.evaluate_product_retrieval import covers_all, covers_expected, language_groups

# 取自评测集的真实形状：retreval-005 与 retrieval-020 这类混排题，
# 关键词是为权威笔记（中文）写的，英文资料用的是另一套词。
MIXED = ["泄露", "错误操作", "恶意", "Prompt Injection"]
LLMOPS = ["监控", "审计", "版本", "质量回归", "LLMOps"]
GRPO = ["GRPO", "RLVR", "可验证奖励"]

ENGLISH_DOC = ("prompt injection happens when untrusted input is treated as "
               "instructions, letting an attacker leak data or take malicious actions.")
CHINESE_DOC = "落地这套流程要建立监控、审计、版本管理与质量回归，缺一不可。"


class LanguageGroupTests(unittest.TestCase):
    def test_splits_by_script_not_by_word(self):
        groups = language_groups(["GRPO", "可验证奖励", "RLVR"])
        self.assertEqual(groups["latin"], ["GRPO", "RLVR"])
        self.assertEqual(groups["cjk"], ["可验证奖励"])

    def test_a_keyword_with_both_scripts_counts_as_chinese(self):
        # `预训练 Infra` 这类混排关键词归中文组：英文文档覆盖不了它，
        # 但中文文档能，归哪一组决定的是哪份文档能受益。
        self.assertEqual(language_groups(["预训练 Infra"])["cjk"], ["预训练 Infra"])


class CoversExpectedTests(unittest.TestCase):
    """宽松口径已是权威口径，它的判据必须对得起"任何能答的文档"这句话。"""

    def test_an_english_document_can_now_qualify_on_its_own_language(self):
        # retrieval-001：拉丁组有两个词。旧口径还要求中文的"可验证奖励"也出现，
        # 一份讲 GRPO/RLVR 的英文资料永远不合格。
        doc = ("GRPO is a policy optimization method that relies on RLVR, "
               "using verifiable rewards instead of a learned reward model.").lower()
        self.assertFalse(covers_all(doc, GRPO))
        self.assertTrue(covers_expected(doc, GRPO))

    def test_the_residual_limitation_is_documented_not_hidden(self):
        """(3,1) 形状的题：拉丁组只有 1 个词，英文文档仍然不合格。

        这是规则的已知残留局限，不是 bug——要解决得给评测集补双语概念对照，
        匹配器不该靠猜。这条测试把它钉住，防止将来有人误以为已经修完。
        """
        doc = ("prompt injection happens when untrusted input is treated as "
               "instructions, letting an attacker leak data or take malicious actions.").lower()
        self.assertFalse(covers_expected(doc, MIXED))

    def test_a_single_keyword_group_cannot_qualify_on_its_own(self):
        # 只出现一次 "LLMOps" 更可能是蹭词而不是回答了监控与审计。
        self.assertFalse(covers_expected("我们用 llmops 改造了流程".lower(), LLMOPS))

    def test_covering_the_full_chinese_group_qualifies_without_the_lone_latin_word(self):
        # 旧口径还额外要求 "LLMOps" 也出现，一篇通篇没提这个词的中文笔记反而不合格。
        self.assertFalse(covers_all(CHINESE_DOC.lower(), LLMOPS))
        self.assertTrue(covers_expected(CHINESE_DOC.lower(), LLMOPS))

    def test_a_question_in_one_language_behaves_exactly_as_before(self):
        doc = "grpo 用 rlvr 的可验证奖励做强化学习".lower()
        self.assertTrue(covers_expected(doc, GRPO))
        self.assertTrue(covers_expected("跟这道题无关".lower(), ["GRPO", "RLVR", "可验证奖励"]) is False)

    def test_no_text_qualifies_nothing(self):
        for keywords in (MIXED, LLMOPS, GRPO):
            with self.subTest(keywords=keywords):
                self.assertFalse(covers_expected(None, keywords))
                self.assertFalse(covers_expected("", keywords))

    def test_the_rule_only_loosens_never_tightens(self):
        """任一组的关键词都是全集合的子集，所以旧口径算对的文档不能翻车。"""
        samples = [
            (ENGLISH_DOC.lower(), MIXED),
            (CHINESE_DOC.lower(), LLMOPS),
            ("grpo 与 rlvr 的关系，以及可验证奖励".lower(), GRPO),
        ]
        for text, keywords in samples:
            with self.subTest(keywords=keywords):
                if covers_all(text, keywords):
                    self.assertTrue(covers_expected(text, keywords))


class CoversAllPrimitiveTests(unittest.TestCase):
    def test_still_requires_every_keyword(self):
        self.assertTrue(covers_all("有泄露和恶意行为", ["泄露", "恶意"]))
        self.assertFalse(covers_all("只有泄露", ["泄露", "恶意"]))

    def test_blank_text_never_matches(self):
        self.assertFalse(covers_all(None, ["泄露"]))
        self.assertFalse(covers_all("", ["泄露"]))


if __name__ == "__main__":
    unittest.main()
