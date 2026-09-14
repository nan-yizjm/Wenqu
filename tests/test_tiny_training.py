from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from src.train_tiny_lm import TrainConfig, learning_rate, next_token_loss, train
from src.training_data import BOS, EOS, IGNORE, PAD, UNK, load_data, make_windows, prepare_data


class TinyTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "dataset"
        docs = [{"id": f"doc{i}", "group_id": f"group{i // 2}",
                 "text": f"小林记录第{i}天的工作。先检查材料，再保存结果。"} for i in range(12)]
        prepare_data(self.data, docs)

    def test_shift_padding_and_no_group_leakage(self):
        x, y, _ = make_windows([{"id": "a", "ids": [BOS, 4, 5, EOS]}], 2)
        self.assertEqual(x.tolist(), [[BOS, 4], [5, PAD]])
        self.assertEqual(y.tolist(), [[4, 5], [EOS, IGNORE]])
        data, _, _, _ = load_data(self.data)
        self.assertFalse({doc["group_id"] for doc in data["train"]} & {doc["group_id"] for doc in data["validation"]})

    def test_vocab_is_train_only_and_duplicate_text_rejected(self):
        docs = [{"id": "a", "group_id": "a", "text": "甲甲"}, {"id": "b", "group_id": "b", "text": "乙乙"}]
        target = self.root / "unknown"
        prepare_data(target, docs)
        data, _, manifest, _ = load_data(target)
        self.assertEqual(data["validation"][0]["ids"], [BOS, UNK, UNK, EOS])
        self.assertEqual(manifest["diagnostics"]["validation_unk_tokens"], 2)
        docs[1]["text"] = "甲甲"
        with self.assertRaises(ValueError):
            prepare_data(self.root / "duplicates", docs)

    def test_loss_ignores_padding_and_lr_plan_is_not_stop_after(self):
        logits = torch.randn(1, 3, 6)
        targets = torch.tensor([[4, IGNORE, IGNORE]])
        torch.testing.assert_close(next_token_loss(logits, targets), next_token_loss(logits[:, :1], targets[:, :1]))
        config = TrainConfig(steps=10, warmup_steps=2)
        self.assertAlmostEqual(learning_rate(2, config), config.learning_rate)
        self.assertAlmostEqual(learning_rate(10, config), config.learning_rate * 0.1)

    def test_cpu_resume_matches_uninterrupted_weights_and_optimizer(self):
        config = TrainConfig(device="cpu", steps=6, warmup_steps=1, batch_size=2, block_size=12,
                             eval_every=3, eval_batch_size=4)
        model = {"d_model": 32, "heads": 4, "layers": 2, "max_seq_len": 16, "dropout": 0.2}
        full = self.root / "full"
        part = self.root / "part"
        resumed = self.root / "resumed"
        train(self.data, config, model, output=full)
        train(self.data, config, model, output=part, stop_after=3)
        train(self.data, config, model, output=resumed, resume=part / "last.pt")
        first = torch.load(full / "last.pt", weights_only=True)
        second = torch.load(resumed / "last.pt", weights_only=True)
        for key in first["model_state"]:
            self.assertTrue(torch.equal(first["model_state"][key], second["model_state"][key]), key)
        for parameter, state in first["optimizer_state"]["state"].items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, second["optimizer_state"]["state"][parameter][key]))
        self.assertTrue(torch.equal(first["sample_rng"], second["sample_rng"]))
        self.assertTrue(torch.equal(first["cpu_rng"], second["cpu_rng"]))
        with self.assertRaises(ValueError):
            train(self.data, replace(config, learning_rate=0.01), model, output=self.root / "wrong",
                  resume=part / "last.pt")
