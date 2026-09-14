import unittest

import torch

from src.tiny_decoder import DecoderConfig, TinyDecoder


class TinyDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = TinyDecoder(DecoderConfig(vocab_size=20, d_model=32, heads=4, layers=2,
                                               max_seq_len=16, dropout=0.2)).eval()
        self.ids = torch.randint(4, 20, (2, 12))

    def test_logits_and_multi_layer_cache_use_correct_positions(self):
        with torch.inference_mode():
            full, _ = self.model(self.ids)
            first, cache = self.model(self.ids[:, :4], use_cache=True)
            second, cache = self.model(self.ids[:, 4:7], cache, use_cache=True)
            third, cache = self.model(self.ids[:, 7:], cache, use_cache=True)
        self.assertEqual(tuple(full.shape), (2, 12, 20))
        self.assertEqual([item.length for item in cache], [12, 12])
        torch.testing.assert_close(full, torch.cat((first, second, third), dim=1), rtol=1e-5, atol=1e-6)

    def test_no_future_leakage_and_position_limit(self):
        changed = self.ids.clone()
        changed[:, 6:] = 4
        torch.testing.assert_close(self.model(self.ids)[0][:, :6], self.model(changed)[0][:, :6])
        with torch.inference_mode():
            _, cache = self.model(self.ids, use_cache=True)
            with self.assertRaises(ValueError):
                self.model(self.ids[:, :5], cache, use_cache=True)
        with self.assertRaises(ValueError):
            self.model(self.ids, use_cache=True)

    def test_generate_restores_training_mode_and_backprop_works(self):
        self.model.train()
        generated = self.model.generate([1, 4, 5], max_new_tokens=3)
        self.assertTrue(self.model.training)
        self.assertLessEqual(generated["generated_tokens"], 3)
        self.model(self.ids)[0].sum().backward()
        for module in (self.model.token_embedding, self.model.position_embedding, self.model.lm_head):
            self.assertTrue(torch.isfinite(module.weight.grad).all())
