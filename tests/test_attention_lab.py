import unittest

import torch

from src.attention_lab import CausalSelfAttention, causal_mask, decode_fixed_inputs


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = CausalSelfAttention(16, 4).double().eval()
        self.inputs = torch.randn(2, 12, 16, dtype=torch.float64)

    def test_matches_torch_sdpa(self):
        result, _ = self.model(self.inputs)
        query, key, value = self.model.project(self.inputs)
        expected = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True, dropout_p=0)
        expected = self.model.output(expected.transpose(1, 2).contiguous().view_as(self.inputs))
        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-10)

    def test_future_inputs_do_not_change_past_outputs(self):
        changed = self.inputs.clone()
        changed[:, 6:] += 100
        before, _ = self.model(self.inputs)
        after, _ = self.model(changed)
        torch.testing.assert_close(before[:, :6], after[:, :6], rtol=1e-10, atol=1e-10)

    def test_single_and_multi_token_decode_match_full(self):
        with torch.inference_mode():
            full, _ = self.model(self.inputs)
            incremental, _ = decode_fixed_inputs(self.model, self.inputs, 5, cached=True)
            torch.testing.assert_close(full, incremental, rtol=1e-10, atol=1e-10)
            first, cache = self.model(self.inputs[:, :5], use_cache=True)
            middle, cache = self.model(self.inputs[:, 5:8], cache, use_cache=True)
            end, cache = self.model(self.inputs[:, 8:], cache, use_cache=True)
            torch.testing.assert_close(full, torch.cat((first, middle, end), dim=1), rtol=1e-10, atol=1e-10)
            self.assertEqual(cache.logical_bytes, 2 * 2 * 4 * 12 * 4 * 8)

    def test_rectangular_mask_uses_absolute_query_position(self):
        mask = causal_mask(2, 5, 3)
        self.assertEqual(mask.tolist(), [[True, True, True, True, False], [True] * 5])
        self.assertEqual(int(causal_mask(1, 5, 4).sum()), 5)
        with self.assertRaises(ValueError):
            causal_mask(2, 5, 0)

    def test_cache_is_inference_only_and_instance_specific(self):
        with self.assertRaises(ValueError):
            self.model(self.inputs, use_cache=True)
        with torch.inference_mode():
            _, cache = self.model(self.inputs[:, :3], use_cache=True)
            other = CausalSelfAttention(16, 4).double()
            with self.assertRaises(ValueError):
                other(self.inputs[:, 3:4], cache, use_cache=True)
            with self.assertRaises(ValueError):
                self.model(self.inputs[:1, 3:4], cache, use_cache=True)
            _, next_cache = self.model(self.inputs[:, 3:4], cache, use_cache=True)
            self.assertEqual(cache.length, 3)  # 旧 cache 对象没有被原地增长。
            self.assertEqual(next_cache.length, 4)

    def test_single_token_cache_does_not_retain_fused_qkv_storage(self):
        with torch.inference_mode():
            _, cache = self.model(self.inputs[:1, :1], use_cache=True)
        for tensor in (cache.key, cache.value):
            self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size())

    def test_work_counter_and_uncached_autograd(self):
        with torch.inference_mode():
            self.model.reset_work()
            decode_fixed_inputs(self.model, self.inputs, 8, cached=True)
            self.assertEqual(self.model.projected_positions, 2 * 12)
            self.model.reset_work()
            decode_fixed_inputs(self.model, self.inputs, 8, cached=False)
            self.assertEqual(self.model.projected_positions, 2 * (8 + 9 + 10 + 11 + 12))
        self.model(self.inputs)[0].sum().backward()
        self.assertTrue(torch.isfinite(self.model.qkv.weight.grad).all())
