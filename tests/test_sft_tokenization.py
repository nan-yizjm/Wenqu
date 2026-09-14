import unittest

import torch

from src.sft_tokenization import collate_response_only, response_only_tokens


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.assert_tokenize = tokenize
        if add_generation_prompt:
            return [10, 11, 12]
        return [10, 11, 12, 20, 21]


class SFTTokenizationTests(unittest.TestCase):
    def test_only_assistant_suffix_and_end_marker_are_supervised(self):
        tokenizer = FakeTokenizer()
        record = {"id": "x", "messages": [{"role": "user", "content": "q"}], "response": "a"}
        encoded = response_only_tokens(record, tokenizer, max_length=8)
        self.assertEqual(encoded["labels"], [-100, -100, -100, 20, 21])
        second = {"input_ids": [10, 20], "attention_mask": [1, 1], "labels": [-100, 20]}
        batch = collate_response_only([encoded, second], pad_token_id=0)
        self.assertTrue(torch.equal(batch["input_ids"][1], torch.tensor([10, 20, 0, 0, 0])))
        self.assertTrue(torch.equal(batch["labels"][1], torch.tensor([-100, 20, -100, -100, -100])))


if __name__ == "__main__":
    unittest.main()
