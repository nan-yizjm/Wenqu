import unittest

from src.hf_qlora_client import HFQLoRAClient


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": [10, 11, 12, 13], "attention_mask": [1, 1, 1, 1]}


class HFQLoRAClientTests(unittest.TestCase):
    def test_count_chat_tokens_counts_input_ids_not_batch_encoding_fields(self):
        client = HFQLoRAClient.__new__(HFQLoRAClient)
        client._tokenizer = FakeTokenizer()
        count = client.count_chat_tokens([{"role": "user", "content": "RAG"}])
        self.assertEqual(count, 4)


if __name__ == "__main__":
    unittest.main()
