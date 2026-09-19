import json
import unittest
from unittest.mock import patch
import threading

from src.llm import (JSON_MODE_NUM_PREDICT, JSON_MODE_TEMPERATURE, DeepSeekClient,
                     OllamaClient, ollama_metrics)


class LinesResponse:
    def __init__(self, lines): self.lines = [line.encode() for line in lines]
    def __enter__(self): return self
    def __exit__(self, *_args): return None
    def __iter__(self): return iter(self.lines)


class OllamaMetricsTests(unittest.TestCase):
    def test_units_and_missing_values(self):
        result = ollama_metrics({"load_duration": 2_000_000_000, "eval_count": 40,
                                 "eval_duration": 500_000_000}, 2600)
        self.assertEqual(result["load_ms"], 2000)
        self.assertEqual(result["decode_tokens_per_second"], 80)
        self.assertIsNone(result["prompt_eval_cached_count"])
        self.assertIsNone(result["prompt_eval_ms"])
        self.assertIsNone(ollama_metrics({"eval_count": 1, "eval_duration": 0}, 1)["decode_tokens_per_second"])

    @patch("src.llm.urlopen")
    def test_ollama_ndjson_stream(self, open_url):
        open_url.return_value = LinesResponse([
            '{"message":{"content":"你"},"done":false}\n',
            '{"message":{"content":"好"},"done":true,"eval_count":2,"eval_duration":1000000000}\n',
        ])
        client = OllamaClient()
        self.assertEqual("".join(client.stream_chat([{"role": "user", "content": "hi"}])), "你好")
        self.assertEqual(client.last_metrics["decode_tokens_per_second"], 2)

    @patch("src.llm.urlopen")
    def test_json_mode_adds_format_generation_cap_and_experiment_temperature(self, open_url):
        """json_mode = 文法约束 + 生成上限 + 实验温度。format=json 排除了"格式错了
        就停"这种自然停止点，num_predict 是失控复读的唯一硬停（实测连出 5.7 万
        token）；温度 0.8 是对照实验验证过的——0.2 下同主题无限复读。"""
        open_url.return_value = LinesResponse(['{"message":{"content":"{}"},"done":true}\n'])

        client = OllamaClient(json_mode=True)
        list(client.stream_chat([{"role": "user", "content": "hi"}]))
        payload = json.loads(open_url.call_args[0][0].data.decode("utf-8"))

        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["options"]["temperature"], JSON_MODE_TEMPERATURE)
        self.assertEqual(payload["options"]["num_predict"], JSON_MODE_NUM_PREDICT)

        client = OllamaClient()
        list(client.stream_chat([{"role": "user", "content": "hi"}]))
        payload = json.loads(open_url.call_args[0][0].data.decode("utf-8"))
        self.assertNotIn("format", payload)
        self.assertNotIn("num_predict", payload["options"])

    @patch("src.llm.urlopen")
    def test_deepseek_sse_stream_and_cancel(self, open_url):
        open_url.return_value = LinesResponse([
            'data: {"choices":[{"delta":{"content":"答"}}]}\n',
            'data: {"choices":[{"delta":{"content":"案"}}]}\n',
            'data: [DONE]\n',
        ])
        client = DeepSeekClient(api_key="fixture-key")
        self.assertEqual("".join(client.stream_chat([])), "答案")
        stopped = threading.Event(); stopped.set()
        self.assertEqual(list(client.stream_chat([], stopped)), [])
