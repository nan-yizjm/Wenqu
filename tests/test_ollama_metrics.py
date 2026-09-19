import json
import unittest
from unittest.mock import patch
import threading
from urllib.error import URLError

from src.llm import (JSON_MODE_NUM_PREDICT, JSON_MODE_TEMPERATURE, DeepSeekClient,
                     OllamaBusy, OllamaClient, ollama_metrics)


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


class DyingStream:
    """响应头正常返回、迭代到一半超时：模拟生成中途连接停滞。"""

    def __enter__(self): return self
    def __exit__(self, *_args): return None
    def __iter__(self):
        yield '{"message":{"content":"你"},"done":false}\n'.encode()
        raise TimeoutError("timed out")


class OllamaBusyTests(unittest.TestCase):
    """双实例抢同一个 Ollama 时，第二个请求在队列里等满 120 秒读超时。

    41 号实证过这条路径：错误码 ollama_unavailable，界面让人"确认服务已启动"——
    而服务好端端的，只是正被另一个实例的长生成占着。报错必须把这件事说出来。
    """

    @patch("src.llm.urlopen")
    def test_a_wait_timeout_reports_busy_not_disconnected(self, open_url):
        """排队超时被 urllib 包装成 URLError(TimeoutError)：连上了、没数据。"""
        open_url.side_effect = URLError(TimeoutError("timed out"))

        client = OllamaClient(timeout=120)
        with self.assertRaises(OllamaBusy) as caught:
            client.chat([{"role": "user", "content": "hi"}])

        self.assertIn("120", str(caught.exception))
        self.assertIn("没有返回", str(caught.exception))

    @patch("src.llm.urlopen")
    def test_a_mid_stream_timeout_also_reports_busy(self, open_url):
        """响应头回来后行间停滞超时是裸 TimeoutError，不经 urllib 包装——此前它会
        冒泡成 generation_failed"生成暂时失败"，同样没说真因。"""
        open_url.return_value = DyingStream()

        client = OllamaClient()
        with self.assertRaises(OllamaBusy):
            list(client.stream_chat([{"role": "user", "content": "hi"}]))

    @patch("src.llm.urlopen")
    def test_a_refused_connection_is_still_reported_as_a_disconnect(self, open_url):
        """真连不上（服务没起）必须保持"无法连接"的原话，不能都赖成忙。"""
        open_url.side_effect = URLError(ConnectionRefusedError(10061))

        client = OllamaClient()
        with self.assertRaises(RuntimeError) as caught:
            client.chat([{"role": "user", "content": "hi"}])

        self.assertNotIsInstance(caught.exception, OllamaBusy)
        self.assertIn("无法连接", str(caught.exception))

    @patch("src.llm.urlopen")
    def test_a_refused_stream_connection_is_still_a_disconnect(self, open_url):
        open_url.side_effect = URLError(ConnectionRefusedError(10061))

        client = OllamaClient()
        with self.assertRaises(RuntimeError) as caught:
            list(client.stream_chat([{"role": "user", "content": "hi"}]))

        self.assertNotIsInstance(caught.exception, OllamaBusy)
        self.assertIn("无法连接", str(caught.exception))
