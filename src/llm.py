"""本地与远程大模型客户端。"""

import json
import os
import threading
import time
from typing import Any, Iterator, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
def ollama_metrics(payload: dict, wall_ms: float) -> dict:
    """保留缺失值；Ollama duration 单位是 ns，不能直接当作 ms。"""
    result = {"wall_ms": round(wall_ms, 3)}
    for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
        value = payload.get(key)
        result[key.replace("_duration", "_ms")] = (
            value / 1_000_000 if type(value) in (int, float) and value >= 0 else None)
    for key in ("prompt_eval_count", "prompt_eval_cached_count", "eval_count"):
        value = payload.get(key)
        result[key] = value if type(value) is int and value >= 0 else None
    duration, count = result["eval_ms"], result["eval_count"]
    result["decode_tokens_per_second"] = (count * 1000 / duration
                                           if duration and count is not None else None)
    result["done_reason"] = payload.get("done_reason")
    return result

# json_mode 的生成 token 硬上限。format=json 的文法约束排除了"格式错了就停"这种
# 自然停止点：实测（2026-09-19）qwen2.5:7b 会陷入无限复读，64 t/s 连出 5.7 万
# token 还在靠 context shift 丢旧上下文续命，没有 num_predict 就永不停止。
# 上限按实验标定：三主题成品最长 1245 字符（34 句），连同 JSON 结构开销约 2000
# token，取 3072 留余量。顶到上限多半意味着复读，截断的 JSON 会走
# invalid_model_output 如实失败（重试即可），而不是渲染成半截假正文。
JSON_MODE_NUM_PREDICT = 3072
# json_mode 用实验验证过的温度。对照实测（同主题同证据）：产品统一口径 0.2 下
# "检索评测口径"主题无限复读（3013 token 顶满上限被硬停、如实失败），而 0.8
# 7.1 秒自然结束、11/11 全带源——实验三主题也全部正常。低温把模型压进复读
# 环，温度是这条线的参数而不是全局参数：问答仍用 0.2。
JSON_MODE_TEMPERATURE = 0.8


class OllamaClient:
    """通过 Ollama 本地 HTTP API 调用模型。"""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 120,
        generation_options: dict | None = None,
        json_mode: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.generation_options = dict(generation_options or {})
        # Ollama 的 `format: "json"` 用文法约束整个输出（不是提示词层面的"请输出
        # JSON"），模型物理上写不出非法 JSON。逐句 JSON 产出依赖这一层。
        self.json_mode = json_mode
        self.last_metrics: dict = {}

    def chat(self, messages: list[dict[str, str]]) -> str:
        """发送对话消息，并返回模型生成的纯文本回答。"""
        self.last_metrics = {}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.2,
                **self.generation_options,
            },
        }
        if self.json_mode:
            payload["format"] = "json"
            payload["options"]["temperature"] = JSON_MODE_TEMPERATURE
            payload["options"]["num_predict"] = JSON_MODE_NUM_PREDICT

        request = Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama 请求失败，HTTP 状态码：{error.code}，详情：{detail}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                "无法连接到 Ollama。请确认 Ollama 已安装并正在运行。"
            ) from error

        content = result.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Ollama 返回格式异常：{result}")

        self.last_metrics = ollama_metrics(result, (time.perf_counter() - started) * 1000)
        return content.strip()

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        cancel_event: threading.Event | None = None,
    ) -> Iterator[str]:
        """逐条读取 Ollama 的 NDJSON 响应；关闭生成器会同时关闭连接。"""
        self.last_metrics = {}
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": True,
            "options": {"temperature": 0.2, **self.generation_options},
        }
        if self.json_mode:
            payload["format"] = "json"
            payload["options"]["temperature"] = JSON_MODE_TEMPERATURE
            payload["options"]["num_predict"] = JSON_MODE_NUM_PREDICT
        request = Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"}, method="POST",
        )
        started, final_payload, received = time.perf_counter(), {}, False
        try:
            with urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    if not raw_line.strip():
                        continue
                    item = json.loads(raw_line.decode("utf-8"))
                    if item.get("error"):
                        raise RuntimeError(f"Ollama 生成失败：{item['error']}")
                    content = item.get("message", {}).get("content")
                    if isinstance(content, str) and content:
                        received = True
                        yield content
                    if item.get("done"):
                        final_payload = item
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama 请求失败，HTTP 状态码：{error.code}，详情：{detail}") from error
        except URLError as error:
            raise RuntimeError("无法连接到 Ollama。请确认 Ollama 已安装并正在运行。") from error
        if final_payload:
            self.last_metrics = ollama_metrics(
                final_payload, (time.perf_counter() - started) * 1000)
        if not received and not (cancel_event and cancel_event.is_set()):
            raise RuntimeError("Ollama 未返回有效文本。")

class ChatClient(Protocol):
    """RAG 主流程依赖的最小大模型能力。"""

    def chat(self, messages: list[dict[str, str]]) -> str:
        """发送消息，并返回模型回答的纯文本。"""

    def stream_chat(self, messages: list[dict[str, str]], cancel_event=None) -> Iterator[str]:
        """逐段返回生成文本。"""


class DeepSeekClient:
    """通过 DeepSeek Chat Completions API 调用模型。"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        if not self.api_key:
            raise RuntimeError(
                "未找到 DEEPSEEK_API_KEY。请先在环境变量中设置 DeepSeek API Key。"
            )

    def chat(self, messages: list[dict[str, str]]) -> str:
        """发送对话消息，并返回模型生成的纯文本回答。"""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }

        request = Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"DeepSeek 请求失败，HTTP 状态码：{error.code}，详情：{detail}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                "无法连接到 DeepSeek API。请检查网络连接或 API 地址。"
            ) from error

        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"DeepSeek 返回格式异常：{result}")

        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"DeepSeek 未返回有效文本：{result}")

        return content.strip()

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        cancel_event: threading.Event | None = None,
    ) -> Iterator[str]:
        """读取兼容 OpenAI Chat Completions 的 SSE 数据流。"""
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "temperature": 0.2,
            "stream": True,
        }
        request = Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8",
                     "Authorization": f"Bearer {self.api_key}"}, method="POST",
        )
        received = False
        try:
            with urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    item = json.loads(data)
                    choices = item.get("choices") or []
                    content = choices[0].get("delta", {}).get("content") if choices else None
                    if isinstance(content, str) and content:
                        received = True
                        yield content
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"DeepSeek 请求失败，HTTP 状态码：{error.code}，详情：{detail}") from error
        except URLError as error:
            raise RuntimeError("无法连接到 DeepSeek API。请检查网络连接或 API 地址。") from error
        if not received and not (cancel_event and cancel_event.is_set()):
            raise RuntimeError("DeepSeek 未返回有效文本。")

if __name__ == "__main__":
    client = OllamaClient()

    answer = client.chat(
        [
            {
                "role": "user",
                "content": "请用两句话解释 RAG 中检索器的职责。",
            }
        ]
    )

    print(answer)
