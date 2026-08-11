"""本地与远程大模型客户端。"""

import json
import os
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from src.config import load_env_file

load_env_file()

class OllamaClient:
    """通过 Ollama 本地 HTTP API 调用模型。"""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, messages: list[dict[str, str]]) -> str:
        """发送对话消息，并返回模型生成的纯文本回答。"""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        }

        request = Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )

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

        return content.strip()

class ChatClient(Protocol):
    """RAG 主流程依赖的最小大模型能力。"""

    def chat(self, messages: list[dict[str, str]]) -> str:
        """发送消息，并返回模型回答的纯文本。"""


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