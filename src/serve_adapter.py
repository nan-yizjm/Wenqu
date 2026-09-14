"""启动常驻 QLoRA adapter 实验服务；不改变默认 Ollama/DeepSeek 服务。"""

import argparse
import logging
from pathlib import Path

from .adapter_runtime import AdapterRAGRuntime
from .serve import create_app
from .settings import DEFAULT_CONFIG, load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--queue-size", type=int)
    parser.add_argument("--queue-wait-timeout", type=float)
    args = parser.parse_args()
    settings = load_settings(args.config)

    def runtime_factory(loaded_settings):
        return AdapterRAGRuntime(
            loaded_settings,
            adapter=args.adapter,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
        )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import uvicorn
    # 模型和索引都常驻单进程；多 worker 会在 8 GB 显存上重复加载模型。
    uvicorn.run(
        create_app(settings, runtime_factory=runtime_factory,
                   max_queue_size=args.queue_size,
                   queue_wait_timeout=args.queue_wait_timeout),
        host=settings.host,
        port=settings.port,
        workers=1,
        access_log=False,
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
