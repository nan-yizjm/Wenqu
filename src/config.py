"""项目本地配置加载工具。"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(env_file: Path = DEFAULT_ENV_FILE) -> None:
    """读取简单的 KEY=value 格式 .env 文件。

    已经由系统环境变量设置的同名配置优先，不会被 .env 覆盖。
    """
    if not env_file.exists():
        return

    for line_number, raw_line in enumerate(
        env_file.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            raise ValueError(
                f"{env_file} 第 {line_number} 行格式错误，应为 KEY=value"
            )

        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()

        if not key:
            raise ValueError(
                f"{env_file} 第 {line_number} 行配置名不能为空"
            )

        if (
            len(value) >= 2
            and value[0] in {"'", '"'}
            and value[-1] == value[0]
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)