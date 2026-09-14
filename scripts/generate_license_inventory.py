"""从产品运行依赖递归生成版本与许可证清单。"""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import re

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def roots(requirements_file):
    values = []
    for line in requirements_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-")):
            values.append(Requirement(line).name)
    return values


def license_name(distribution):
    expression = distribution.metadata.get("License-Expression")
    if expression:
        return expression
    value = (distribution.metadata.get("License") or "").strip()
    if value and len(value) < 100 and "\n" not in value:
        return value
    classifiers = distribution.metadata.get_all("Classifier") or []
    matches = [item.rsplit("::", 1)[-1].strip() for item in classifiers if "License ::" in item]
    return ", ".join(matches) or "未在包元数据中声明"


def collect(requirements_file):
    installed = {canonicalize_name(item.metadata["Name"]): item
                 for item in metadata.distributions() if item.metadata.get("Name")}
    pending = [canonicalize_name(name) for name in roots(requirements_file)]
    included = {}
    while pending:
        name = pending.pop()
        if name in included or name not in installed:
            continue
        distribution = installed[name]; included[name] = distribution
        for raw in distribution.requires or []:
            try:
                requirement = Requirement(raw)
                if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                    continue
                pending.append(canonicalize_name(requirement.name))
            except Exception:
                continue
    return sorted(included.values(), key=lambda item: item.metadata["Name"].lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements", type=Path, default=Path("requirements-product.txt"))
    parser.add_argument("--output", type=Path, default=Path("docs/THIRD_PARTY_LICENSES.md"))
    args = parser.parse_args()
    rows = []
    for item in collect(args.requirements):
        name = item.metadata["Name"]
        homepage = item.metadata.get("Home-page") or item.metadata.get("Project-URL") or ""
        homepage = re.sub(r"^[^,]+,\s*", "", homepage).strip()
        rows.append(f"| {name} | {item.version} | {license_name(item)} | {homepage} |")
    text = "\n".join([
        "# 第三方依赖许可证清单", "",
        "此文件由 `scripts/generate_license_inventory.py` 根据发布环境的产品运行依赖递归生成。",
        "它是依赖识别清单，不替代各组件随包附带的完整许可证文本。", "",
        "| 组件 | 版本 | 许可证元数据 | 项目地址 |",
        "|---|---:|---|---|", *rows, "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {len(rows)} dependencies to {args.output}")


if __name__ == "__main__":
    main()
