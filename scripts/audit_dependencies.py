"""Read-only OSV audit of installed Python packages; an unavailable audit fails closed."""
import argparse
from importlib.metadata import distributions
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    packages = sorted((d.metadata["Name"], d.version) for d in distributions() if d.metadata.get("Name"))
    body = {"queries": [{"package": {"name": name, "ecosystem": "PyPI"},
                         "version": version.split("+")[0]} for name, version in packages]}
    try:
        request = Request("https://api.osv.dev/v1/querybatch", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=45) as response:
            results = json.load(response)["results"]
        if len(results) != len(packages):
            raise ValueError("incomplete OSV response")
        findings = [{"package": name, "version": version,
                     "advisories": sorted(v["id"] for v in result.get("vulns", []))}
                    for (name, version), result in zip(packages, results) if result.get("vulns")]
        report = {"source": "https://api.osv.dev", "checked_packages": len(packages), "findings": findings}
    except Exception as error:
        print(f"Dependency audit unavailable: {type(error).__name__}", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
