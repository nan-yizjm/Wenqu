"""审计正在运行的本机容器，并保存一次真实 RAG 调用结果。"""

from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen

from .config import PROJECT_ROOT
from .experiment_utils import sha256


CONTAINER = "obsidian-rag-local-rag-service-1"
IMAGE = "obsidian-rag:local"


def command(*args):
    result = subprocess.run(
        args, cwd=PROJECT_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=True,
    )
    return result.stdout.strip()


def get(path):
    with urlopen("http://127.0.0.1:8000" + path, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    container = json.loads(command("docker", "inspect", CONTAINER))[0]
    image = json.loads(command("docker", "image", "inspect", IMAGE))[0]
    files = command(
        "docker", "run", "--rm", "--entrypoint", "find", IMAGE,
        "/app", "-type", "f",
    ).splitlines()
    ready = get("/health/ready")
    index = get("/v1/index")

    question = "PagedAttention 是什么？它解决什么问题？"
    raw = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
    request = Request(
        "http://127.0.0.1:8000/v1/answer", data=raw,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urlopen(request, timeout=120) as response:
        answer = json.loads(response.read().decode("utf-8"))
        status = response.status
    client_elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    metrics = get("/v1/metrics")

    port_bindings = container["HostConfig"]["PortBindings"]["8000/tcp"]
    mounts = {item["Destination"]: item for item in container["Mounts"]}
    environment_names = sorted(
        item.split("=", 1)[0] for item in container["Config"].get("Env", [])
    )
    application_environment_names = {
        name for name in environment_names
        if name.startswith(("OLLAMA_", "DEEPSEEK_", "RAG_"))
    }
    disallowed_files = [path for path in files if any(marker in path for marker in (
        "/.env", "/data/", "__pycache__", ".pyc", "train_", "adapter_", "tiny_",
    ))]
    source_labels = {source["label"] for source in answer.get("sources", [])}
    checks = {
        "container_running_and_healthy": (
            container["State"]["Running"]
            and container["State"]["Health"]["Status"] == "healthy"
        ),
        "image_runs_as_non_root": image["Config"]["User"] == "rag",
        "port_published_to_loopback_only": (
            len(port_bindings) == 1 and port_bindings[0]["HostIp"] == "127.0.0.1"
        ),
        "vault_mount_is_read_only": mounts["/data/vault"]["RW"] is False,
        "runtime_artifacts_are_mounted_not_baked": (
            "/data/index_store" in mounts and "/cache/huggingface" in mounts
        ),
        "image_omits_secret_data_cache_and_training_files": not disallowed_files,
        "application_environment_is_ollama_only": (
            application_environment_names == {"OLLAMA_BASE_URL", "OLLAMA_MODEL"}
        ),
        "ready_uses_container_index_and_host_ollama": (
            ready["ready"] and ready["backend_check"] == "ollama_model_installed"
            and index["provider"] == "ollama" and index["chunks"] == 911
        ),
        "real_answer_completed_with_valid_sources": (
            status == 200 and not answer["rejected"] and answer["generation_calls"] == 1
            and answer["citation_validation"]["is_valid"] and not answer["citation_validation"]["invalid_labels"]
            and set(answer["citation_validation"]["cited_labels"]) <= source_labels
        ),
        "retrieval_method_is_explicit": answer.get("retrieval_method") == "hybrid",
        "metrics_are_content_free": all(
            marker not in json.dumps(metrics["requests"], ensure_ascii=False)
            for marker in ("PagedAttention", "结论：", "KV Cache")
        ),
    }
    report = {
        "schema": "container-runtime-probe-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "image": {
            "id": image["Id"], "size_bytes": image["Size"],
            "user": image["Config"]["User"], "cmd": image["Config"]["Cmd"],
            "files": files,
        },
        "container": {
            "name": CONTAINER,
            "state": container["State"],
            "port_bindings": port_bindings,
            "mounts": [{"destination": item["Destination"], "rw": item["RW"],
                        "type": item["Type"]} for item in container["Mounts"]],
            "environment_names": environment_names,
            "application_environment_names": sorted(application_environment_names),
        },
        "ready": ready,
        "index": index,
        "request": {
            "question": question, "status": status,
            "client_elapsed_ms": client_elapsed_ms, "response": answer,
        },
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "source_signature": {
            "Dockerfile": sha256(PROJECT_ROOT / "Dockerfile"),
            ".dockerignore": sha256(PROJECT_ROOT / ".dockerignore"),
            "compose.yaml": sha256(PROJECT_ROOT / "compose.yaml"),
            "src/container_runtime_probe.py": sha256(Path(__file__)),
            "src/runtime.py": sha256(PROJECT_ROOT / "src/runtime.py"),
        },
    }
    output_dir = PROJECT_ROOT / "data/generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        "container_runtime_probe_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
    )
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        "image": report["image"] | {"files": len(files)},
        "ready": ready,
        "request": {
            "status": status, "answer": answer.get("answer"),
            "retrieval_method": answer.get("retrieval_method"),
            "phase_timings_ms": answer.get("phase_timings_ms"),
        },
        "checks": checks,
        "report": str(output),
    }, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
