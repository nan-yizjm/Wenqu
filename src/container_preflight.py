"""静态审计容器构建边界，并报告 Docker Desktop 是否可运行。"""

from datetime import datetime
import json
from pathlib import Path
import re
import subprocess

from .config import PROJECT_ROOT
from .experiment_utils import sha256


def run(command):
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def main():
    dockerfile = PROJECT_ROOT / "Dockerfile"
    dockerignore = PROJECT_ROOT / ".dockerignore"
    compose_file = PROJECT_ROOT / "compose.yaml"
    container_config = PROJECT_ROOT / "container/rag.container.toml"
    example_env = PROJECT_ROOT / "container/.env.container.example"

    dockerfile_text = dockerfile.read_text(encoding="utf-8")
    ignore_text = dockerignore.read_text(encoding="utf-8")
    config_text = container_config.read_text(encoding="utf-8")
    copy_lines = [line.strip() for line in dockerfile_text.splitlines()
                  if line.lstrip().upper().startswith("COPY ")]

    compose_code, compose_stdout, compose_stderr = run([
        "docker", "compose", "--env-file", str(example_env),
        "--profile", "tools", "config", "--format", "json",
    ])
    compose = json.loads(compose_stdout) if compose_code == 0 else {}
    services = compose.get("services", {})
    service = services.get("rag-service", {})
    indexer = services.get("rag-index", {})
    ports = service.get("ports", [])
    service_volumes = service.get("volumes", [])
    index_volumes = indexer.get("volumes", [])

    def volume(volumes, target):
        return next((item for item in volumes if item.get("target") == target), None)

    service_vault = volume(service_volumes, "/data/vault")
    index_vault = volume(index_volumes, "/data/vault")
    environment_keys = set((service.get("environment") or {}).keys())
    checks = {
        "compose_parses": compose_code == 0,
        "dockerfile_uses_copy_allowlist": (
            bool(copy_lines)
            and not any(re.match(r"COPY\s+(?:--\S+\s+)*\.\s", line, re.IGNORECASE)
                        for line in copy_lines)
        ),
        "service_source_copy_excludes_training_and_experiments": (
            "COPY src ./src" not in dockerfile_text
            and "train_qlora.py" not in dockerfile_text
            and "adapter_runtime.py" not in dockerfile_text
            and all(f"src/{name}.py" in dockerfile_text for name in (
                "serve", "runtime", "answer", "index_store", "vector_retrieve",
                "container_prepare",
            ))
        ),
        "dockerignore_blocks_secrets_and_runtime_data": all(
            pattern in ignore_text.splitlines()
            for pattern in (
                ".env", ".env.*", "data", ".venv", ".git",
                "**/__pycache__/", "**/*.pyc",
            )
        ),
        "container_env_files_excluded_from_image": (
            ".env.*" in ignore_text.splitlines()
            and not any(".env.container" in line for line in copy_lines)
        ),
        "container_config_has_no_windows_private_path": (
            "C:\\Users" not in config_text and "/data/vault" in config_text
        ),
        "container_config_keeps_safe_host_policy": "host = '127.0.0.1'" in config_text,
        "socket_wildcard_is_explicit_container_override": (
            '"--bind-host", "0.0.0.0"' in dockerfile_text
        ),
        "published_port_is_loopback_only": (
            len(ports) == 1 and ports[0].get("host_ip") == "127.0.0.1"
        ),
        "vault_is_read_only_for_index_and_service": (
            bool(service_vault and service_vault.get("read_only"))
            and bool(index_vault and index_vault.get("read_only"))
        ),
        "service_environment_has_no_api_key": (
            environment_keys <= {"OLLAMA_BASE_URL", "OLLAMA_MODEL"}
            and not any("KEY" in name or "TOKEN" in name or "SECRET" in name
                        for name in environment_keys)
        ),
        "index_and_cache_are_runtime_mounts": all(
            volume(service_volumes, target) and volume(index_volumes, target)
            for target in ("/data/index_store", "/cache/huggingface")
        ),
        "model_download_is_explicit_tool_service": (
            "rag-model-cache" in services
            and "src.container_prepare" in " ".join(
                services["rag-model-cache"].get("command") or []
            )
        ),
    }

    version_code, version_stdout, version_stderr = run([
        "docker", "version", "--format", "{{json .}}",
    ])
    desktop_code, desktop_stdout, desktop_stderr = run(["docker", "desktop", "status"])
    environment = {
        "docker_cli_available": "Client" in version_stdout,
        "docker_daemon_available": version_code == 0,
        "docker_version_output": version_stdout,
        "docker_version_error": version_stderr,
        "desktop_status_code": desktop_code,
        "desktop_status": desktop_stdout or desktop_stderr,
        "runtime_validation": (
            "available" if version_code == 0
            else "blocked_until_docker_desktop_is_started"
        ),
    }
    report = {
        "schema": "container-preflight-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "artifact_checks": checks,
        "artifacts_passed": all(checks.values()),
        "environment": environment,
        "copy_lines": copy_lines,
        "compose_summary": {
            "services": sorted(services),
            "published_ports": ports,
            "service_environment_keys": sorted(environment_keys),
            "service_mount_targets": sorted(item.get("target", "")
                                            for item in service_volumes),
        },
        "source_signature": {
            "Dockerfile": sha256(dockerfile),
            ".dockerignore": sha256(dockerignore),
            "compose.yaml": sha256(compose_file),
            "container/rag.container.toml": sha256(container_config),
            "src/container_preflight.py": sha256(Path(__file__)),
        },
        "interpretation": (
            "artifact checks 通过只说明构建边界和 Compose 展开符合约束；"
            "Docker daemon 不可用时，不能宣称镜像已构建或容器已运行。"
        ),
    }
    output_dir = PROJECT_ROOT / "data/generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        "container_preflight_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
    )
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({
        "artifact_checks": checks,
        "artifacts_passed": report["artifacts_passed"],
        "environment": environment,
        "report": str(output),
    }, ensure_ascii=False, indent=2))
    if not report["artifacts_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
