"""固定版本的 CPU/GPU 检索与本地生成实验；每个设备组合在独立子进程运行。"""

import argparse
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

from .config import PROJECT_ROOT
from .experiment_utils import GPUSampler, gpu_sample, save_report, sha256, synchronize, torch_memory
from .settings import DEFAULT_CONFIG, load_settings

QUESTIONS = (
    "一个基础 RAG 系统从用户问题到最终回答通常经历哪些步骤？",
    "PagedAttention 是什么？它解决什么问题？",
    "如何缓解长上下文推理时的显存占用？",
)
PROFILES = {
    "e5_cpu": {"device": "cpu", "rerank": False},
    "e5_cuda": {"device": "cuda", "rerank": False},
    "e5_cuda_rerank_cpu": {"device": "cuda", "rerank": True, "rerank_device": "cpu"},
    "e5_cuda_rerank_cuda": {"device": "cuda", "rerank": True, "rerank_device": "cuda"},
}


class TimedComponent:
    def __init__(self, component, method):
        self.component, self.method, self.last_ms = component, method, None

    def __getattr__(self, name):
        value = getattr(self.component, name)
        if name != self.method:
            return value
        def measured(*args, **kwargs):
            synchronize()
            started = time.perf_counter()
            result = value(*args, **kwargs)
            synchronize()
            self.last_ms = (time.perf_counter() - started) * 1000
            return result
        return measured


def local_models(base_url):
    with urlopen(base_url + "/api/ps", timeout=3) as response:
        return json.load(response).get("models", [])


def backend_experiment():
    from .llm import OllamaClient
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    if urlsplit(base_url).hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("部署实验只允许本机 Ollama，不向远程端点发送请求")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    options = {"temperature": 0, "seed": 42, "num_predict": 96, "num_ctx": 4096}
    client = OllamaClient(model=model, base_url=base_url, generation_options=options)
    messages = [{"role": "user", "content": "用两句话解释 RAG 中检索器和生成模型如何分工。"}]
    records = []
    for attempt in (1, 2):
        resident = local_models(base_url)
        with GPUSampler() as sampler:
            answer = client.chat(messages)
        records.append({"attempt": attempt, "models_before": resident, "answer": answer,
                        "metrics": client.last_metrics, "gpu": sampler.report()})
        print(f"Ollama 请求 {attempt}: {client.last_metrics}", flush=True)
    return {"model": model, "options": options, "messages": messages, "records": records,
            "note": "不主动卸载任何模型；以驻留列表和 load_ms 判断是否观察到冷加载。重复 prompt 会影响缓存。"}


def profile_worker(args):
    started = time.perf_counter()
    import torch
    from .runtime import RAGRuntime
    torch.set_num_threads(args.cpu_threads)
    settings = replace(load_settings(args.config), **PROFILES[args.worker], retriever="hybrid")
    with GPUSampler() as sampler:
        load_started = time.perf_counter()
        runtime = RAGRuntime(settings)
        synchronize()
        load_ms = (time.perf_counter() - load_started) * 1000
        if runtime.version != args.index_version:
            raise RuntimeError("索引版本在对比过程中切换，本组合不纳入比较")
        engine = runtime.answerer.engine
        engine.vector = TimedComponent(engine.vector, "search")
        if engine.reranker is not None:
            engine.reranker = TimedComponent(engine.reranker, "score")
        def query_once(question):
            synchronize()
            began = time.perf_counter()
            hits = engine.search(question, "hybrid", settings.top_k)
            synchronize()
            return {"question": question, "retrieval_ms": (time.perf_counter() - began) * 1000,
                    "vector_search_ms": engine.vector.last_ms,
                    "rerank_ms": engine.reranker.last_ms if engine.reranker else None,
                    "chunk_ids": [hit.chunk["id"] for hit in hits],
                    "sources": [hit.chunk["source_file"] for hit in hits]}
        first = query_once(QUESTIONS[0])
        for question in QUESTIONS:  # 所有问题先跑一次，热测量不混入某一道首次执行。
            query_once(question)
        measurements = [query_once(question) for _ in range(args.repeats) for question in QUESTIONS]
        allocated = torch_memory()
    record = {
        "profile": args.worker, "status": "ok", "index_version": runtime.version,
        "settings": PROFILES[args.worker], "cpu_threads": torch.get_num_threads(),
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "cache_status": engine.vector.cache_status, "runtime_load_ms": load_ms,
        "worker_total_ms": (time.perf_counter() - started) * 1000,
        "first_search": first, "measurements": measurements,
        "hot_median_ms": statistics.median(item["retrieval_ms"] for item in measurements),
        "hot_max_ms": max(item["retrieval_ms"] for item in measurements),
        "torch_memory": allocated, "gpu": sampler.report(),
    }
    save_report(args.run_dir / (args.worker + ".json"), record)
    print(f"{args.worker}: 加载 {load_ms:.0f} ms；热检索中位数 {record['hot_median_ms']:.2f} ms", flush=True)


def run(args):
    from .index_store import IndexStore
    from .llm import OllamaClient  # 加载既有 .env，不输出密钥。
    settings = load_settings(args.config)
    if settings.provider != "ollama":
        raise ValueError("自动部署实验只运行本地 Ollama")
    selected = IndexStore(settings.store_dir).current()
    if not selected:
        raise RuntimeError("请先构建并发布索引")
    version_path, manifest = selected
    paths = [PROJECT_ROOT / "src" / name for name in (
        "deployment_lab.py", "experiment_utils.py", "runtime.py", "answer.py", "llm.py",
        "hybrid_retrieve.py", "vector_retrieve.py", "reranker.py", "settings.py")]
    paths.extend([args.config.resolve(), version_path / "manifest.json"])
    hashes = {str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path): sha256(path)
              for path in paths}
    run_dir = PROJECT_ROOT / "data/generated" / ("deployment_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    backend = backend_experiment()
    save_report(run_dir / "ollama.json", backend)
    profiles = []
    for name in PROFILES:
        print(f"开始独立进程：{name}", flush=True)
        command = [sys.executable, "-X", "utf8", "-m", "src.deployment_lab", "--worker", name,
                   "--run-dir", str(run_dir), "--config", str(args.config.resolve()),
                   "--index-version", manifest["version"], "--repeats", str(args.repeats),
                   "--cpu-threads", str(args.cpu_threads)]
        before = gpu_sample()
        began = time.perf_counter()
        try:
            process = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
                                     timeout=240, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            path = run_dir / (name + ".json")
            if process.returncode != 0 or not path.exists():
                raise RuntimeError(f"子进程失败，退出码 {process.returncode}")
            profile = json.loads(path.read_text(encoding="utf-8"))
            print(f"完成：{name}，热检索中位数 {profile['hot_median_ms']:.2f} ms", flush=True)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            profile = {"profile": name, "status": "failed", "error_type": type(error).__name__}
            print(f"组合未完成：{name}（{type(error).__name__}），不会伪填 0 ms", flush=True)
        profile["subprocess_wall_ms"] = (time.perf_counter() - began) * 1000
        profile["gpu_before"] = before
        profiles.append(profile)
    inputs_changed = any(sha256(path) != hashes[str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)]
                         for path in paths)
    report = {"index_version": manifest["version"], "created_at": datetime.now().astimezone().isoformat(),
              "source_hashes": hashes, "inputs_changed_during_run": inputs_changed,
              "questions": QUESTIONS, "repeats": args.repeats, "cpu_threads": args.cpu_threads,
              "backend": backend, "profiles": profiles,
              "limitations": ["只有检索设备组合对比，不是答案正确率评测", "固定顺序、少量重复，未控制温度和系统负载",
                              "各组合独立模型进程；磁盘页缓存不清空，加载时间不是严格磁盘冷启动",
                              "Ollama 自然驻留有到期时间；整卡显存包含外部服务，离散峰值可能漏采样"]}
    save_report(run_dir / "summary.json", report)
    print(f"报告：{run_dir / 'summary.json'}", flush=True)
    if inputs_changed or any(profile["status"] != "ok" for profile in profiles):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--worker", choices=tuple(PROFILES), help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--index-version", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or not 1 <= args.cpu_threads <= 32:
        parser.error("要求 repeats 在 1～10、cpu-threads 在 1～32")
    if args.worker:
        if args.run_dir is None or args.index_version is None:
            parser.error("worker 必须由实验主进程启动")
        profile_worker(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
