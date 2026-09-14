"""实验辅助：哈希、时间同步和整卡显存采样。不是生产监控系统。"""

import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_report(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def gpu_sample():
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=3, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        values = []
        for line in process.stdout.splitlines():
            index, name, used, total, utilization = (item.strip() for item in line.split(","))
            values.append({"index": int(index), "name": name, "used_mib": float(used),
                           "total_mib": float(total), "utilization_percent": float(utilization)})
        return values
    except (OSError, ValueError, subprocess.SubprocessError):
        return None  # 不可用不是显存为 0。


class GPUSampler:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def __enter__(self):
        self.started = time.perf_counter()
        def sample_loop():
            while not self.stop_event.is_set():
                sample = gpu_sample()
                self.samples.append({"seconds": round(time.perf_counter() - self.started, 3), "gpus": sample})
                self.stop_event.wait(self.interval)
        self.thread = threading.Thread(target=sample_loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=4)

    def report(self):
        peaks = {}
        for sample in self.samples:
            for gpu in sample["gpus"] or []:
                key = str(gpu["index"])
                peaks[key] = max(peaks.get(key, 0), gpu["used_mib"])
        return {"interval_after_sample_seconds": self.interval,
                "sampled_peak_used_mib": peaks, "samples": self.samples,
                "warning": "整卡使用量，包含 Ollama/桌面/其他进程；离散采样不是精确峰值"}


def synchronize():
    import torch
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def torch_memory():
    import torch
    if not torch.cuda.is_initialized():
        return None
    return {"allocated_mib": torch.cuda.memory_allocated() / 2**20,
            "reserved_mib": torch.cuda.memory_reserved() / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
