"""
profiling_gpu.py
Full inference benchmarking with GPU/CPU utilization tracking.

Outputs (inside <out_dir>/benchmark):
- inference_latency_hist.png
- gpu_usage.png
- gpu_vram.png
- cpu_usage.png
- memory_timeline.png
- profiling.csv

Intended to be used for "full inference benchmark" on the validation loader.
"""

import time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil
import GPUtil
import seaborn as sns
import torch

sns.set_theme(style="whitegrid")


# =============================================================
# Utility: Safe plotting
# =============================================================
def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


# =============================================================
# BENCHMARK CORE
# =============================================================
@torch.no_grad()
def benchmark_inference(model, loader, device, out_dir: Path) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Measures, per batch:
        - latency
        - GPU utilization (%)
        - GPU memory (VRAM %) – first GPU only
        - CPU utilization (%)
        - System RAM (%)

    Logs per batch to profiling.csv in out_dir.
    """
    latencies = []
    gpu_util = []
    gpu_mem = []
    cpu_util = []
    sys_mem = []

    model.eval()

    csv_rows = []
    t0 = time.time()

    for i, (xb, _) in enumerate(loader):
        start = time.time()

        xb = xb.to(device, non_blocking=True)
        _ = model(xb)  # forward pass only

        end = time.time()

        # record latency
        latency = end - start
        latencies.append(latency)

        # sample system stats
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        cpu_util.append(cpu)
        sys_mem.append(mem)

        # sample GPU stats (first GPU if present)
        gpus = GPUtil.getGPUs()
        if gpus:
            cuda = gpus[0]
            gpu_util.append(cuda.load * 100.0)
            gpu_mem.append(cuda.memoryUtil * 100.0)
        else:
            gpu_util.append(0.0)
            gpu_mem.append(0.0)

        csv_rows.append(
            [
                i,
                latency,
                cpu_util[-1],
                sys_mem[-1],
                gpu_util[-1],
                gpu_mem[-1],
            ]
        )

    total_time = time.time() - t0

    # ---------------------------------------------------------
    # Write profiling CSV
    # ---------------------------------------------------------
    out_csv = out_dir / "profiling.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("batch,latency,cpu,sys_ram,gpu,gpu_vram\n")
        for row in csv_rows:
            f.write(",".join(str(x) for x in row) + "\n")

    print(f"[Profiler] batches={len(latencies)} | time={total_time:.2f}s")
    return (
        np.array(latencies, dtype=float),
        np.array(cpu_util, dtype=float),
        np.array(sys_mem, dtype=float),
        np.array(gpu_util, dtype=float),
        np.array(gpu_mem, dtype=float),
    )


# =============================================================
# VISUALIZATION
# =============================================================
def plot_latency_hist(latencies: np.ndarray, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.histplot(latencies, bins=40, color="#4e79a7")
    plt.xlabel("Latency per batch (seconds)")
    plt.ylabel("Frequency")
    plt.title("Inference Latency Histogram")
    _save(out_dir / "inference_latency_hist.png")


def plot_timelines(
    cpu: np.ndarray,
    sys_ram: np.ndarray,
    gpu: np.ndarray,
    gpu_mem: np.ndarray,
    out_dir: Path,
) -> None:
    steps = np.arange(len(cpu))

    # CPU
    plt.figure(figsize=(6, 4))
    plt.plot(steps, cpu, color="#2ca02c")
    plt.title("CPU Utilization (%)")
    plt.xlabel("Batch")
    plt.ylabel("CPU %")
    _save(out_dir / "cpu_usage.png")

    # GPU
    plt.figure(figsize=(6, 4))
    plt.plot(steps, gpu, color="#1f77b4")
    plt.title("GPU Utilization (%)")
    plt.xlabel("Batch")
    plt.ylabel("GPU %")
    _save(out_dir / "gpu_usage.png")

    # GPU VRAM
    plt.figure(figsize=(6, 4))
    plt.plot(steps, gpu_mem, color="#ff7f0e")
    plt.title("GPU VRAM Utilization (%)")
    plt.xlabel("Batch")
    plt.ylabel("VRAM %")
    _save(out_dir / "gpu_vram.png")

    # System RAM
    plt.figure(figsize=(6, 4))
    plt.plot(steps, sys_ram, color="#d62728")
    plt.title("System RAM Usage (%)")
    plt.xlabel("Batch")
    plt.ylabel("RAM %")
    _save(out_dir / "memory_timeline.png")


# =============================================================
# PUBLIC ENTRY
# =============================================================
def run_gpu_benchmark(model, loader, device, out_dir: Path) -> dict:
    """
    Convenience entry:
      - runs full benchmark over `loader`
      - writes CSV + plots to <out_dir>/benchmark
      - returns a small summary dict (batches, mean latency, p95)
    """
    bench_dir = out_dir / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)

    latencies, cpu, sys_ram, gpu, gpu_mem = benchmark_inference(
        model, loader, device, bench_dir
    )

    if len(latencies) == 0:
        return {
            "batches": 0,
            "mean_latency": 0.0,
            "p95_latency": 0.0,
        }

    plot_latency_hist(latencies, bench_dir)
    plot_timelines(cpu, sys_ram, gpu, gpu_mem, bench_dir)

    summary = {
        "batches": int(len(latencies)),
        "mean_latency": float(latencies.mean()),
        "p95_latency": float(np.percentile(latencies, 95)),
    }
    print(
        f"[Profiler] mean={summary['mean_latency']:.4f}s | "
        f"p95={summary['p95_latency']:.4f}s"
    )
    return summary
