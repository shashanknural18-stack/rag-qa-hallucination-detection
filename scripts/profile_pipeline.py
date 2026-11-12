"""Benchmark profiling script for RAG QA pipeline.

Runs 3 benchmark queries and prints a detailed breakdown of stage times and peak RAM.
Usage:
    python scripts/profile_pipeline.py [--mode FAST|FULL]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psutil
from src.profiling import get_profiler, stage


BENCHMARK_QUERIES = [
    "How fast must an e-commerce platform acknowledge a return request?",
    "What are the required conditions for valid consent to share personal data?",
    "What is the penalty for copyright infringement?",  # Expected refusal
]


def run_benchmark(mode: str = "FULL") -> None:
    """Run benchmark queries and print a profiling table."""
    profiler = get_profiler()
    profiler.clear()

    print(f"\n{'='*70}")
    print(f"BENCHMARK PROFILING RUN (Mode: {mode})")
    print(f"Process PID: {os.getpid()} | Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"{'='*70}\n")

    # 1. Model Loading
    print("Loading pipeline and models...")
    t_load_start = time.perf_counter()
    with stage("model_load"):
        from src.pipeline import RAGPipeline
        # If fast mode requested and supported, configure accordingly
        if mode.upper() == "FAST":
            from src.config import PRESETS
            pipe_kwargs = PRESETS.get("FAST", {})
            pipeline = RAGPipeline(
                top_k=pipe_kwargs.get("TOP_K_FINAL", 3),
            )
        else:
            pipeline = RAGPipeline()
    t_load_end = time.perf_counter()
    print(f"Pipeline ready in {t_load_end - t_load_start:.2f}s.\n")

    # 2. Run queries
    use_judge = False if mode.upper() == "FAST" else True
    num_samples = 0 if mode.upper() == "FAST" else 3
    top_k = 3 if mode.upper() == "FAST" else 5

    results = []
    for idx, q in enumerate(BENCHMARK_QUERIES, 1):
        print(f"Query {idx}/3: \"{q}\"")
        t0 = time.perf_counter()
        out = pipeline.ask(
            q,
            top_k=top_k,
            use_judge=use_judge,
            num_samples=num_samples,
        )
        total_time = time.perf_counter() - t0
        print(f"  -> Decision: {out.final.decision.value} | Confidence: {out.final.confidence:.2f} | Time: {total_time:.2f}s\n")
        results.append(out)

    # 3. Print summary table
    records = profiler.records
    print("\n" + "=" * 78)
    print(f"{'Stage':<20} | {'Duration (s)':<14} | {'Peak RAM (MB)':<14} | {'Details'}")
    print("-" * 78)

    stage_totals: dict[str, list[float]] = {}
    stage_ram: dict[str, list[float]] = {}
    stage_extras: dict[str, list[str]] = {}

    for r in records:
        stage_totals.setdefault(r.name, []).append(r.duration)
        stage_ram.setdefault(r.name, []).append(r.peak_ram_mb)
        if r.extra:
            detail_str = ", ".join(f"{k}={v}" for k, v in r.extra.items())
            stage_extras.setdefault(r.name, []).append(detail_str)

    for stage_name in stage_totals:
        avg_dur = sum(stage_totals[stage_name]) / len(stage_totals[stage_name])
        max_ram = max(stage_ram[stage_name])
        count = len(stage_totals[stage_name])
        extras = "; ".join(stage_extras.get(stage_name, []))
        detail_note = f"calls={count}" + (f" ({extras})" if extras else "")
        print(f"{stage_name:<20} | {avg_dur:<14.4f} | {max_ram:<14.1f} | {detail_note}")

    print("=" * 78)
    total_rss = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"Final Process Peak RAM: {total_rss:.1f} MB\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["FAST", "FULL"], default="FULL", help="Pipeline execution mode")
    args = parser.parse_args()
    run_benchmark(mode=args.mode)

