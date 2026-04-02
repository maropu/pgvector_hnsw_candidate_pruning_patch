#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ann_benchmark.py - Run ann-benchmarks pgvector suite and display results as a table

Runs ann-benchmarks/run.py against a pgvector-enabled PostgreSQL (typically the
Docker container from dockerfiles/run/Dockerfile), then parses the HDF5 result
files and prints a summary table of Recall, QPS, and latency percentiles.

Usage:
  # Run benchmark and display results
  python scripts/run_ann_benchmark.py \
      --host 127.0.0.1 \
      --port 5432 \
      --user postgres \
      --password postgres \
      --dbname postgres \
      --dataset sift-128-euclidean \
      --count 10 \
      --runs 5

  # Display results only (skip re-running the benchmark)
  python scripts/run_ann_benchmark.py \
      --dataset sift-128-euclidean \
      --count 10 \
      --results-only

Notes:
  - Requires scripts/.ann-benchmarks to exist (clone with:
      git clone https://github.com/erikbern/ann-benchmarks.git scripts/.ann-benchmarks)
  - HDF5 result files are written to scripts/.ann-benchmarks/results/
  - ANN_BENCHMARKS_PG_START_SERVICE is always set to 0 (assumes PostgreSQL is
    already running via Docker)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
ANN_DIR = SCRIPT_DIR / ".ann-benchmarks"

if not ANN_DIR.exists():
    print(
        f"[error] {ANN_DIR} not found.\n"
        "Clone ann-benchmarks first:\n"
        "  git clone https://github.com/erikbern/ann-benchmarks.git scripts/.ann-benchmarks",
        file=sys.stderr,
    )
    sys.exit(1)

# Add ann-benchmarks to sys.path so we can reuse its modules directly
sys.path.insert(0, str(ANN_DIR))

from ann_benchmarks.datasets import get_dataset  # noqa: E402
from ann_benchmarks.plotting.metrics import get_recall_values, knn_threshold  # noqa: E402
from ann_benchmarks.results import load_all_results  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run ann-benchmarks pgvector suite and display results."
    )
    ap.add_argument("--host", default="127.0.0.1",
                    help="PostgreSQL host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=5432,
                    help="PostgreSQL port (default: 5432)")
    ap.add_argument("--user", default="postgres",
                    help="PostgreSQL user (default: postgres)")
    ap.add_argument("--password", default="postgres",
                    help="PostgreSQL password (default: postgres)")
    ap.add_argument("--dbname", default="postgres",
                    help="PostgreSQL database name (default: postgres)")
    ap.add_argument("--dataset", default="sift-128-euclidean",
                    help="ann-benchmarks dataset name (default: sift-128-euclidean)")
    ap.add_argument("--count", type=int, default=10,
                    help="Number of nearest neighbours (default: 10)")
    ap.add_argument("--runs", type=int, default=5,
                    help="Query runs per configuration (default: 5)")
    ap.add_argument("--results-only", action="store_true",
                    help="Skip running the benchmark; only parse and display existing results")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env.update({
        "ANN_BENCHMARKS_PG_HOST": args.host,
        "ANN_BENCHMARKS_PG_PORT": str(args.port),
        "ANN_BENCHMARKS_PG_USER": args.user,
        "ANN_BENCHMARKS_PG_PASSWORD": args.password or "",
        "ANN_BENCHMARKS_PG_DBNAME": args.dbname,
        "ANN_BENCHMARKS_PG_START_SERVICE": "0",
    })

    cmd = [
        sys.executable, "run.py",
        "--dataset", args.dataset,
        "--count", str(args.count),
        "--algorithm", "pgvector",
        "--runs", str(args.runs),
        "--local",
    ]

    print(f"[benchmark] cwd : {ANN_DIR}")
    print(f"[benchmark] cmd : {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=ANN_DIR, env=env)
    if result.returncode != 0:
        print(f"\n[error] run.py exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------

def display_results(args: argparse.Namespace) -> None:
    # load_all_results uses relative path "results/", so chdir first
    prev_dir = Path.cwd()
    os.chdir(ANN_DIR)
    try:
        _display(args)
    finally:
        os.chdir(prev_dir)


def _display(args: argparse.Namespace) -> None:
    dataset, _ = get_dataset(args.dataset)
    true_distances = np.array(dataset["distances"])
    count = args.count

    # NOTE: load_all_results is a generator that opens each HDF5 file inside a
    # `with` block and yields while it is still open.  Converting to list() first
    # closes the files before we access the datasets, so we must process each
    # result inside the loop while the file handle is still live.
    rows: list[tuple] = []
    found = False
    for properties, run in load_all_results(args.dataset, count):
        found = True
        run_distances = np.array(run["distances"])
        times = np.array(run["times"])

        recall, _, _ = get_recall_values(true_distances, run_distances, count, knn_threshold)

        best_search_time = properties.get("best_search_time", None)
        qps = 1.0 / best_search_time if best_search_time else float("nan")
        build_time = properties.get("build_time", float("nan"))
        p50 = float(np.percentile(times, 50)) * 1000
        p95 = float(np.percentile(times, 95)) * 1000
        p99 = float(np.percentile(times, 99)) * 1000

        name = properties.get("name", properties.get("algo", "?"))
        rows.append((name, float(recall), qps, float(build_time), p50, p95, p99))

    if not found:
        print(f"[warn] No result files found under results/{args.dataset}/{count}/",
              file=sys.stderr)
        return

    # Sort by recall descending, then QPS descending
    rows.sort(key=lambda r: (-r[1], -r[2]))

    # ---------------------------------------------------------------------------
    # Print table
    # ---------------------------------------------------------------------------
    COL_NAME = 52
    header = (
        f"{'Algorithm':<{COL_NAME}}"
        f"{'Recall':>8}"
        f"{'QPS':>10}"
        f"{'Build(s)':>10}"
        f"{'p50(ms)':>9}"
        f"{'p95(ms)':>9}"
        f"{'p99(ms)':>9}"
    )
    sep = "─" * len(header)

    print()
    print(f"  Dataset : {args.dataset}")
    print(f"  k       : {count}")
    print(f"  Results : {len(rows)} configuration(s)")
    print()
    print(sep)
    print(header)
    print(sep)
    for name, recall, qps, build_time, p50, p95, p99 in rows:
        print(
            f"{name:<{COL_NAME}}"
            f"{recall:>8.4f}"
            f"{qps:>10.1f}"
            f"{build_time:>10.2f}"
            f"{p50:>9.3f}"
            f"{p95:>9.3f}"
            f"{p99:>9.3f}"
        )
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.results_only:
        run_benchmark(args)

    print("\n[results] Parsing HDF5 result files ...")
    display_results(args)


if __name__ == "__main__":
    main()
