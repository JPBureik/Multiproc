#!/usr/bin/env python3
"""Benchmarks for multiproc package.

This script runs performance comparisons and scaling analysis for the
multiproc_cpu function.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import psutil
from multiproc import multiproc_cpu


def cpu_intensive_task(n: int) -> int:
    """A CPU-intensive task for benchmarking."""
    total = 0
    for i in range(n):
        total += i * i % 1000
    return total


def benchmark_serial(data: list[int]) -> tuple[list[int], float]:
    """Run serial execution and return results with timing."""
    start = time.perf_counter()
    results = [cpu_intensive_task(x) for x in data]
    elapsed = time.perf_counter() - start
    return results, elapsed


def benchmark_parallel(
    data: list[int], free_cores: int | None = None
) -> tuple[list[Any], float]:
    """Run parallel execution and return results with timing."""
    start = time.perf_counter()
    results = multiproc_cpu(data, cpu_intensive_task, free_cores=free_cores)
    elapsed = time.perf_counter() - start
    return results, elapsed


def run_performance_comparison(sizes: list[int], iterations: int = 1000) -> None:
    """Compare serial vs parallel performance for various input sizes."""
    print("\n" + "=" * 60)
    print("Performance Comparison: Serial vs Parallel")
    print("=" * 60)
    print(f"Task: CPU-intensive computation ({iterations} iterations per item)")
    print(f"{'Size':<12} {'Serial (s)':<14} {'Parallel (s)':<14} {'Speedup':<10}")
    print("-" * 60)

    for size in sizes:
        data = [iterations] * size

        _, serial_time = benchmark_serial(data)
        _, parallel_time = benchmark_parallel(data)

        speedup = serial_time / parallel_time if parallel_time > 0 else float("inf")
        print(
            f"{size:<12} {serial_time:<14.3f} {parallel_time:<14.3f} {speedup:<10.2f}x"
        )


def run_scaling_analysis(data_size: int = 10000, iterations: int = 500) -> None:
    """Analyze performance scaling with different numbers of cores."""
    available_cores = psutil.cpu_count(logical=True) or 1

    print("\n" + "=" * 60)
    print("Scaling Analysis: Performance vs Number of Cores")
    print("=" * 60)
    print(f"Data size: {data_size} items")
    print(f"Task: {iterations} iterations per item")
    print(f"Available cores: {available_cores}")
    print(f"{'Cores':<10} {'Time (s)':<14} {'Speedup':<12} {'Efficiency':<12}")
    print("-" * 60)

    data = [iterations] * data_size

    # Get baseline with 1 core
    _, baseline_time = benchmark_parallel(data, free_cores=available_cores - 1)
    print(f"{1:<10} {baseline_time:<14.3f} {'1.00x':<12} {'100%':<12}")

    # Test with increasing cores
    core_counts = [2, 4, 8, 16, 32]
    for cores in core_counts:
        if cores > available_cores:
            break

        free = available_cores - cores
        _, elapsed = benchmark_parallel(data, free_cores=free if free > 0 else None)

        speedup = baseline_time / elapsed if elapsed > 0 else float("inf")
        efficiency = (speedup / cores) * 100

        print(f"{cores:<10} {elapsed:<14.3f} {speedup:<12.2f}x {efficiency:<12.1f}%")


def main() -> None:
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description="Run multiproc benchmarks")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks with smaller data sizes",
    )
    parser.add_argument(
        "--performance-only",
        action="store_true",
        help="Run only performance comparison",
    )
    parser.add_argument(
        "--scaling-only",
        action="store_true",
        help="Run only scaling analysis",
    )
    args = parser.parse_args()

    if args.quick:
        sizes = [100, 500, 1000]
        iterations = 100
        data_size = 1000
    else:
        sizes = [100, 500, 1000, 5000, 10000]
        iterations = 500
        data_size = 5000

    print("Multiproc Benchmarks")
    print(
        f"CPU: {psutil.cpu_count(logical=False)} physical cores, "
        f"{psutil.cpu_count(logical=True)} threads"
    )

    if not args.scaling_only:
        run_performance_comparison(sizes, iterations)

    if not args.performance_only:
        run_scaling_analysis(data_size, iterations)

    print("\n" + "=" * 60)
    print("Benchmarks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
