import time
import psutil
import numpy as np
import pandas as pd

from vector_lab.methods.brute_force import BruteForceSearch
from vector_lab.core.metrics import recall_at_k


def benchmark_method(
    method,
    X: np.ndarray,
    Q: np.ndarray,
    k: int = 10,
    exact_method=None,
):
    """
    Benchmark one vector search method against brute force.

    Returns a dictionary with:
    - build_time_sec
    - mean_recall
    - mean_latency_ms
    - p95_latency_ms
    - memory_mb
    """

    if exact_method is None:
        exact_method = BruteForceSearch().fit(X)

    process = psutil.Process()

    mem_before = process.memory_info().rss / 1024**2

    build_start = time.perf_counter()
    method.fit(X)
    build_time = time.perf_counter() - build_start

    mem_after = process.memory_info().rss / 1024**2

    recalls = []
    latencies = []

    for q in Q:
        exact_idx, _ = exact_method.search(q, k=k)

        start = time.perf_counter()
        approx_idx, _ = method.search(q, k=k)
        elapsed = time.perf_counter() - start

        recalls.append(recall_at_k(exact_idx, approx_idx, k=k))
        latencies.append(elapsed * 1000)

    return {
        "method": method.name,
        "build_time_sec": build_time,
        "mean_recall_at_k": float(np.mean(recalls)),
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "memory_delta_mb": float(mem_after - mem_before),
    }


def benchmark_many(methods, X: np.ndarray, Q: np.ndarray, k: int = 10) -> pd.DataFrame:
    exact_method = BruteForceSearch().fit(X)

    rows = []
    for method in methods:
        row = benchmark_method(
            method=method,
            X=X,
            Q=Q,
            k=k,
            exact_method=exact_method,
        )
        rows.append(row)

    return pd.DataFrame(rows)