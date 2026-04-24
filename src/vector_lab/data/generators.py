import numpy as np

from vector_lab.core.distances import normalize_rows


def make_random_unit_vectors(
    n: int = 10_000,
    d: int = 128,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    return normalize_rows(X).astype(np.float32)


def make_query_vectors(
    n_queries: int = 100,
    d: int = 128,
    seed: int = 123,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(n_queries, d)).astype(np.float32)
    return normalize_rows(Q).astype(np.float32)