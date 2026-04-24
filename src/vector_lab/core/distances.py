import numpy as np


def normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize each row of X to unit norm.
    Useful when using cosine similarity or dot product search.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


def dot_scores(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Compute dot product scores between all rows of X and query q.
    """
    return X @ q


def squared_l2_distances(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Compute squared Euclidean distance between all rows of X and query q.
    """
    diff = X - q
    return np.sum(diff * diff, axis=1)


def pairwise_squared_l2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared Euclidean distances between rows of A and rows of B.
    Returns a matrix of shape len(A) x len(B).
    """
    A_norm = np.sum(A * A, axis=1, keepdims=True)
    B_norm = np.sum(B * B, axis=1, keepdims=True).T
    D = A_norm + B_norm - 2.0 * (A @ B.T)
    return np.maximum(D, 0.0)
