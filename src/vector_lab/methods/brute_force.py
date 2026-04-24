import numpy as np

from vector_lab.methods.base import VectorSearchMethod


class BruteForceSearch(VectorSearchMethod):
    """
    Exact baseline using full dot product search.

    Assumption:
    - X and q are normalized if you want cosine similarity.
    """

    name = "brute_force"

    def __init__(self, metric: str = "dot_product"):
        if metric not in {"dot_product"}:
            raise ValueError("Only dot_product is implemented for now.")
        self.metric = metric
        self.X = None

    def fit(self, X: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        return self

    def search(self, q: np.ndarray, k: int = 10):
        if self.X is None:
            raise RuntimeError("The method must be fitted before search.")

        q = np.asarray(q, dtype=np.float32)
        scores = self.X @ q

        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        return idx, scores[idx]