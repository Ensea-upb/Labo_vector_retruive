import numpy as np
from sklearn.neighbors import BallTree

from vector_lab.methods.base import VectorSearchMethod
from vector_lab.core.distances import pairwise_squared_l2


class SatelliteSearch(VectorSearchMethod):
    """
    Anchor / satellite-based approximate nearest neighbor method.

    Version improved with BallTree index on satellite signatures.
    """

    name = "satellite_balltree"

    def __init__(
        self,
        n_satellites: int = 32,
        n_candidates: int = 500,
        strategy: str = "random",
        seed: int = 0,
        leaf_size: int = 40,
    ):
        if strategy not in {"random", "farthest"}:
            raise ValueError("strategy must be 'random' or 'farthest'.")

        self.n_satellites = n_satellites
        self.n_candidates = n_candidates
        self.strategy = strategy
        self.seed = seed
        self.leaf_size = leaf_size

        self.X = None
        self.S = None
        self.Z = None
        self.tree = None

    def fit(self, X: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.S = self._choose_satellites(self.X)
        self.Z = self._signature(self.X)

        self.tree = BallTree(self.Z, leaf_size=self.leaf_size, metric="euclidean")

        return self

    def search(self, q: np.ndarray, k: int = 10):
        if self.X is None or self.S is None or self.Z is None or self.tree is None:
            raise RuntimeError("The method must be fitted before search.")

        q = np.asarray(q, dtype=np.float32).reshape(1, -1)

        zq = self._signature(q)

        M = min(self.n_candidates, len(self.X))

        _, candidate_idx = self.tree.query(zq, k=M)
        candidate_idx = candidate_idx[0]

        q_vec = q[0]
        scores = self.X[candidate_idx] @ q_vec

        top = np.argpartition(-scores, kth=k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        final_idx = candidate_idx[top]
        final_scores = scores[top]

        return final_idx, final_scores

    def _signature(self, X: np.ndarray) -> np.ndarray:
        D2 = pairwise_squared_l2(X, self.S)
        return np.sqrt(D2).astype(np.float32)

    def _choose_satellites(self, X: np.ndarray) -> np.ndarray:
        if self.strategy == "random":
            return self._choose_random(X)

        if self.strategy == "farthest":
            return self._choose_farthest(X)

        raise ValueError(f"Unknown strategy: {self.strategy}")

    def _choose_random(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        m = min(self.n_satellites, len(X))
        idx = rng.choice(len(X), size=m, replace=False)
        return X[idx].copy()

    def _choose_farthest(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)

        n = len(X)
        m = min(self.n_satellites, n)

        first = int(rng.integers(n))
        selected = [first]

        min_dist = np.full(n, np.inf, dtype=np.float32)

        for _ in range(1, m):
            last = X[selected[-1]].reshape(1, -1)
            dist = pairwise_squared_l2(X, last).ravel()
            min_dist = np.minimum(min_dist, dist)
            selected.append(int(np.argmax(min_dist)))

        return X[selected].copy()