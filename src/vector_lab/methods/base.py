from abc import ABC, abstractmethod
import numpy as np


class VectorSearchMethod(ABC):
    """
    Common interface for all search methods.

    Every method must implement:
    - fit(X)
    - search(q, k)
    - batch_search(Q, k)
    """

    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray):
        pass

    @abstractmethod
    def search(self, q: np.ndarray, k: int = 10):
        pass

    def batch_search(self, Q: np.ndarray, k: int = 10):
        indices = []
        scores = []

        for q in Q:
            idx, sc = self.search(q, k=k)
            indices.append(idx)
            scores.append(sc)

        return np.array(indices), np.array(scores)