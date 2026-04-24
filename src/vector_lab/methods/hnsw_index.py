# src/vector_lab/methods/hnsw_index.py

import numpy as np
import hnswlib

from vector_lab.methods.base import VectorSearchMethod


class HNSWSearch(VectorSearchMethod):
    """
    Baseline HNSW avec hnswlib.

    HNSW = Hierarchical Navigable Small World graph.

    C'est l'une des méthodes ANN les plus utilisées dans les bases
    vectorielles modernes comme Qdrant, Weaviate, OpenSearch, etc.

    Hypothèse :
        Les vecteurs X et q sont déjà normalisés si on veut utiliser
        la similarité cosinus.
    """

    def __init__(
        self,
        space: str = "cosine",
        dim: int | None = None,
        max_elements: int | None = None,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        seed: int = 0,
    ):
        if space not in {"cosine", "l2", "ip"}:
            raise ValueError("space must be one of: 'cosine', 'l2', 'ip'.")

        self.space = space
        self.dim = dim
        self.max_elements = max_elements
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.seed = seed

        self.index = None
        self.X = None

    @property
    def name(self) -> str:
        return (
            f"hnsw_"
            f"space{self.space}_"
            f"M{self.M}_"
            f"efc{self.ef_construction}_"
            f"efs{self.ef_search}"
        )

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

        n_samples, dim = X.shape

        self.X = X
        self.dim = self.dim or dim
        self.max_elements = self.max_elements or n_samples

        if self.dim != dim:
            raise ValueError(f"dim={self.dim} does not match X.shape[1]={dim}.")

        self.index = hnswlib.Index(
            space=self.space,
            dim=self.dim,
        )

        self.index.init_index(
            max_elements=self.max_elements,
            ef_construction=self.ef_construction,
            M=self.M,
            random_seed=self.seed,
        )

        ids = np.arange(n_samples, dtype=np.int64)

        self.index.add_items(X, ids)
        self.index.set_ef(self.ef_search)

        return self

    def search(self, q: np.ndarray, k: int = 10):
        if self.index is None:
            raise RuntimeError("The method must be fitted before search.")

        q = np.asarray(q, dtype=np.float32)

        if q.ndim != 1:
            raise ValueError("q must be a 1D vector.")

        k = min(k, len(self.X))

        labels, distances = self.index.knn_query(
            q.reshape(1, -1),
            k=k,
        )

        labels = labels[0]
        distances = distances[0]

        # hnswlib retourne une distance.
        # Pour cosine : distance = 1 - cosine_similarity.
        # Pour ip : selon hnswlib, distance liée à inner product.
        # Pour notre benchmark, on retourne un score comparable :
        # plus grand = meilleur.
        if self.space == "cosine":
            scores = 1.0 - distances
        elif self.space == "l2":
            scores = -distances
        else:
            scores = -distances

        return labels.astype(np.int64), scores.astype(np.float32)
