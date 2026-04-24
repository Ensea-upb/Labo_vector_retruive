import numpy as np
import scann

from vector_lab.methods.base import VectorSearchMethod


class ScannSearch(VectorSearchMethod):
    """
    Google ScaNN wrapper.

    This class uses ScaNN as a benchmark competitor.
    """

    name = "scann"

    def __init__(
        self,
        num_neighbors: int = 10,
        distance_measure: str = "dot_product",
        num_leaves: int = 100,
        num_leaves_to_search: int = 10,
        training_sample_size: int | None = None,
        anisotropic_quantization_threshold: float = 0.2,
        reorder: int = 100,
    ):
        self.num_neighbors = num_neighbors
        self.distance_measure = distance_measure
        self.num_leaves = num_leaves
        self.num_leaves_to_search = num_leaves_to_search
        self.training_sample_size = training_sample_size
        self.anisotropic_quantization_threshold = anisotropic_quantization_threshold
        self.reorder = reorder

        self.X = None
        self.searcher = None

    def fit(self, X: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        n = self.X.shape[0]

        num_leaves = min(self.num_leaves, n)
        num_leaves_to_search = min(self.num_leaves_to_search, num_leaves)
        training_sample_size = self.training_sample_size or n
        training_sample_size = min(training_sample_size, n)
        reorder = min(self.reorder, n)

        self.searcher = (
            scann.scann_ops_pybind.builder(
                self.X,
                self.num_neighbors,
                self.distance_measure,
            )
            .tree(
                num_leaves=num_leaves,
                num_leaves_to_search=num_leaves_to_search,
                training_sample_size=training_sample_size,
            )
            .score_ah(
                2,
                anisotropic_quantization_threshold=self.anisotropic_quantization_threshold,
            )
            .reorder(reorder)
            .build()
        )

        return self

    def search(self, q: np.ndarray, k: int = 10):
        if self.searcher is None:
            raise RuntimeError("The method must be fitted before search.")

        q = np.asarray(q, dtype=np.float32)
        idx, scores = self.searcher.search(q, final_num_neighbors=k)

        return np.asarray(idx), np.asarray(scores)