# src/vector_lab/methods/dict_bucket.py

import numpy as np

from vector_lab.methods.base import VectorSearchMethod


class DictBucketRecursiveSearch(VectorSearchMethod):
    """
    Recherche récursive par dictionnaire relationnel optimisée.

    Phase fit :
        Pour chaque pivot i, on construit :

            bucket_index[i][b] = indices j tels que
            round(bucket_scale * cos(x_i, x_j)) = b

        On calcule aussi, pour chaque pivot i, la distribution des tailles
        de ses buckets. Les pivots dont les buckets sont les plus équilibrés
        sont conservés dans self.best_vectors.

        Important :
        On ne stocke pas les écarts-types de tous les pivots.
        On garde seulement les n_best_pivots meilleurs pivots.

    Niveau 1 :
        refine_candidate_subset(q_norm, candidate_subset)

    Niveau 2 :
        search(q, k)
    """

    def __init__(
        self,
        bucket_scale: int = 10,
        n_refinement_steps: int = 5,
        n_best_pivots: int = 20,
        rerank_final: bool = True,
        seed: int = 0,
    ):
        if bucket_scale <= 0:
            raise ValueError("bucket_scale must be positive.")

        if n_refinement_steps <= 0:
            raise ValueError("n_refinement_steps must be positive.")

        if n_best_pivots <= 0:
            raise ValueError("n_best_pivots must be positive.")

        self.bucket_scale = bucket_scale
        self.n_refinement_steps = n_refinement_steps
        self.n_best_pivots = n_best_pivots
        self.rerank_final = rerank_final
        self.seed = seed

        self.X_norm = None
        self.bucket_index = None
        self.best_vectors = None

        self.rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return (
            f"dict_bucket_recursive_"
            f"scale{self.bucket_scale}_"
            f"steps{self.n_refinement_steps}_"
            f"best{self.n_best_pivots}"
        )

    def fit(self, X: np.ndarray):
        """
        Construit le dictionnaire relationnel complet.

        Version optimisée :
        - normalisation de X ;
        - calcul de la matrice des similarités cosinus ;
        - quantification par round(bucket_scale * cosine) ;
        - pour chaque pivot, construction des buckets avec argsort + unique ;
        - sélection des meilleurs pivots selon l'écart-type des tailles de buckets.
        """
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

        n_samples = X.shape[0]

        self.X_norm = X / (
            np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        )

        similarity_matrix = self.X_norm @ self.X_norm.T

        bucket_matrix = np.round(
            similarity_matrix * self.bucket_scale
        ).astype(np.int16)

        self.bucket_index = {}

        best_candidates = []  # tuples : (pivot_std, pivot_id)
        n_buckets = 2 * self.bucket_scale + 1

        for pivot_id in range(n_samples):
            row_buckets = bucket_matrix[pivot_id]

            # Tri une seule fois de la ligne.
            order = np.argsort(row_buckets)
            sorted_buckets = row_buckets[order]

            # Groupes de buckets identiques.
            unique_buckets, start_positions, counts = np.unique(
                sorted_buckets,
                return_index=True,
                return_counts=True,
            )

            # Dictionnaire uniquement pour les buckets non vides.
            pivot_buckets = {}

            for bucket_value, start, count in zip(
                unique_buckets,
                start_positions,
                counts,
            ):
                pivot_buckets[int(bucket_value)] = order[start: start + count]

            # Distribution des tailles des buckets.
            # Les buckets absents ont une taille 0.
            bucket_sizes = np.zeros(n_buckets, dtype=np.float32)
            bucket_positions = unique_buckets.astype(np.int32) + self.bucket_scale
            bucket_sizes[bucket_positions] = counts.astype(np.float32)

            pivot_std = float(np.std(bucket_sizes))

            # On garde seulement les meilleurs pivots rencontrés.
            best_candidates.append((pivot_std, pivot_id))

            if len(best_candidates) > self.n_best_pivots:
                best_candidates = sorted(
                    best_candidates,
                    key=lambda item: item[0],
                )[: self.n_best_pivots]

            self.bucket_index[pivot_id] = pivot_buckets

        self.best_vectors = np.array(
            [pivot_id for _, pivot_id in best_candidates],
            dtype=np.int64,
        )

        return self

    def refine_candidate_subset(
        self,
        q_norm: np.ndarray,
        candidate_subset: np.ndarray,
    ) -> np.ndarray:
        """
        Niveau 1.

        Réduit le sous-ensemble courant de candidats.

        Étapes :
        1. Choisir un pivot intelligent parmi candidate_subset.
        2. Calculer le bucket de q par rapport à ce pivot.
        3. Récupérer bucket_index[pivot_id][query_bucket].
        4. Intersecter avec candidate_subset.
        5. Retourner le nouveau sous-ensemble.

        Si l'intersection est vide, on conserve l'ancien sous-ensemble.
        """
        if len(candidate_subset) == 0:
            return candidate_subset

        pivot_id = self._choose_best_pivot(candidate_subset)
        pivot_vector = self.X_norm[pivot_id]

        query_bucket = int(np.round(
            self.bucket_scale * float(q_norm @ pivot_vector)
        ))

        bucket_candidates = self.bucket_index[pivot_id].get(
            query_bucket,
            np.array([], dtype=np.int64),
        )

        if len(bucket_candidates) == 0:
            return candidate_subset

        refined_subset = np.intersect1d(
            candidate_subset,
            bucket_candidates,
            assume_unique=True,
        )

        if len(refined_subset) == 0:
            return candidate_subset

        return refined_subset

    def search(self, q: np.ndarray, k: int = 10):
        """
        Niveau 2.

        Appelle refine_candidate_subset n_refinement_steps fois,
        puis retourne les k plus proches parmi les candidats restants.
        """
        if self.X_norm is None or self.bucket_index is None:
            raise RuntimeError("The method must be fitted before search.")

        q = np.asarray(q, dtype=np.float32)

        if q.ndim != 1:
            raise ValueError("q must be a 1D vector.")

        q_norm = q / (np.linalg.norm(q) + 1e-12)

        n_samples = self.X_norm.shape[0]
        k = min(k, n_samples)

        candidate_subset = np.arange(n_samples, dtype=np.int64)

        for _ in range(self.n_refinement_steps):
            if len(candidate_subset) <= k:
                break

            candidate_subset = self.refine_candidate_subset(
                q_norm=q_norm,
                candidate_subset=candidate_subset,
            )

        if len(candidate_subset) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        final_scores = self.X_norm[candidate_subset] @ q_norm

        top_size = min(k, len(candidate_subset))

        if self.rerank_final:
            top_positions = np.argsort(-final_scores)[:top_size]
            return candidate_subset[top_positions], final_scores[top_positions]

        return candidate_subset[:top_size], final_scores[:top_size]

    def _choose_best_pivot(self, candidate_subset: np.ndarray) -> int:
        """
        Choisit un pivot parmi les meilleurs pivots pré-calculés.

        Si un des best_vectors appartient au sous-ensemble courant,
        on l'utilise.

        Sinon, on choisit aléatoirement dans candidate_subset.
        """
        mask = np.isin(self.best_vectors, candidate_subset)

        if np.any(mask):
            return int(self.best_vectors[np.argmax(mask)])

        return int(self.rng.choice(candidate_subset))