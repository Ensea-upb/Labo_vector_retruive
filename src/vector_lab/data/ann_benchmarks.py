# src/vector_lab/data/ann_benchmarks.py

import h5py
import numpy as np

from vector_lab.core.distances import normalize_rows


def load_glove_100_angular(
    path: str = "data/ann_benchmarks/glove-100-angular.hdf5",
    n_train: int | None = None,
    n_queries: int | None = None,
    normalize: bool = True,
):
    """
    Charge le benchmark GloVe-100 angular depuis ANN-Benchmarks.

    Le fichier HDF5 contient généralement :
        - train      : vecteurs de base
        - test       : vecteurs requêtes
        - neighbors  : ground truth indices
        - distances  : ground truth distances

    Pour angular/cosine, on normalise les vecteurs afin que le produit scalaire
    corresponde à la similarité cosinus.
    """
    with h5py.File(path, "r") as f:
        X = np.asarray(f["train"][:], dtype=np.float32)
        Q = np.asarray(f["test"][:], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"][:], dtype=np.int64)
        distances = np.asarray(f["distances"][:], dtype=np.float32)

    if n_train is not None:
        X = X[:n_train]

        # Important : si on tronque X, certains voisins ground truth peuvent
        # pointer vers des indices au-delà de n_train.
        # Pour une évaluation rigoureuse avec ground truth officiel,
        # il faut garder tout X.
        # Pour debug, on peut tronquer, mais le recall officiel n'est plus valide.
    
    if n_queries is not None:
        Q = Q[:n_queries]
        neighbors = neighbors[:n_queries]
        distances = distances[:n_queries]

    if normalize:
        X = normalize_rows(X).astype(np.float32)
        Q = normalize_rows(Q).astype(np.float32)

    return X, Q, neighbors, distances
