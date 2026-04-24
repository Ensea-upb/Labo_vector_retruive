# src/vector_lab/core/metrics.py

import numpy as np


def recall_at_k(exact_idx: np.ndarray, approx_idx: np.ndarray, k: int) -> float:
    """
    Recall@K mesure la proportion des vrais top-K voisins retrouvés
    par la méthode approximative.

    Exemple :
    exact top-10   = [1, 2, 3, 4, 5, ...]
    approx top-10  = [1, 3, 8, 9, 10, ...]

    Si 2 vrais voisins sont retrouvés sur 10 :
    Recall@10 = 2 / 10 = 0.2
    """
    exact_set = set(exact_idx[:k].tolist())
    approx_set = set(approx_idx[:k].tolist())

    return len(exact_set.intersection(approx_set)) / k


def precision_at_k(exact_idx: np.ndarray, approx_idx: np.ndarray, k: int) -> float:
    """
    Precision@K mesure la proportion des résultats retournés
    qui appartiennent vraiment au top-K exact.

    Dans notre cas, comme approx_idx contient aussi K éléments,
    Precision@K est souvent égale au Recall@K.

    Mais on garde les deux métriques car elles ont des interprétations différentes.
    """
    exact_set = set(exact_idx[:k].tolist())
    approx_set = set(approx_idx[:k].tolist())

    return len(exact_set.intersection(approx_set)) / len(approx_set)


def mrr_at_k(exact_idx: np.ndarray, approx_idx: np.ndarray, k: int) -> float:
    """
    MRR@K = Mean Reciprocal Rank.

    Cette métrique regarde à quelle position apparaît le premier résultat correct.

    Si le premier résultat correct est en position 1 :
    MRR = 1

    Si le premier résultat correct est en position 2 :
    MRR = 1/2

    Si le premier résultat correct est en position 5 :
    MRR = 1/5

    Si aucun bon résultat n'apparaît dans les K premiers :
    MRR = 0
    """
    exact_set = set(exact_idx[:k].tolist())

    for rank, idx in enumerate(approx_idx[:k], start=1):
        if idx in exact_set:
            return 1.0 / rank

    return 0.0


def compute_retrieval_metrics(
    exact_idx: np.ndarray,
    approx_idx: np.ndarray,
    k: int,
) -> dict:
    """
    Calcule les trois métriques principales pour une requête donnée.
    """
    return {
        f"recall_at_{k}": recall_at_k(exact_idx, approx_idx, k),
        f"precision_at_{k}": precision_at_k(exact_idx, approx_idx, k),
        f"mrr_at_{k}": mrr_at_k(exact_idx, approx_idx, k),
    }


def mean_metrics(metrics_list: list[dict]) -> dict:
    """
    Agrège les métriques sur plusieurs requêtes.

    Exemple :
    metrics_list = [
        {"recall_at_10": 0.8, "precision_at_10": 0.8, "mrr_at_10": 1.0},
        {"recall_at_10": 0.6, "precision_at_10": 0.6, "mrr_at_10": 0.5},
    ]

    Retour :
    {
        "recall_at_10": 0.7,
        "precision_at_10": 0.7,
        "mrr_at_10": 0.75
    }
    """
    if len(metrics_list) == 0:
        return {}

    keys = metrics_list[0].keys()

    return {
        key: float(np.mean([m[key] for m in metrics_list]))
        for key in keys
    }