# Vector Lab

Laboratoire expérimental pour comparer des méthodes de recherche de plus proches voisins approximatifs (**ANN, Approximate Nearest Neighbor**) sur des espaces vectoriels.

L’objectif du projet est double :

1. **comparer des méthodes standards** comme **Brute Force**, **ScaNN** et **HNSW** ;
2. **développer et tester de nouvelles méthodes personnelles**, en particulier une méthode fondée sur un **dictionnaire relationnel de similarité cosinus quantifiée**.

---

## Objectif scientifique

Dans un problème de retrieval vectoriel, on dispose :

- d’une base de vecteurs :

\[
x_1, x_2, \dots, x_n \in \mathbb{R}^d
\]

- d’une requête :

\[
q \in \mathbb{R}^d
\]

et on cherche les \(k\) vecteurs les plus proches de \(q\), sans comparer \(q\) à tous les vecteurs de la base lorsque \(n\) est grand.

Ce projet permet de comparer plusieurs stratégies :

- **Brute Force** : recherche exacte ;
- **ScaNN** : baseline Google pour la recherche vectorielle rapide ;
- **HNSW** : baseline de type graphe, utilisée dans de nombreuses bases vectorielles modernes ;
- **SatelliteSearch** : méthode expérimentale fondée sur des ancres/satellites ;
- **DictBucketRecursiveSearch** : méthode personnelle basée sur des buckets de similarité pré-calculés.

---

## Structure du projet

```text
vector-lab/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── ann_benchmarks/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│
├── outputs/
│   ├── benchmarks/
│   ├── figures/
│   └── indexes/
│
├── notebooks/
│
├── scripts/
│   └── run_benchmark.py
│
└── src/
    └── vector_lab/
        ├── __init__.py
        │
        ├── core/
        │   ├── distances.py
        │   ├── metrics.py
        │   └── timer.py
        │
        ├── data/
        │   ├── generators.py
        │   ├── real_text.py
        │   └── ann_benchmarks.py
        │
        ├── methods/
        │   ├── base.py
        │   ├── brute_force.py
        │   ├── scann_index.py
        │   ├── hnsw_index.py
        │   ├── satellite.py
        │   └── dict_bucket.py
        │
        ├── benchmark/
        │   ├── evaluator.py
        │   └── report.py
        │
        └── experiments/
            └── configs.py
