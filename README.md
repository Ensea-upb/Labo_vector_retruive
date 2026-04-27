# Vector Lab

**A reproducible benchmark for approximate nearest neighbor (ANN) retrieval — including two experimental methods of my own.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## Why this repo

Modern RAG systems live or die on their retrieval layer. The choice between exact search, ScaNN, HNSW, IVF-PQ and friends has direct consequences on latency, recall and memory cost — and on the quality of the answers a downstream LLM can produce.

Vector Lab is a small experimental harness to **measure those tradeoffs on the same data, with the same query set, under the same metrics.** It compares:

| Method | Type | Source |
|---|---|---|
| **Brute Force** | Exact | Reference baseline (cosine similarity) |
| **ScaNN** | ANN — quantization + scoring | [Google Research, 2020](https://github.com/google-research/google-research/tree/master/scann) |
| **HNSW** | ANN — graph | [Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320) |
| **SatelliteSearch** | ANN — anchor/satellite (experimental, original) | This repo |
| **DictBucketRecursiveSearch** | ANN — quantized cosine buckets (experimental, original) | This repo |

The two experimental methods are not meant to beat HNSW at scale — they are research probes to understand what tradeoffs anchor-based and bucket-based decompositions actually buy you on small to medium corpora.

---

## Quick start

```bash
git clone https://github.com/Ensea-upb/Labo_vector_retruive.git
cd Labo_vector_retruive
pip install -r requirements.txt

# Run all benchmarks on the default dataset
python scripts/run_benchmarks.py --dataset sift1m --k 10

# Compare a single pair
python scripts/run_benchmarks.py --methods hnsw,satellite --dataset sift1m --k 10
```

Outputs land in `outputs/benchmarks/` as CSV and as plots (recall@k vs queries-per-second).

---

## Methods at a glance

### Brute Force
Linear scan, exact cosine similarity. Reference for **recall = 1.0** at any k. Slow on large `n`, but the only ground truth.

### ScaNN
Google's anisotropic quantization + asymmetric hashing. Strong baseline on medium-to-large datasets, especially when memory is constrained.

### HNSW
Hierarchical Navigable Small World graph. Industry default in most modern vector databases (Qdrant, Weaviate, Milvus, pgvector). Very high recall at low query latency, at the cost of memory.

### SatelliteSearch *(original)*
Selects a set of anchor vectors and routes each query to a small set of "satellite" anchors before comparing only inside the relevant satellite clusters. Trades recall for a smaller comparison set.

### DictBucketRecursiveSearch *(original)*
Pre-computes a relational dictionary of quantized cosine similarities and recursively narrows the candidate set by bucket lookups. Heavier preprocessing, lighter query-time.

Each method exposes the same interface: `build(vectors)`, `search(query, k)`. Adding a new method = subclassing `BaseRetriever`.

---

## Results (preliminary)

> Replace this table with your actual benchmark results. Format kept consistent with the ann-benchmarks community.

	method| build_time_sec| mean_recall_at_k| mean_latency_ms| p95_latency_ms| memory_delta_mb
brute_force |2.3239990696310997e-06| 1.0| 0.06786782061681151| 0.0731652064132504| 0.0
scann| 0.5335132439940935| 0.7440000000000002| 0.04240015958203003| 0.05543890147237107| 18.0234375
hnsw_spacecosine_M16_efc200_efs100| 0.36301516099774744| 0.9720000000000002| 0.1819357210479211| 0.36130860171397206| 5.76953125
satellite_balltree| 0.29890566899848636| 0.8660000000000001| 7.586552749853581| 67.62494116264861| 2.7421875
dict_bucket_recursive_scale1000_steps5_best20| 4.62648250699567| 0.0| 1.2018083412840497| 0.8171628011041315| 677.01953125

Plots in `outputs/benchmarks/`.

---

## Repo layout

```
vector-lab/
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── ann_benchmarks/
│   └── raw/
├── src/
│   ├── retrievers/      # BaseRetriever + one file per method
│   ├── benchmarks/      # recall, latency, memory measurement
│   └── utils/
├── scripts/
│   └── run_benchmarks.py
└── outputs/
    └── benchmarks/      # CSV + plots
```

---

## Roadmap

- [ ] Wire ann-benchmarks-compatible datasets (SIFT-1M, GIST, GloVe).
- [ ] Add IVF-PQ and FAISS-flat as additional baselines.
- [ ] Sweep configurations of HNSW (M, efConstruction, ef) for a fair Pareto frontier.
- [ ] Quantify the build-time / query-time / memory tradeoff explicitly.
- [ ] Try the two experimental methods on real-world embedding distributions (sentence-transformers on Wikipedia, Cohere on web).
- [ ] Publish a short write-up of the SatelliteSearch and DictBucketRecursiveSearch ideas with toy proofs.

---

## Why this exists

Built after a seminar where a speaker presented their production RAG system and the design choices behind its retrieval layer. The conversation that followed convinced me to stop reading about ANN methods and start measuring them. Two weekends later, this repo.

Work in progress, feedback welcome.

---

## Contact

**Soro Inza Ouada** — M2 ENSAE Paris (Data Science · Generative AI · Econometrics)
inzaouada.soro@ensae.fr · github.com/Ensea-upb

Looking for: 6-month end-of-studies internship from June 2026, or apprenticeship — focus on RAG, retrieval, applied LLM systems.

