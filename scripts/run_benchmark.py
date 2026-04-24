import os
import pandas as pd

from vector_lab.data.generators import make_random_unit_vectors, make_query_vectors
from vector_lab.data.ann_benchmarks import load_glove_100_angular

from vector_lab.benchmark.evaluator import benchmark_many
from vector_lab.methods.brute_force import BruteForceSearch
from vector_lab.methods.scann_index import ScannSearch
from vector_lab.methods.satellite import SatelliteSearch
from vector_lab.methods.pivot_bucket import DictBucketRecursiveSearch
from vector_lab.methods.hnsw_index import HNSWSearch


def main():
    n = 5_000
    d = 128
    n_queries = 1_00
    k = 10

    print(f"Generating data: n={n}, d={d}, queries={n_queries}")
    X, Q, official_neighbors,_ = load_glove_100_angular(
        path="data/ann_benchmarks/glove-100-angular.hdf5",
        n_train=n,
        n_queries=n_queries,
        normalize=True,
    )

    methods = [
            BruteForceSearch(),
        
            ScannSearch(
                num_neighbors=k,
                num_leaves=200,
                num_leaves_to_search=20,
                reorder=100,
            ),
        
            HNSWSearch(
                space="cosine",
                M=16,
                ef_construction=200,
                ef_search=100,
                seed=0,
            ),
        
            SatelliteSearch(
                n_satellites=32,
                n_candidates=1000,
                strategy="farthest",
                seed=0,
            ),
        
            DictBucketRecursiveSearch(
                bucket_scale=1000,
                n_refinement_steps=5,
                n_best_pivots=20,
                rerank_final=True,
                seed=0,
            ),
        ]

    results = benchmark_many(methods, X=X, Q=Q, k=k)

    print("\nBenchmark results:")
    print(results)

    os.makedirs("outputs/benchmarks", exist_ok=True)
    output_path = "outputs/benchmarks/benchmark_results.csv"
    results.to_csv(output_path, index=False)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()