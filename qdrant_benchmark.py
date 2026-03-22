#!/usr/bin/env python3
"""
Qdrant Benchmark Pipeline — Ingests datasets and measures recall/latency.

Usage:
    # Smoke test (10K vectors, local Docker)
    python qdrant_benchmark.py --target docker --sample 0.01 --dataset both

    # Full test (1M vectors)
    python qdrant_benchmark.py --target docker --sample 1.0 --dataset both

    # Cloud with sampling
    QDRANT_URL=https://... QDRANT_API_KEY=... \\
    python qdrant_benchmark.py --target cloud --sample 0.05 --dataset dbpedia

Dependencies: pip install qdrant-client datasets numpy
"""

import argparse
import json
import os
import sys
import time
from itertools import islice

import numpy as np
from datasets import load_dataset
from qdrant_client import QdrantClient, models


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_client(target):
    if target == "cloud":
        url = os.environ.get("QDRANT_URL")
        api_key = os.environ.get("QDRANT_API_KEY")
        if not url or not api_key:
            print("ERROR: QDRANT_URL and QDRANT_API_KEY must be set for cloud target.")
            sys.exit(1)
        client = QdrantClient(url=url, api_key=api_key, timeout=180)
    else:
        client = QdrantClient(host="localhost", port=6333, timeout=180)

    # Verify connection
    try:
        client.get_collections()
        print(f"Connected to Qdrant ({target})")
    except Exception as e:
        print(f"ERROR: Cannot connect to Qdrant ({target}): {e}")
        sys.exit(1)

    return client


# ---------------------------------------------------------------------------
# Collection Creation — These configs ARE the architectural proof
# ---------------------------------------------------------------------------

def create_collection_dbpedia(client):
    """
    Scenario 1: Search Team — OpenAI dbpedia, 3072 dims
    Architecture: Scalar quantization (int8), Cosine distance
    Rationale: 95% recall at top-100 allows scalar quantization with rescoring.
    Quantized vectors in RAM (~2.9 GB), full vectors on disk via mmap.
    """
    name = "dbpedia_openai"

    if client.collection_exists(name):
        print(f"  Collection '{name}' already exists, skipping creation.")
        return name

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=3072,
            distance=models.Distance.COSINE,
            on_disk=True,          # Full vectors on disk (mmap for rescoring)
        ),
        hnsw_config=models.HnswConfigDiff(
            m=20,                  # RELAXED regime + top-k adjustment (>=50 bumps one tier)
            ef_construct=200,      # RELAXED default
        ),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=0.99,     # Clip outliers at 1st/99th percentile
                always_ram=True,   # Keep quantized vectors in RAM for fast search
            ),
        ),
        optimizers_config=models.OptimizersConfigDiff(
            memmap_threshold=20000,   # Promote segments to mmap after 20K points
        ),
    )
    print(f"  Created collection '{name}' (Cosine, scalar int8, m=20, ef_construct=200)")
    return name


def create_collection_gist(client):
    """
    Scenario 2: Data Science Team — gist-960, 960 dims
    Architecture: No quantization (full float32 in RAM), Euclidean distance
    Rationale: 99% recall at top-10 demands full precision. Classical CV features
    (GIST) are quantization-resistant. 3.8 GB of vectors fits comfortably in RAM.
    """
    name = "gist_960"

    if client.collection_exists(name):
        print(f"  Collection '{name}' already exists, skipping creation.")
        return name

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=960,
            distance=models.Distance.EUCLID,
            on_disk=False,         # Full vectors in RAM (no quantization layer)
        ),
        hnsw_config=models.HnswConfigDiff(
            m=24,                  # STRICT regime + top-k adjustment (<=20 drops one tier)
            ef_construct=400,      # STRICT default (batch exception would be 400 too)
        ),
    )
    print(f"  Created collection '{name}' (Euclid, no quantization, m=24, ef_construct=400)")
    return name


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_dbpedia(sample_rate):
    """Load OpenAI dbpedia dataset from HuggingFace with streaming."""
    print(f"  Loading dbpedia dataset (sample={sample_rate})...")
    ds = load_dataset(
        "Supabase/dbpedia-openai-3-large-1M",
        split="train",
        streaming=True,
    )

    total = int(1_000_000 * sample_rate)
    step = max(1, int(1 / sample_rate)) if sample_rate < 1.0 else 1

    count = 0
    for i, row in enumerate(ds):
        if sample_rate < 1.0 and i % step != 0:
            continue
        if count >= total:
            break
        yield models.PointStruct(
            id=count,
            vector=row["embedding"],
            payload={"title": row.get("title", ""), "text": row.get("text", "")[:200]},
        )
        count += 1

    print(f"  Loaded {count} dbpedia vectors")


def load_gist(sample_rate):
    """Load gist-960 dataset from HuggingFace. Returns (train_points, test_vecs, neighbors)."""
    print(f"  Loading gist-960 dataset (sample={sample_rate})...")

    # Train split — vectors to ingest
    # gist-960 uses config names "train", "test", "neighbors" (not "default")
    ds_train = load_dataset("open-vdb/gist-960-euclidean", "train", split="train")
    total = int(len(ds_train) * sample_rate)

    # Determine the embedding column name
    emb_cols = [c for c in ds_train.column_names if c not in ("id", "idx")]
    emb_col = emb_cols[0] if emb_cols else ds_train.column_names[0]
    print(f"  Train columns: {ds_train.column_names}, using '{emb_col}' for embeddings")

    def train_points():
        step = max(1, int(1 / sample_rate)) if sample_rate < 1.0 else 1
        indices = range(0, len(ds_train), step)
        count = 0
        for idx in indices:
            if count >= total:
                break
            row = ds_train[idx]
            vec = row[emb_col]
            yield models.PointStruct(id=count, vector=vec, payload={})
            count += 1
        print(f"  Loaded {count} gist train vectors")

    # Test split — query vectors
    ds_test = load_dataset("open-vdb/gist-960-euclidean", "test", split="test")
    test_emb_col = [c for c in ds_test.column_names if c not in ("id", "idx")]
    test_emb_col = test_emb_col[0] if test_emb_col else ds_test.column_names[0]
    test_vectors = [row[test_emb_col] for row in ds_test]
    print(f"  Test columns: {ds_test.column_names}, using '{test_emb_col}'")

    # Neighbors split — ground truth
    try:
        ds_neighbors = load_dataset("open-vdb/gist-960-euclidean", "neighbors", split="neighbors")
        print(f"  Neighbors columns: {ds_neighbors.column_names}")
        neighbor_col = [c for c in ds_neighbors.column_names if "neighbor" in c.lower() or "idx" in c.lower()]
        if neighbor_col:
            ground_truth = [row[neighbor_col[0]] for row in ds_neighbors]
        else:
            ground_truth = [row[ds_neighbors.column_names[0]] for row in ds_neighbors]
        print(f"  Loaded {len(test_vectors)} test vectors, {len(ground_truth)} ground truth rows")
    except Exception as e:
        print(f"  WARNING: Could not load ground truth neighbors: {e}")
        print(f"  Will use exact search for ground truth instead.")
        ground_truth = None

    return train_points, test_vectors, ground_truth


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest(client, collection_name, point_iterator, batch_size=256):
    """Batch upsert points into collection with progress logging."""
    batch = []
    total = 0
    t0 = time.time()

    for point in point_iterator:
        batch.append(point)
        if len(batch) >= batch_size:
            client.upsert(collection_name=collection_name, points=batch)
            total += len(batch)
            batch = []
            if total % 10000 == 0:
                elapsed = time.time() - t0
                rate = total / elapsed if elapsed > 0 else 0
                print(f"    {total:>8,} vectors ingested ({rate:.0f}/s)")

    if batch:
        client.upsert(collection_name=collection_name, points=batch)
        total += len(batch)

    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else 0
    print(f"    Done: {total:,} vectors in {elapsed:.1f}s ({rate:.0f}/s)")

    # Wait for indexing to complete
    print(f"    Waiting for indexing to complete...")
    wait_for_indexing(client, collection_name)
    return total


def wait_for_indexing(client, collection_name, timeout=600):
    """Wait until the collection is fully indexed."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        info = client.get_collection(collection_name)
        if info.status == models.CollectionStatus.GREEN:
            print(f"    Indexing complete.")
            return
        time.sleep(2)
    print(f"    WARNING: Indexing did not complete within {timeout}s")


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_dbpedia(client, collection_name, num_queries=200, k=100):
    """
    Benchmark recall@100 for dbpedia. No pre-computed ground truth,
    so we use exact search to establish it, then compare HNSW results.
    """
    print(f"\n  Benchmarking {collection_name} (recall@{k}, {num_queries} queries)...")

    # Sample query vectors from the collection
    info = client.get_collection(collection_name)
    total_points = info.points_count
    if total_points == 0:
        print("    ERROR: Collection is empty.")
        return None

    query_ids = np.random.choice(total_points, size=min(num_queries, total_points), replace=False)

    # Fetch the actual vectors for these points
    fetched = client.retrieve(collection_name, ids=query_ids.tolist(), with_vectors=True)
    query_vectors = [(p.id, p.vector) for p in fetched if p.vector is not None]

    if not query_vectors:
        print("    ERROR: Could not retrieve vectors for benchmarking.")
        return None

    recalls = []
    latencies = []

    for qid, qvec in query_vectors:
        # Ground truth: exact search
        exact_results = client.query_points(
            collection_name=collection_name,
            query=qvec,
            limit=k,
            search_params=models.SearchParams(exact=True),
        ).points
        gt_ids = set(p.id for p in exact_results)

        # HNSW search with our architecture params
        t0 = time.perf_counter()
        hnsw_results = client.query_points(
            collection_name=collection_name,
            query=qvec,
            limit=k,
            search_params=models.SearchParams(
                hnsw_ef=200,
                quantization=models.QuantizationSearchParams(
                    rescore=True,
                    oversampling=2.0,
                ),
            ),
        ).points
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        hnsw_ids = set(p.id for p in hnsw_results)
        recall = len(gt_ids & hnsw_ids) / len(gt_ids) if gt_ids else 0
        recalls.append(recall)

    return _format_results(collection_name, recalls, latencies, k, len(query_vectors))


def benchmark_gist(client, collection_name, test_vectors, ground_truth,
                   sample_rate, k=10, num_queries=500):
    """
    Benchmark recall@10 for gist-960. Uses pre-computed ground truth
    neighbors if available, otherwise falls back to exact search.
    """
    print(f"\n  Benchmarking {collection_name} (recall@{k}, up to {num_queries} queries)...")

    num_queries = min(num_queries, len(test_vectors))
    use_ground_truth = ground_truth is not None and len(ground_truth) >= num_queries

    recalls = []
    latencies = []

    for i in range(num_queries):
        qvec = test_vectors[i]

        if use_ground_truth:
            gt_neighbors = ground_truth[i]
            # Ground truth neighbors are indices into the full dataset.
            # If we sampled, we need to check which neighbors are actually in our collection.
            # For full ingestion (sample=1.0), indices match point IDs directly.
            if sample_rate < 1.0:
                # With sampling, point IDs may not match ground truth indices.
                # Fall back to exact search for sampled runs.
                exact_results = client.query_points(
                    collection_name=collection_name,
                    query=qvec,
                    limit=k,
                    search_params=models.SearchParams(exact=True),
                ).points
                gt_ids = set(p.id for p in exact_results)
            else:
                gt_ids = set(int(n) for n in gt_neighbors[:k])
        else:
            exact_results = client.query_points(
                collection_name=collection_name,
                query=qvec,
                limit=k,
                search_params=models.SearchParams(exact=True),
            ).points
            gt_ids = set(p.id for p in exact_results)

        # HNSW search with our architecture params
        t0 = time.perf_counter()
        hnsw_results = client.query_points(
            collection_name=collection_name,
            query=qvec,
            limit=k,
            search_params=models.SearchParams(hnsw_ef=128),
        ).points
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        hnsw_ids = set(p.id for p in hnsw_results)
        recall = len(gt_ids & hnsw_ids) / len(gt_ids) if gt_ids else 0
        recalls.append(recall)

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{num_queries} queries done...")

    return _format_results(collection_name, recalls, latencies, k, num_queries)


def _format_results(collection_name, recalls, latencies, k, num_queries):
    recall_mean = np.mean(recalls)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    total_time = sum(latencies) / 1000  # seconds
    qps = num_queries / total_time if total_time > 0 else 0

    results = {
        "collection": collection_name,
        f"recall@{k}": round(recall_mean, 4),
        "p50_ms": round(p50, 2),
        "p99_ms": round(p99, 2),
        "est_qps": round(qps, 0),
        "num_queries": num_queries,
    }

    print(f"\n  Results for {collection_name}:")
    print(f"    Recall@{k}:    {recall_mean:.4f}")
    print(f"    P50 latency:   {p50:.2f} ms")
    print(f"    P99 latency:   {p99:.2f} ms")
    print(f"    Est. QPS:      {qps:.0f} (single-threaded)")
    print(f"    Queries run:   {num_queries}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Qdrant Benchmark Pipeline")
    parser.add_argument("--target", choices=["docker", "cloud"], default="docker",
                        help="Qdrant target: docker (localhost:6333) or cloud (env vars)")
    parser.add_argument("--sample", type=float, default=0.01,
                        help="Fraction of dataset to ingest (0.01=1%%, 1.0=full)")
    parser.add_argument("--dataset", choices=["dbpedia", "gist", "both"], default="both",
                        help="Which dataset(s) to process")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingestion, only run benchmarks on existing collections")
    parser.add_argument("--num-queries", type=int, default=200,
                        help="Number of benchmark queries to run")
    args = parser.parse_args()

    print("=" * 60)
    print("  Qdrant Benchmark Pipeline")
    print("=" * 60)
    print(f"  Target:     {args.target}")
    print(f"  Sample:     {args.sample} ({int(1_000_000 * args.sample):,} vectors per dataset)")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Skip ingest: {args.skip_ingest}")
    print()

    client = get_client(args.target)
    all_results = []

    # --- dbpedia ---
    if args.dataset in ("dbpedia", "both"):
        print("\n--- Scenario 1: dbpedia_openai (3072d, Cosine, scalar int8) ---")
        collection = create_collection_dbpedia(client)

        if not args.skip_ingest:
            points = load_dbpedia(args.sample)
            ingest(client, collection, points)

        results = benchmark_dbpedia(
            client, collection,
            num_queries=args.num_queries,
            k=100,
        )
        if results:
            all_results.append(results)

    # --- gist-960 ---
    if args.dataset in ("gist", "both"):
        print("\n--- Scenario 2: gist_960 (960d, Euclid, no quantization) ---")
        collection = create_collection_gist(client)

        if not args.skip_ingest:
            train_fn, test_vectors, ground_truth = load_gist(args.sample)
            ingest(client, collection, train_fn())
        else:
            _, test_vectors, ground_truth = load_gist(1.0)  # need test vectors for benchmark

        results = benchmark_gist(
            client, collection, test_vectors, ground_truth,
            sample_rate=args.sample,
            k=10,
            num_queries=min(args.num_queries, len(test_vectors)),
        )
        if results:
            all_results.append(results)

    # --- Summary ---
    if all_results:
        print("\n" + "=" * 60)
        print("  BENCHMARK SUMMARY")
        print("=" * 60)
        print(json.dumps(all_results, indent=2))
        print("=" * 60)


if __name__ == "__main__":
    main()
