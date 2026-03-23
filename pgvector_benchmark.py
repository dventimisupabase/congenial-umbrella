#!/usr/bin/env python3
"""
pgvector Benchmark — Ingest + recall/latency measurement.

Tests pgvector performance with two datasets:
  1. OpenAI dbpedia (3072d, neural text, cosine)
  2. GIST-960 (960d, classical CV, L2)

Targets:
  --target local   → local Supabase/Postgres (default: postgresql://postgres:postgres@localhost:54322/postgres)
  --target cloud   → Supabase Cloud (set DATABASE_URL env var)
  --target custom  → custom connection (set DATABASE_URL env var)

Usage:
  pip install psycopg2-binary datasets numpy

  # Local Supabase (start with: npx supabase start)
  python pgvector_benchmark.py --target local --sample 0.01 --dataset both

  # Supabase Cloud
  DATABASE_URL="postgresql://..." python pgvector_benchmark.py --target cloud --sample 0.05 --dataset both
"""

import argparse
import io
import os
import sys
import time
import json
import numpy as np

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    print("Install psycopg2-binary: pip install psycopg2-binary")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("Install datasets: pip install datasets")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DATASETS = {
    "dbpedia": {
        "hf_name": "Supabase/dbpedia-openai-3-large-1M",
        "split": "train",
        "vec_col": "embedding",
        "dimensions": 3072,
        "distance": "cosine",
        "table": "bench_dbpedia",
        "ops_class": "vector_cosine_ops",
        "dist_op": "<=>",
        "target_recall": 0.95,
        "top_k": 100,
        "hnsw_m": 16,
        "hnsw_ef_construction": 128,
        "hnsw_ef_search": 200,
    },
    "gist960": {
        "hf_name": "open-vdb/gist-960-euclidean",
        "hf_config": "train",
        "split": "train",
        "vec_col": "emb",
        "dimensions": 960,
        "distance": "l2",
        "table": "bench_gist960",
        "ops_class": "vector_l2_ops",
        "dist_op": "<->",
        "target_recall": 0.99,
        "top_k": 10,
        "hnsw_m": 24,
        "hnsw_ef_construction": 256,
        "hnsw_ef_search": 128,
    },
}


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection(target):
    if target == "local":
        dsn = os.environ.get(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:54422/postgres",
        )
    else:
        dsn = os.environ.get("DATABASE_URL")
        if not dsn:
            print("Set DATABASE_URL for cloud/custom targets.")
            sys.exit(1)

    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    return conn


def ensure_pgvector(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def create_table(conn, cfg):
    table = cfg["table"]
    dims = cfg["dimensions"]
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table};")
        cur.execute(f"""
            CREATE TABLE {table} (
                id bigserial PRIMARY KEY,
                embedding vector({dims})
            );
        """)
        # STORAGE PLAIN avoids TOAST overhead but only works when row fits in 8KB page
        # vector overhead: 4 * dims + 8 bytes, plus ~36 bytes tuple overhead
        row_bytes = 4 * dims + 8 + 36
        if row_bytes <= 8160:
            cur.execute(f"ALTER TABLE {table} ALTER COLUMN embedding SET STORAGE PLAIN;")
            print(f"  Created table {table} (vector({dims}), STORAGE PLAIN)")
        else:
            print(f"  Created table {table} (vector({dims}), STORAGE EXTERNAL — row too large for PLAIN)")
            cur.execute(f"ALTER TABLE {table} ALTER COLUMN embedding SET STORAGE EXTERNAL;")


def create_index(conn, cfg):
    table = cfg["table"]
    ops = cfg["ops_class"]
    m = cfg["hnsw_m"]
    efc = cfg["hnsw_ef_construction"]
    dims = cfg["dimensions"]
    idx_name = f"{table}_embedding_idx"

    # HNSW has a 2000 dimension limit for vector type.
    # For higher dimensions, use expression index with halfvec.
    use_halfvec_index = dims > 2000
    if use_halfvec_index:
        halfvec_ops = cfg.get("halfvec_ops_class", ops.replace("vector_", "halfvec_"))
        col_expr = f"(embedding::halfvec({dims}))"
        index_ops = halfvec_ops
    else:
        col_expr = "embedding"
        index_ops = ops

    with conn.cursor() as cur:
        cur.execute("SET maintenance_work_mem = '128MB';")
        cur.execute("SET max_parallel_maintenance_workers = 0;")

        suffix = " (halfvec expression index)" if use_halfvec_index else ""
        print(f"  Building HNSW index (m={m}, ef_construction={efc}){suffix}...")
        t0 = time.time()
        cur.execute(f"""
            CREATE INDEX {idx_name} ON {table}
            USING hnsw ({col_expr} {index_ops})
            WITH (m = {m}, ef_construction = {efc});
        """)
        elapsed = time.time() - t0
        print(f"  Index built in {elapsed:.1f}s")

        cur.execute("ANALYZE {};".format(table))

    # Store whether we used halfvec index so queries can cast appropriately
    cfg["_use_halfvec_index"] = use_halfvec_index


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def load_vectors(cfg, sample_frac):
    print(f"  Loading {cfg['hf_name']} (sample={sample_frac})...")
    hf_config = cfg.get("hf_config")
    if hf_config:
        ds = load_dataset(cfg["hf_name"], hf_config, split=cfg["split"])
    else:
        ds = load_dataset(cfg["hf_name"], split=cfg["split"])

    n_total = len(ds)
    n_sample = max(int(n_total * sample_frac), 100)
    if n_sample < n_total:
        ds = ds.shuffle(seed=42).select(range(n_sample))

    vecs = np.array(ds[cfg["vec_col"]], dtype=np.float32)
    print(f"  Loaded {len(vecs)} vectors ({cfg['dimensions']}d)")
    return vecs


def ingest(conn, cfg, vectors, target="local"):
    table = cfg["table"]
    n = len(vectors)

    # Check how many rows already exist (resume after partial ingest)
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}")
        existing = cur.fetchone()[0]

    if existing > 0:
        print(f"  Resuming ingest: {existing:,} rows already present", flush=True)
        vectors = vectors[existing:]

    n_total = existing + len(vectors)
    n_remaining = len(vectors)
    if n_remaining == 0:
        print(f"  All {n_total:,} vectors already ingested", flush=True)
        return conn

    print(f"  Ingesting {n_remaining:,} vectors into {table} via COPY...", flush=True)

    t0 = time.time()
    chunk_size = 50000
    max_retries = 5
    offset = 0

    while offset < n_remaining:
        end = min(offset + chunk_size, n_remaining)
        chunk = vectors[offset:end]

        buf = io.StringIO()
        for v in chunk:
            buf.write("[" + ",".join(f"{x:.6f}" for x in v) + "]\n")
        buf.seek(0)

        for attempt in range(max_retries):
            try:
                with conn.cursor() as cur:
                    cur.copy_expert(
                        f"COPY {table} (embedding) FROM STDIN WITH (FORMAT text)",
                        buf,
                    )
                break
            except (psycopg2.OperationalError, psycopg2.errors.ReadOnlySqlTransaction) as e:
                if attempt < max_retries - 1:
                    wait = 30 * (attempt + 1)
                    print(f"    Retry in {wait}s ({attempt + 2}/{max_retries}): {e}", flush=True)
                    try:
                        conn.close()
                    except Exception:
                        pass
                    time.sleep(wait)
                    conn = get_connection(target)
                    conn.autocommit = True
                    buf.seek(0)
                else:
                    raise

        done = existing + end
        elapsed_so_far = time.time() - t0
        rate = end / elapsed_so_far if elapsed_so_far > 0 else 0
        pct = done / n_total * 100
        print(f"    {pct:.0f}% ({done:,}/{n_total:,}) — {rate:.0f} vec/s", flush=True)
        offset = end

    elapsed = time.time() - t0
    rate = n_remaining / elapsed if elapsed > 0 else 0
    print(f"  Ingested {n_remaining:,} vectors in {elapsed:.1f}s ({rate:.0f} vec/s)")
    return conn


# ---------------------------------------------------------------------------
# Benchmark: Recall + Latency
# ---------------------------------------------------------------------------

def compute_ground_truth(conn, cfg, query_vecs, top_k):
    """Exact brute-force search (no index) for ground truth."""
    table = cfg["table"]
    dist_op = cfg["dist_op"]

    print(f"  Computing ground truth (brute-force, {len(query_vecs)} queries)...")
    ground_truth = []

    with conn.cursor() as cur:
        # Disable index for brute-force
        cur.execute("SET enable_indexscan = off;")
        cur.execute("SET enable_bitmapscan = off;")

        for qv in query_vecs:
            vec_str = "[" + ",".join(f"{x:.6f}" for x in qv) + "]"
            cur.execute(
                f"SELECT id FROM {table} ORDER BY embedding {dist_op} %s::vector LIMIT %s",
                (vec_str, top_k),
            )
            ids = [row[0] for row in cur.fetchall()]
            ground_truth.append(set(ids))

        cur.execute("SET enable_indexscan = on;")
        cur.execute("SET enable_bitmapscan = on;")

    return ground_truth


def benchmark_queries(conn, cfg, query_vecs, ground_truth, top_k):
    """ANN search (with index) and measure recall + latency."""
    table = cfg["table"]
    dist_op = cfg["dist_op"]
    ef_search = cfg["hnsw_ef_search"]
    use_halfvec = cfg.get("_use_halfvec_index", False)
    dims = cfg["dimensions"]

    # When using halfvec expression index, queries must cast to match
    if use_halfvec:
        cast_type = f"halfvec({dims})"
        order_expr = f"embedding::halfvec({dims})"
    else:
        cast_type = "vector"
        order_expr = "embedding"

    latencies = []
    recalls = []

    with conn.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = {ef_search};")

        for i, qv in enumerate(query_vecs):
            vec_str = "[" + ",".join(f"{x:.6f}" for x in qv) + "]"

            t0 = time.perf_counter()
            cur.execute(
                f"SELECT id FROM {table} ORDER BY {order_expr} {dist_op} %s::{cast_type} LIMIT %s",
                (vec_str, top_k),
            )
            ids = set(row[0] for row in cur.fetchall())
            elapsed_ms = (time.perf_counter() - t0) * 1000

            latencies.append(elapsed_ms)
            recall = len(ids & ground_truth[i]) / len(ground_truth[i]) if ground_truth[i] else 1.0
            recalls.append(recall)

    return latencies, recalls


def run_benchmark(conn, cfg, vectors, n_queries=200):
    top_k = cfg["top_k"]

    # Use last n_queries vectors as query set
    n_queries = min(n_queries, len(vectors) // 10, 500)
    query_vecs = vectors[-n_queries:]

    # Ground truth
    ground_truth = compute_ground_truth(conn, cfg, query_vecs, top_k)

    # ANN benchmark
    print(f"  Running ANN benchmark ({n_queries} queries, top-{top_k}, ef_search={cfg['hnsw_ef_search']})...")
    latencies, recalls = benchmark_queries(conn, cfg, query_vecs, ground_truth, top_k)

    latencies_arr = np.array(latencies)
    recalls_arr = np.array(recalls)

    results = {
        "dataset": cfg["table"],
        "n_vectors": len(vectors),
        "n_queries": n_queries,
        "top_k": top_k,
        "target_recall": cfg["target_recall"],
        "hnsw_ef_search": cfg["hnsw_ef_search"],
        "recall_mean": float(np.mean(recalls_arr)),
        "recall_min": float(np.min(recalls_arr)),
        "p50_ms": float(np.percentile(latencies_arr, 50)),
        "p95_ms": float(np.percentile(latencies_arr, 95)),
        "p99_ms": float(np.percentile(latencies_arr, 99)),
        "mean_ms": float(np.mean(latencies_arr)),
        "est_qps_single_thread": float(1000 / np.mean(latencies_arr)) if np.mean(latencies_arr) > 0 else 0,
    }

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_results(results):
    print()
    print("=" * 60)
    print(f"  BENCHMARK RESULTS — {results['dataset']}")
    print("=" * 60)
    print(f"  Vectors:            {results['n_vectors']:,}")
    print(f"  Queries:            {results['n_queries']}")
    print(f"  Top-k:              {results['top_k']}")
    print(f"  hnsw.ef_search:     {results['hnsw_ef_search']}")
    print()
    recall_pct = results['recall_mean'] * 100
    target_pct = results['target_recall'] * 100
    status = "PASS" if results['recall_mean'] >= results['target_recall'] else "FAIL"
    print(f"  Recall (mean):      {recall_pct:.2f}%  (target: {target_pct:.0f}%) [{status}]")
    print(f"  Recall (min):       {results['recall_min'] * 100:.2f}%")
    print()
    print(f"  P50 latency:        {results['p50_ms']:.2f} ms")
    print(f"  P95 latency:        {results['p95_ms']:.2f} ms")
    print(f"  P99 latency:        {results['p99_ms']:.2f} ms")
    print(f"  Mean latency:       {results['mean_ms']:.2f} ms")
    print(f"  Est. QPS (1 thread):{results['est_qps_single_thread']:.0f}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Ensure real-time output even when piped/redirected
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="pgvector benchmark")
    parser.add_argument("--target", choices=["local", "cloud", "custom"], default="local")
    parser.add_argument("--dataset", choices=["dbpedia", "gist960", "both"], default="both")
    parser.add_argument("--sample", type=float, default=0.01, help="Fraction of dataset to use (0.01 = 1%%)")
    parser.add_argument("--queries", type=int, default=200, help="Number of queries for benchmark")
    parser.add_argument("--resume", action="store_true", help="Resume ingest without dropping tables")
    args = parser.parse_args()

    datasets_to_run = (
        ["dbpedia", "gist960"] if args.dataset == "both" else [args.dataset]
    )

    conn = get_connection(args.target)
    ensure_pgvector(conn)
    print(f"Connected to PostgreSQL ({args.target})")

    all_results = []

    for ds_name in datasets_to_run:
        cfg = DATASETS[ds_name]
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name} ({cfg['dimensions']}d, {cfg['distance']})")
        print(f"{'='*60}")

        vectors = load_vectors(cfg, args.sample)
        if not args.resume:
            create_table(conn, cfg)
        conn = ingest(conn, cfg, vectors, target=args.target)
        create_index(conn, cfg)
        results = run_benchmark(conn, cfg, vectors, n_queries=args.queries)
        print_results(results)
        all_results.append(results)

    # Write results to JSON
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("Results written to benchmark_results.json")

    conn.close()


if __name__ == "__main__":
    main()
