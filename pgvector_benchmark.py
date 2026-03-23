#!/usr/bin/env python3
"""
pgvector Benchmark Pipeline — Ingests datasets and measures recall/latency.

Adapted from qdrant_benchmark.py for PostgreSQL + pgvector on Supabase.

Usage:
    # Smoke test (10K vectors, local Supabase)
    python pgvector_benchmark.py --target local --sample 0.01 --dataset both

    # Full test (1M vectors)
    python pgvector_benchmark.py --target local --sample 1.0 --dataset both

    # Cloud with sampling
    DATABASE_URL="postgresql://..." \\
    python pgvector_benchmark.py --target cloud --sample 0.05 --dataset dbpedia

    # Benchmark only (skip ingest, sample queries from DB)
    DATABASE_URL="postgresql://..." \\
    python pgvector_benchmark.py --target cloud --skip-ingest --dataset both

Dependencies: pip install psycopg2-binary datasets numpy tqdm
"""

import argparse
import io
import json
import os
import sys
import time

import numpy as np

try:
    import psycopg2
    import psycopg2.errors
except ImportError:
    print("Install psycopg2-binary: pip install psycopg2-binary")
    sys.exit(1)

try:
    from datasets import load_dataset
except ImportError:
    print("Install datasets: pip install datasets")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# Configuration — following alloy's config-object pattern
# ---------------------------------------------------------------------------

class ScenarioConfig:
    """Encapsulates table, index, and search configuration for a scenario."""

    def __init__(self, table_name, dimensions, distance_op, ops_class,
                 hnsw_m, ef_construction, ef_search, target_recall, top_k,
                 use_halfvec_index=False, halfvec_ops_class=None):
        self.table_name = table_name
        self.dimensions = dimensions
        self.distance_op = distance_op
        self.ops_class = ops_class
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.target_recall = target_recall
        self.top_k = top_k
        # HNSW has a 2000-dimension limit for vector type.
        # For higher dimensions, use expression index with halfvec.
        self.use_halfvec_index = use_halfvec_index or dimensions > 2000
        self.halfvec_ops_class = halfvec_ops_class or ops_class.replace(
            "vector_", "halfvec_")


# Scenario 1: OpenAI dbpedia — 3072d, cosine, 95% recall @100
# Uses halfvec expression index (3072d exceeds HNSW's 2000d limit).
DBPEDIA_CONFIG = ScenarioConfig(
    table_name="bench_dbpedia",
    dimensions=3072,
    distance_op="<=>",
    ops_class="vector_cosine_ops",
    hnsw_m=16,
    ef_construction=128,
    ef_search=200,
    target_recall=0.95,
    top_k=100,
)

# Scenario 2: gist-960 — 960d, L2, 99% recall @10
# Full float32 vectors, no halfvec needed.
GIST_CONFIG = ScenarioConfig(
    table_name="bench_gist960",
    dimensions=960,
    distance_op="<->",
    ops_class="vector_l2_ops",
    hnsw_m=24,
    ef_construction=256,
    ef_search=128,
    target_recall=0.99,
    top_k=10,
)


# ---------------------------------------------------------------------------
# Connection — environment-driven with sensible defaults
# ---------------------------------------------------------------------------

DATABASE_URL_LOCAL = os.getenv(
    "DATABASE_URL_LOCAL",
    "postgresql://postgres:postgres@localhost:54422/postgres",
)


def _get_dsn(target):
    if target == "local":
        return os.environ.get("DATABASE_URL", DATABASE_URL_LOCAL)
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: Set DATABASE_URL for cloud/custom targets.")
        sys.exit(1)
    return dsn


def _setup_postgres(target):
    """Initialize PostgreSQL connection and ensure pgvector is available."""
    dsn = _get_dsn(target)
    conn = psycopg2.connect(dsn)
    conn.autocommit = True

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("SELECT version();")
        version = cur.fetchone()[0].split(" on ")[0]

    print(f"Connected to {version} ({target})")
    return conn


# ---------------------------------------------------------------------------
# Table Creation
# ---------------------------------------------------------------------------

def _create_table(conn, config):
    """Create the benchmark table. Drops if exists."""
    table = config.table_name
    dims = config.dimensions

    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table};")
        cur.execute(f"""
            CREATE TABLE {table} (
                id bigserial PRIMARY KEY,
                embedding vector({dims})
            );
        """)

        # STORAGE PLAIN avoids TOAST overhead when row fits in 8KB page
        row_bytes = 4 * dims + 8 + 36
        if row_bytes <= 8160:
            cur.execute(f"ALTER TABLE {table} ALTER COLUMN embedding SET STORAGE PLAIN;")
            storage = "PLAIN"
        else:
            cur.execute(f"ALTER TABLE {table} ALTER COLUMN embedding SET STORAGE EXTERNAL;")
            storage = "EXTERNAL"

    print(f"  Created table {table} (vector({dims}), STORAGE {storage})")


def _table_exists(conn, config):
    """Check if the table exists and has data."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
            (config.table_name,),
        )
        exists = cur.fetchone()[0]
        if not exists:
            return False, 0
        cur.execute(f"SELECT count(*) FROM {config.table_name}")
        count = cur.fetchone()[0]
        return True, count


# ---------------------------------------------------------------------------
# Data Loading — streaming batches, constant memory
# ---------------------------------------------------------------------------

def _prepare_dbpedia_batches(sample_rate, batch_size=256):
    """Stream dbpedia from HuggingFace, yield batches of (vector_text,).
    Uses streaming=True to avoid downloading all 17.8 GB upfront."""
    print(f"  Loading dbpedia dataset (sample={sample_rate})...")
    ds = load_dataset(
        "Supabase/dbpedia-openai-3-large-1M",
        split="train",
        streaming=True,
    )

    num_records = int(1_000_000 * sample_rate)
    step = max(1, int(1 / sample_rate)) if sample_rate < 1.0 else 1

    batch = []
    count = 0
    for i, row in enumerate(ds):
        if sample_rate < 1.0 and i % step != 0:
            continue
        if count >= num_records:
            break

        vec = row["embedding"]
        vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
        batch.append(vec_str)
        count += 1

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch

    print(f"  Prepared {count} dbpedia vectors in batches of {batch_size}")


def _prepare_gist_batches(sample_rate, batch_size=256):
    """Load gist-960 train data, yield batches of vector text."""
    print(f"  Loading gist-960 dataset (sample={sample_rate})...")
    ds_train = load_dataset("open-vdb/gist-960-euclidean", "train", split="train")

    num_records = int(len(ds_train) * sample_rate)
    step = max(1, int(1 / sample_rate)) if sample_rate < 1.0 else 1

    # Auto-detect embedding column
    emb_cols = [c for c in ds_train.column_names if c not in ("id", "idx")]
    emb_col = emb_cols[0] if emb_cols else ds_train.column_names[0]
    print(f"  Using '{emb_col}' for embeddings")

    batch = []
    count = 0
    for idx in range(0, len(ds_train), step):
        if count >= num_records:
            break
        row = ds_train[idx]
        vec = row[emb_col]
        vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
        batch.append(vec_str)
        count += 1

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch

    print(f"  Prepared {count} gist vectors in batches of {batch_size}")


def _load_gist_test_data():
    """Load gist-960 test vectors and ground truth neighbors."""
    ds_test = load_dataset("open-vdb/gist-960-euclidean", "test", split="test")
    test_emb_col = [c for c in ds_test.column_names if c not in ("id", "idx")]
    test_emb_col = test_emb_col[0] if test_emb_col else ds_test.column_names[0]
    test_vectors = [row[test_emb_col] for row in ds_test]
    print(f"  Loaded {len(test_vectors)} test vectors (column: '{test_emb_col}')")

    ground_truth = None
    try:
        ds_neighbors = load_dataset(
            "open-vdb/gist-960-euclidean", "neighbors", split="neighbors")
        ground_truth = [row["neighbors_id"] for row in ds_neighbors]
        print(f"  Loaded {len(ground_truth)} ground truth rows "
              f"({len(ground_truth[0])} neighbors each)")
    except Exception as e:
        print(f"  WARNING: Could not load ground truth neighbors: {e}")
        print(f"  Will use exact search for ground truth instead.")

    return test_vectors, ground_truth


# ---------------------------------------------------------------------------
# Ingestion — COPY-based, with retry for Supabase disk auto-scale
# ---------------------------------------------------------------------------

def _insert_batches(conn, config, batch_generator, batch_size, target):
    """Insert batches via COPY. Retries on transient errors."""
    table = config.table_name
    total = 0
    start_time = time.time()
    max_retries = 5

    for batch in tqdm(batch_generator, desc=f"  Inserting into {table}"):
        buf = io.StringIO()
        for vec_str in batch:
            buf.write(vec_str + "\n")
        buf.seek(0)

        for attempt in range(max_retries):
            try:
                with conn.cursor() as cur:
                    cur.copy_expert(
                        f"COPY {table} (embedding) FROM STDIN WITH (FORMAT text)",
                        buf,
                    )
                total += len(batch)
                break
            except (psycopg2.OperationalError,
                    psycopg2.errors.ReadOnlySqlTransaction,
                    psycopg2.errors.DiskFull) as e:
                if attempt < max_retries - 1:
                    wait = 30 * (attempt + 1)
                    print(f"    Retry in {wait}s ({attempt + 2}/{max_retries}): {e}",
                          flush=True)
                    try:
                        conn.close()
                    except Exception:
                        pass
                    time.sleep(wait)
                    conn = _setup_postgres(target)
                    buf.seek(0)
                else:
                    raise

        if total % 50000 < batch_size:
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0
            print(f"    {total:>10,} vectors ({rate:.0f}/s)", flush=True)

    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0
    print(f"  Inserting time: {elapsed:.1f}s ({total:,} vectors, {rate:.0f}/s)")
    return conn, total


# ---------------------------------------------------------------------------
# Index Creation
# ---------------------------------------------------------------------------

def _create_index(conn, config, target="local"):
    """Build HNSW index. Uses halfvec expression index for >2000d vectors."""
    table = config.table_name
    idx_name = f"{table}_embedding_idx"
    m = config.hnsw_m
    efc = config.ef_construction
    dims = config.dimensions

    if config.use_halfvec_index:
        col_expr = f"(embedding::halfvec({dims}))"
        index_ops = config.halfvec_ops_class
    else:
        col_expr = "embedding"
        index_ops = config.ops_class

    # Local Docker has limited shared memory; cloud instances have more
    maint_mem = "128MB" if target == "local" else "4GB"
    workers = 0 if target == "local" else 3

    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = 0;")
        cur.execute(f"SET maintenance_work_mem = '{maint_mem}';")
        cur.execute(f"SET max_parallel_maintenance_workers = {workers};")

        # Drop existing index if any
        cur.execute(f"DROP INDEX IF EXISTS {idx_name};")

        suffix = " (halfvec expression)" if config.use_halfvec_index else ""
        print(f"  Building HNSW index (m={m}, ef_construction={efc}){suffix}...",
              flush=True)
        t0 = time.time()
        cur.execute(f"""
            CREATE INDEX {idx_name} ON {table}
            USING hnsw ({col_expr} {index_ops})
            WITH (m = {m}, ef_construction = {efc});
        """)
        elapsed = time.time() - t0
        print(f"  Index built in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

        cur.execute(f"ANALYZE {table};")


def _index_exists(conn, config):
    """Check if the HNSW index exists."""
    idx_name = f"{config.table_name}_embedding_idx"
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = %s)",
            (idx_name,),
        )
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Query Vector Sampling
# ---------------------------------------------------------------------------

def _sample_query_vectors(conn, config, num_queries):
    """Sample query vectors directly from the database.
    Mirrors the Qdrant approach of client.retrieve(with_vectors=True)."""
    table = config.table_name
    print(f"  Sampling {num_queries} query vectors from {table}...", flush=True)

    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = 0;")
        cur.execute(
            f"SELECT id, embedding::text FROM {table} "
            f"TABLESAMPLE BERNOULLI (1) LIMIT %s",
            (num_queries,),
        )
        rows = cur.fetchall()

    query_vectors = []
    for row_id, vec_str in rows:
        vals = [float(x) for x in vec_str.strip("[]").split(",")]
        query_vectors.append((row_id, vals))

    print(f"  Sampled {len(query_vectors)} query vectors")
    return query_vectors


# ---------------------------------------------------------------------------
# Search Execution
# ---------------------------------------------------------------------------

def _execute_search(conn, config, query_vector, top_k):
    """Execute HNSW search with configured ef_search."""
    table = config.table_name
    dist_op = config.distance_op
    dims = config.dimensions

    if config.use_halfvec_index:
        cast_type = f"halfvec({dims})"
        order_expr = f"embedding::halfvec({dims})"
    else:
        cast_type = "vector"
        order_expr = "embedding"

    vec_str = "[" + ",".join(f"{x:.6f}" for x in query_vector) + "]"

    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id FROM {table} ORDER BY {order_expr} {dist_op} "
            f"%s::{cast_type} LIMIT %s",
            (vec_str, top_k),
        )
        return [row[0] for row in cur.fetchall()]


def _execute_exact_search(conn, config, query_vector, top_k):
    """Execute brute-force exact search for ground truth."""
    table = config.table_name
    dist_op = config.distance_op
    vec_str = "[" + ",".join(f"{x:.6f}" for x in query_vector) + "]"

    with conn.cursor() as cur:
        cur.execute("SET enable_indexscan = off;")
        cur.execute("SET enable_bitmapscan = off;")
        cur.execute(
            f"SELECT id FROM {table} ORDER BY embedding {dist_op} "
            f"%s::vector LIMIT %s",
            (vec_str, top_k),
        )
        results = [row[0] for row in cur.fetchall()]
        cur.execute("SET enable_indexscan = on;")
        cur.execute("SET enable_bitmapscan = on;")
        return results


def _compute_recall(result_ids, ground_truth_ids):
    if not ground_truth_ids:
        return 0.0
    return len(set(result_ids) & set(ground_truth_ids)) / len(ground_truth_ids)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def _benchmark_dbpedia(conn, config, num_queries=200):
    """Benchmark recall@100 for dbpedia. No pre-computed ground truth,
    so we sample vectors from the DB, compute exact GT, then compare HNSW."""
    print(f"\n  Benchmarking {config.table_name} "
          f"(recall@{config.top_k}, {num_queries} queries)...")

    exists, count = _table_exists(conn, config)
    if not exists or count == 0:
        print("    ERROR: Table is empty.")
        return None

    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = 0;")
        cur.execute(f"SET hnsw.ef_search = {config.ef_search};")

    query_vectors = _sample_query_vectors(conn, config, num_queries)
    if not query_vectors:
        print("    ERROR: Could not sample query vectors.")
        return None

    recalls = []
    latencies = []

    for qid, qvec in tqdm(query_vectors, desc="    Searching"):
        # Ground truth: exact search
        gt_ids = _execute_exact_search(conn, config, qvec, config.top_k)

        # HNSW search
        start_time = time.perf_counter()
        hnsw_ids = _execute_search(conn, config, qvec, config.top_k)
        latency_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(latency_ms)

        recalls.append(_compute_recall(hnsw_ids, gt_ids))

    return _format_results(config, recalls, latencies, count)


def _benchmark_gist(conn, config, test_vectors, ground_truth,
                    sample_rate, num_queries=500):
    """Benchmark recall@10 for gist-960. Uses pre-computed ground truth
    neighbors if available, otherwise falls back to exact search."""
    print(f"\n  Benchmarking {config.table_name} "
          f"(recall@{config.top_k}, up to {num_queries} queries)...")

    exists, count = _table_exists(conn, config)
    if not exists or count == 0:
        print("    ERROR: Table is empty.")
        return None

    num_queries = min(num_queries, len(test_vectors))
    use_ground_truth = (ground_truth is not None
                        and len(ground_truth) >= num_queries
                        and sample_rate >= 1.0)

    if use_ground_truth:
        print(f"  Using pre-computed ground truth (sample_rate={sample_rate})")
    else:
        print(f"  Using exact search for ground truth")

    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = 0;")
        cur.execute(f"SET hnsw.ef_search = {config.ef_search};")

    recalls = []
    latencies = []

    for i in tqdm(range(num_queries), desc="    Searching"):
        qvec = test_vectors[i]

        if use_ground_truth:
            # Full ingestion: ground truth indices match point IDs
            # gist-960 IDs are 0-based, our table IDs are 1-based
            gt_ids = [int(n) + 1 for n in ground_truth[i][:config.top_k]]
        else:
            gt_ids = _execute_exact_search(conn, config, qvec, config.top_k)

        # HNSW search
        start_time = time.perf_counter()
        hnsw_ids = _execute_search(conn, config, qvec, config.top_k)
        latency_ms = (time.perf_counter() - start_time) * 1000
        latencies.append(latency_ms)

        recalls.append(_compute_recall(hnsw_ids, gt_ids))

    return _format_results(config, recalls, latencies, count)


def _format_results(config, recalls, latencies, n_vectors):
    """Format benchmark results and print summary."""
    recall_mean = np.mean(recalls)
    recall_min = np.min(recalls)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mean_ms = np.mean(latencies)
    total_time = sum(latencies) / 1000
    qps = len(latencies) / total_time if total_time > 0 else 0
    status = "PASS" if recall_mean >= config.target_recall else "FAIL"

    results = {
        "table": config.table_name,
        "n_vectors": n_vectors,
        f"recall@{config.top_k}": round(float(recall_mean), 4),
        "recall_min": round(float(recall_min), 4),
        "p50_ms": round(float(p50), 2),
        "p95_ms": round(float(p95), 2),
        "p99_ms": round(float(p99), 2),
        "mean_ms": round(float(mean_ms), 2),
        "est_qps": round(float(qps), 0),
        "num_queries": len(latencies),
        "hnsw_ef_search": config.ef_search,
        "target_recall": config.target_recall,
        "status": status,
    }

    print(f"\n  Results for {config.table_name}:")
    print(f"    Vectors:       {n_vectors:,}")
    print(f"    Recall@{config.top_k}:    "
          f"{recall_mean:.4f} (target: {config.target_recall}) [{status}]")
    print(f"    Recall (min):  {recall_min:.4f}")
    print(f"    P50 latency:   {p50:.2f} ms")
    print(f"    P95 latency:   {p95:.2f} ms")
    print(f"    P99 latency:   {p99:.2f} ms")
    print(f"    Est. QPS:      {qps:.0f} (single-threaded)")
    print(f"    Queries run:   {len(latencies)}")

    return results


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="pgvector Benchmark Pipeline")
    parser.add_argument("--target", choices=["local", "cloud", "custom"],
                        default="local",
                        help="Target: local (Supabase local) or cloud (DATABASE_URL)")
    parser.add_argument("--sample", type=float, default=0.01,
                        help="Fraction of dataset to ingest (0.01=1%%, 1.0=full)")
    parser.add_argument("--dataset", choices=["dbpedia", "gist960", "both"],
                        default="both", help="Which dataset(s) to process")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingestion/indexing, only run benchmarks")
    parser.add_argument("--num-queries", type=int, default=200,
                        help="Number of benchmark queries to run")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Records per batch during ingestion")
    args = parser.parse_args()

    print("=" * 60)
    print("  pgvector Benchmark Pipeline")
    print("=" * 60)
    print(f"  Target:       {args.target}")
    print(f"  Sample:       {args.sample} "
          f"({int(1_000_000 * args.sample):,} vectors per dataset)")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Dataset:      {args.dataset}")
    print(f"  Skip ingest:  {args.skip_ingest}")
    print()

    conn = _setup_postgres(args.target)
    all_results = []

    # --- Scenario 1: dbpedia ---
    if args.dataset in ("dbpedia", "both"):
        print("\n--- Scenario 1: dbpedia (3072d, cosine, halfvec index) ---")

        if not args.skip_ingest:
            _create_table(conn, DBPEDIA_CONFIG)
            batches = _prepare_dbpedia_batches(args.sample, args.batch_size)
            conn, total = _insert_batches(
                conn, DBPEDIA_CONFIG, batches, args.batch_size, args.target)
            _create_index(conn, DBPEDIA_CONFIG, args.target)

        results = _benchmark_dbpedia(
            conn, DBPEDIA_CONFIG, num_queries=args.num_queries)
        if results:
            all_results.append(results)

    # --- Scenario 2: gist-960 ---
    if args.dataset in ("gist960", "both"):
        print("\n--- Scenario 2: gist-960 (960d, L2, full precision) ---")

        if not args.skip_ingest:
            _create_table(conn, GIST_CONFIG)
            batches = _prepare_gist_batches(args.sample, args.batch_size)
            conn, total = _insert_batches(
                conn, GIST_CONFIG, batches, args.batch_size, args.target)
            _create_index(conn, GIST_CONFIG, args.target)

        test_vectors, ground_truth = _load_gist_test_data()
        results = _benchmark_gist(
            conn, GIST_CONFIG, test_vectors, ground_truth,
            sample_rate=args.sample,
            num_queries=min(args.num_queries, 1000),
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

        with open("benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("Results written to benchmark_results.json")

    conn.close()


if __name__ == "__main__":
    main()
