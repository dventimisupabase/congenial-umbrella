# pgvector Sizing Tools

Architecture sizing toolkit for [pgvector](https://github.com/pgvector/pgvector) on [Supabase](https://supabase.com). Input your workload requirements — vector count, dimensions, recall target, QPS — and get a complete deployment architecture: Supabase compute tier, PostgreSQL configuration, HNSW/IVFFlat index strategy, and ready-to-run SQL.

**[Launch the tools](https://congenial-umbrella-ashy.vercel.app)**

## What's Here

### Interactive Sizing Wizard

A step-by-step web wizard that walks you through your workload parameters and produces a complete architecture recommendation.

**[Open Sizing Wizard](https://congenial-umbrella-ashy.vercel.app/sizer.html)**

What it produces:
- **Supabase compute tier** (Micro through 16XL) with monthly cost estimate
- **PostgreSQL configuration** — `shared_buffers` (sized to hold the HNSW index, not the standard 25%), `effective_cache_size`, `maintenance_work_mem`, `work_mem`, parallel workers
- **HNSW index parameters** — `m` (with top-k adjustment), `ef_construction`, `ef_search` (with scale adjustment for 1M+ vectors)
- **Vector type** — `vector` (float32), `halfvec` (float16), or binary quantization, based on embedding class and recall target
- **Generated SQL** — ready-to-run DDL for table, index, storage mode, and example query

The sizer engine is calibrated against real benchmarks at 1M vectors on Supabase Cloud (index size predictions within 10% of measured, CPU cost model from EXPLAIN ANALYZE).

### Architecture Field Guide

A 15-stage decision-tree reference for sizing any pgvector deployment. Start at Stage 1 and work through sequentially — by the end you have a concrete, defensible architecture.

**[Read the Field Guide](https://congenial-umbrella-ashy.vercel.app/field-guide.html)** (or [view raw markdown](pgvector-field-guide.md))

Covers: embedding classification, recall regimes, HNSW vs IVFFlat, halfvec feasibility, binary quantization, latency budgets, memory sizing, disk sizing, compute sizing, Supabase tier selection, PostgreSQL configuration, scaling strategies, operational considerations, and real-world benchmark results.

### Benchmark Pipeline

A Python benchmark script for validating sizing decisions against real pgvector performance. Streams data from HuggingFace, ingests via COPY, builds indexes, and measures recall + latency with pre-computed ground truth where available.

```bash
pip install psycopg2-binary datasets numpy tqdm

# Smoke test (10K vectors, local Supabase)
python pgvector_benchmark.py --target local --sample 0.01 --dataset both

# Full test (1M vectors, Supabase Cloud)
DATABASE_URL="postgresql://..." \
python pgvector_benchmark.py --target cloud --sample 1.0 --dataset both

# Benchmark only (skip ingest, sample queries from DB)
DATABASE_URL="postgresql://..." \
python pgvector_benchmark.py --target cloud --skip-ingest --dataset both
```

## Key Findings

These results come from benchmarking 1,000,000 vectors on Supabase Cloud across multiple compute tiers.

### shared_buffers Must Hold the Index

The HNSW index **must** fit entirely in `shared_buffers` for acceptable query performance. The standard PostgreSQL recommendation of 25% of RAM is wrong for vector workloads.

| Cache state | Buffer reads | Query latency |
|---|---|---|
| Warm (index in shared_buffers) | `shared hit=4839, read=0` | **21 ms** |
| Cold (index on disk) | `shared hit=3261, read=1578` | **834 ms** |

**40-60x latency degradation** when the index doesn't fit. Use `pg_prewarm` after every restart.

### ef_search Is the Primary Recall Tuning Knob

`hnsw.ef_search` controls the recall/latency trade-off at query time. The sizer produces a starting value, but the right ef_search for your workload depends on your data distribution and can only be determined by benchmarking. Here's what we measured at 1M vectors:

| ef_search | gist-960 Recall@10 | dbpedia Recall@100 |
|---|---|---|
| 128 | 96.2% | 98.2% |
| 200 | 98.5% | 98.7% |
| **300** | **99.1%** | 99.0% |
| 400 | 99.6% | 99.1% |

For gist-960's 99% recall target, ef_search=300 was the minimum. For dbpedia's 95% target, ef_search=128 was already sufficient. The sizer accounts for this scale dependence, but **always benchmark at your target vector count** — recall behavior at 10K vectors does not predict recall at 1M.

### HNSW vs IVFFlat

At 1M vectors, HNSW is 4.8x faster at 99% recall but IVFFlat builds 7x faster and uses 49% less space.

| | HNSW (m=24) | IVFFlat (lists=1000) |
|---|---|---|
| Index size | 7,678 MB | 3,912 MB |
| Build time | ~35 min | ~5 min |
| 99% recall latency | 100 ms | 479 ms |

### Sizer Engine Accuracy

The sizer's index size formula is calibrated against measured values:

| Dataset | Predicted | Actual | Error |
|---|---|---|---|
| dbpedia (3072d halfvec, m=16) | 7,935 MB | 7,813 MB | 2% |
| gist-960 (960d vector, m=24) | 7,050 MB | 7,678 MB | 9% |

## Do You Actually Need All This?

The sizer and field guide are designed for workloads with **specific SLA targets** — sustained QPS, latency budgets, recall constraints. But data from ~60,000 Supabase organizations shows that most real vector workloads are much simpler:

- **93% have fewer than 1M vector rows.** Median DB size at 100K–1M rows is 2.1 GB.
- **At 1M–10M rows, 60% run on Micro or Small.** They don't need 1K QPS — they're doing 10–100 queries/second alongside their regular application.
- **If the index fits in shared_buffers and QPS is modest (< 100), almost any tier works.**

Use the full sizing pipeline when you have defined QPS/latency/recall targets for a production vector search feature. For a RAG chatbot, product search, or prototype — start on a smaller tier and scale up if needed. The field guide has a [Real-World Usage Patterns](pgvector-field-guide.md#real-world-usage-patterns) section with more detail.

## Test Scenarios

Two scenarios are included in `scenarios.json` and used throughout the field guide:

| | Scenario 1: Text Search | Scenario 2: CV Features |
|---|---|---|
| Dataset | [OpenAI dbpedia 1M](https://huggingface.co/datasets/Supabase/dbpedia-openai-3-large-1M) | [gist-960 1M](https://huggingface.co/datasets/open-vdb/gist-960-euclidean) |
| Dimensions | 3,072 | 960 |
| Distance | Cosine | L2 (Euclidean) |
| Index | HNSW halfvec expression | HNSW vector (float32) |
| Recall target | 95% @100 | 99% @10 |
| Measured recall | 98.3% | 99.6% |
| Server latency (warm) | 26 ms | 21 ms |
| Supabase tier | 4XL (16 vCPU, 64 GB) | 4XL (16 vCPU, 64 GB) |

## Project Structure

| File | Role |
|------|------|
| `pgvector-field-guide.md` | Architecture field guide — 15 stages, 18 decision-tree diagrams, empirical benchmark data |
| `sizer_engine.js` | Core sizing logic (ES module) — calibrated formulas, Supabase tier catalog, full pipeline |
| `pgvector_benchmark.py` | Benchmark pipeline — streaming ingest, COPY, recall/latency measurement, pre-computed ground truth |
| `sizer.html` | Interactive sizing wizard (imports `sizer_engine.js`) |
| `field-guide.html` | Styled field guide viewer (renders markdown with mermaid diagrams) |
| `index.html` | Landing page |
| `scenarios.json` | Example scenario inputs |

## Supabase-Specific Notes

- **Compute tiers**: Micro through 16XL (2-core shared to 64-core dedicated)
- **shared_buffers**: Configurable via CLI (`supabase postgres-config update`), requires restart
- **Disk**: Auto-scales at 90% capacity (1.5x expansion, 6h cooldown). Enters read-only at 95% if hit between expansions. Pre-size disk before large imports.
- **Connection pooling**: Supavisor in session mode (port 5432) required for `SET` commands; transaction mode (port 6543) resets session state
- **pg_prewarm**: Available on all tiers — use it after every restart or compute tier change
