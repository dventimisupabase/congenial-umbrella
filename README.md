# pgvector Sizing Tools

Interactive sizing wizard and field guide for PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) deployments. Input your workload requirements, get an architecture recommendation with instance selection, PostgreSQL configuration, and generated SQL.

## Tools

| Asset | Description |
|-------|-------------|
| [**Sizing Wizard** (web)](https://congenial-umbrella-ashy.vercel.app) | Interactive wizard — input your requirements, get an architecture |
| [**Field Guide**](pgvector-field-guide.md) | Decision-tree reference for sizing any pgvector deployment ([view as styled page](https://congenial-umbrella-ashy.vercel.app/field-guide.html)) |

## What the Sizer Produces

- **Instance selection**: RDS (M-series, R-series) or self-managed EC2 (C-series) with cost estimates
- **PostgreSQL configuration**: `shared_buffers`, `effective_cache_size`, `maintenance_work_mem`, `work_mem`, parallel workers, and pgvector-specific settings (`hnsw.ef_search` or `ivfflat.probes`)
- **Index strategy**: HNSW or IVFFlat with tuned parameters based on your recall target
- **Vector type**: `vector` (float32) or `halfvec` (float16) based on embedding type and recall requirements
- **Generated SQL**: Ready-to-run DDL for table, index, and example query

## Example Scenarios

Two illustrative scenarios are included in `scenarios.json`:

| | Scenario 1: Text Search | Scenario 2: CV Features |
|--|--|--|
| Dataset | OpenAI dbpedia (3072d) | gist-960 (960d) |
| Recall target | 95% @100 | 99% @10 |
| QPS / P99 | 1,000 / 50ms | 3,000 / 500ms |
| Vector type | halfvec (2x compression) | vector (full float32) |

## Benchmark

A benchmark script is included to validate sizing decisions against real pgvector performance:

```bash
# Install dependencies
pip install psycopg2-binary datasets numpy

# Start Supabase locally (includes pgvector)
npx supabase init && npx supabase start

# Run benchmark against local Supabase
python pgvector_benchmark.py --target local --sample 0.01 --dataset both

# Run against Supabase Cloud
DATABASE_URL="postgresql://..." \
python pgvector_benchmark.py --target cloud --sample 0.05 --dataset both
```

## Project Structure

| File | Role |
|------|------|
| `sizer_engine.js` | Core sizing logic (ES module) — all constants, decision tables, and pipeline |
| `index.html` | Interactive web wizard that imports `sizer_engine.js` |
| `field-guide.html` | Styled viewer for the field guide markdown |
| `pgvector-field-guide.md` | Decision-tree reference for sizing pgvector deployments |
| `pgvector_benchmark.py` | Benchmark pipeline — ingests datasets, measures recall/latency |
| `scenarios.json` | Example scenario inputs |
