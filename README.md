# Qdrant Cluster Sizing: Senior SA Take-Home

Two Qdrant deployments sized, configured, benchmarked, and costed for a VP of Platform Engineering with 1M-vector workloads on different embedding models.

## The Punchline

The VP assumed the 960-dimension dataset would need significantly less infrastructure than the 3072-dimension dataset. **The opposite is true.** Both land at the same instance class and cost (~$500/mo) because quantization eligibility and recall requirements dominate cost — not raw dimensions.

| | Scenario 1: Search Team | Scenario 2: Data Science Team |
|--|--|--|
| Dataset | OpenAI dbpedia (3072d) | gist-960 (960d) |
| Recall target | 95% @100 | 99% @10 |
| QPS / P99 | 1,000 / 50ms | 3,000 / 500ms |
| Quantization | Scalar int8 (4x compression) | None (full float32) |
| **Measured recall** | **99.6%** (target: 95%) | **99.2%** (target: 99%) |
| Instance | c6i.4xlarge | c6i.4xlarge |
| Cost | ~$500/mo | ~$499/mo |

## Deliverables

| File | What It Is |
|------|-----------|
| [**architecture_writeup.md**](architecture_writeup.md) | Cluster sizing with math, Qdrant configs, benchmark results, cost analysis |
| [**vp_email.md**](vp_email.md) | Executive summary email to the VP |
| [**qdrant_benchmark.py**](qdrant_benchmark.py) | Ingestion + recall/latency benchmark pipeline |

## Run the Benchmark

```bash
# Install dependencies
pip install qdrant-client datasets numpy

# Start Qdrant locally
docker run -d -p 6333:6333 qdrant/qdrant

# Smoke test (10K vectors, ~2 min)
python qdrant_benchmark.py --target docker --sample 0.01 --dataset both

# Full test (1M vectors, ~1 hr)
python qdrant_benchmark.py --target docker --sample 1.0 --dataset both

# Against Qdrant Cloud
QDRANT_URL=https://... QDRANT_API_KEY=... \
python qdrant_benchmark.py --target cloud --sample 0.05 --dataset both
```

## Bonus: Field Guide & Sizer Tools

Built during the process of developing the architecture. These are the reusable frameworks behind the sizing decisions.

| Asset | Description |
|-------|-------------|
| [**Field Guide**](qdrant-field-guide.md) | Decision-tree reference for sizing any Qdrant cluster ([view as styled page](https://vercel-test-mauve-seven-20.vercel.app/field-guide.html)) |
| [**Cluster Sizer (web)**](https://vercel-test-mauve-seven-20.vercel.app) | Interactive wizard — input your requirements, get an architecture |
| [**Cluster Sizer (CLI)**](qdrant_sizer_cli.js) | Same logic as the web sizer: `node qdrant_sizer_cli.js --file scenarios.json` |
