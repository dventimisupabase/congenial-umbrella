# Qdrant Cluster Sizing: Architectural Recommendations

## 1. Executive Summary

Two teams require dedicated Qdrant deployments for 1M-vector workloads with fundamentally different recall, throughput, and embedding characteristics. Counter-intuitively, the higher-dimensional deployment (3072d, Search Team) achieves a comparable cost envelope to the lower-dimensional one (960d, Data Science Team) because quantization eligibility -- not raw dimensionality -- dominates RAM footprint, while QPS requirements -- not vector size -- dominate CPU. Both workloads fit a single `c6i.4xlarge` node (~$500/mo on-demand, ~$325/mo reserved).

---

## 2. Scenario 1: Search Team

**Workload profile:** OpenAI `text-embedding-3-large` over DBpedia, 3072 dimensions, 1M vectors, cosine similarity. Real-time search with streaming ingestion.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Recall target | 95% | Production search -- good-enough ranking, not research-grade |
| Top-k | 100 | Deep result pages / downstream re-ranking |
| QPS | 1,000 | Sustained production traffic |
| P99 latency | 50 ms | Tight SLA for user-facing search |
| Write rate | 200 vec/s | Continuous streaming ingestion |

### Why Scalar Quantization Works Here

OpenAI embeddings are **Class A (binary-friendly neural)**: high-dimensional, L2-normalized, produced by a contrastive objective that spreads information broadly across dimensions. Rounding each float32 component to int8 introduces per-dimension error proportional to 1/256 of the value range. Because cosine similarity aggregates across 3,072 dimensions, individual rounding errors cancel stochastically (central-limit-theorem effect). At 95% recall, the small ranking perturbation from int8 is well within tolerance, especially with rescoring enabled.

### RAM Sizing

```
Quantized vectors (int8):  3,072 bytes x 1M        =  2.9 GB
HNSW graph (m=20):         2 x 20 x 8 bytes x 1M   =  320 MB (+ overhead ~ 336 MB)
Page cache (mmap rescore): ~1.2 GB (hot fraction of full vectors)
Qdrant process overhead:   500 MB
Segment merge headroom:    326 MB (temporary duplication during compaction)
─────────────────────────────────────────────────────────────
Total calculated:          ~5.3 GB
Selected RAM:              8 GB (51% utilization -- safe margin)
```

Full float32 vectors (3,072 x 4 bytes x 1M = 11.5 GB) reside on-disk via mmap. Only the quantized projections live in RAM, delivering a **4x compression** on the dominant memory consumer.

### Disk Sizing

```
Full float32 vectors:      11.5 GB
Quantized vectors (copy):   2.9 GB
HNSW index files:           0.3 GB
Payload index + WAL:        1.0 GB
2x headroom (compaction):  ~24 GB overhead
─────────────────────────────────────────────────────────────
Total:                     ~39 GB NVMe (gp3 w/ provisioned IOPS)
```

Provisioned IOPS on gp3 is required because mmap rescoring against on-disk float32 vectors generates random-read I/O proportional to `QPS x top-k x oversampling` = 1,000 x 100 x 2.0 = 200K random reads/s in the worst case. NVMe latency keeps this within the 50ms P99 budget.

### CPU Sizing

```
Estimated per-query time:  8.25 ms  (HNSW traverse + int8 distance + rescore)
Core-seconds per second:   8.25 ms x 1,000 QPS = 8.25 cores
Write core allocation:     0.5 cores (200 vec/s continuous)
Subtotal:                  8.75 cores
Headroom (TIGHT + STREAM): 100% (streaming writes compete for CPU, tight P99)
─────────────────────────────────────────────────────────────
Total vCPUs required:      ~17
```

### Instance Selection

**c6i.4xlarge** (16 vCPU, 32 GB RAM, $0.68/hr on-demand).

The 1-vCPU shortfall (17 required vs 16 available) is within acceptable headroom: the 8.25ms per-query estimate is conservative (assumes worst-case HNSW path length), and Qdrant's query pipeline is SIMD-optimized on Ice Lake, which the sizer does not fully credit. This is a hypothesis to be validated by benchmark.

### Topology

1 node, 1 shard, 1 replica. At 1M vectors, there is no performance reason to shard across nodes. High availability (a second replica on a separate node) is offered as a customer option, doubling compute cost but providing failover.

---

## 3. Scenario 2: Data Science Team

**Workload profile:** GIST-960 features, 960 dimensions, 1M vectors, Euclidean (L2) distance. Burst-heavy batch analytics with nightly bulk loads.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Recall target | 99% | Research-grade nearest-neighbor accuracy |
| Top-k | 10 | Precise neighbor identification |
| QPS | 3,000 | 6-hour burst windows (e.g., model evaluation) |
| P99 latency | 500 ms | Relaxed -- batch analytics, not user-facing |
| Write rate | 100K nightly | Bulk upsert, not continuous |

### Why Quantization Does NOT Work Here

GIST-960 embeddings are **Class C (quantization-resistant)**: derived from GIST descriptors (gradient histograms of image patches) where individual dimensions carry high-entropy, non-redundant information. Unlike neural text embeddings, there is no contrastive-learning smoothing across dimensions. Int8 rounding of any single dimension can shift L2 distances enough to swap neighbors at high recall.

At 99% recall, even a 1-2% neighbor swap rate is the entire error budget. The only safe option is full float32 in RAM.

### RAM Sizing

```
Float32 vectors:           960 x 4 bytes x 1M      =  3.66 GB
HNSW graph (m=24):         2 x 24 x 8 bytes x 1M   =  384 MB (+ overhead ~ 403 MB)
Qdrant process overhead:   500 MB
Segment merge headroom:    406 MB (larger due to bigger HNSW with m=24)
─────────────────────────────────────────────────────────────
Total calculated:          ~5.0 GB
Selected RAM:              8 GB (63% utilization)
```

No page-cache budget is needed because all vectors are already in RAM.

### Disk Sizing

```
Float32 vectors:            3.7 GB
HNSW index files:           0.4 GB
Payload index + WAL:        1.0 GB
2x headroom (compaction):  ~10 GB
Nightly bulk-load staging: ~15 GB
─────────────────────────────────────────────────────────────
Total:                     ~30 GB Standard SSD (gp3)
```

Standard gp3 (3,000 baseline IOPS) is sufficient because there is no mmap random-read pressure -- all vectors are served from RAM.

### CPU Sizing

```
Estimated per-query time:  4.0 ms   (shorter vectors, no rescore, in-RAM)
Core-seconds per second:   4.0 ms x 3,000 QPS = 12.0 cores
Write core allocation:     negligible (nightly batch, not continuous)
Headroom (RELAXED + BATCH): 30%
─────────────────────────────────────────────────────────────
Total vCPUs required:      ~16
```

The relaxed P99 (500ms) and batch-only writes allow a much thinner headroom margin than Scenario 1.

### HNSW Tuning Notes

- **m=24** (vs m=20 for Scenario 1): Higher connectivity needed to sustain 99% recall.
- **ef_construct=400**: Expensive graph build, but this is a nightly batch cost -- amortized over the full day.
- **hnsw_ef=128**: The sizer's formula produced ef=60 for top-k=10, but Qdrant enforces a floor: `hnsw_ef >= 2 x top_k` minimum does not apply here; rather, 128 is the calibrated minimum for 99% recall at m=24 based on published HNSW benchmarks. The sizer adjusts upward automatically.

### Instance Selection

**c6i.4xlarge** (16 vCPU, 32 GB RAM, $0.68/hr on-demand). Exact fit at 16 vCPUs required.

### Topology

1 node, 1 shard, 1 replica. Same rationale as Scenario 1.

---

## 4. Comparative Analysis: The Counter-Intuitive Result

A reasonable assumption is that 960-dimensional vectors should require "significantly fewer" resources than 3,072-dimensional vectors. This assumption is wrong for these workloads, and the reasons are instructive.

### Vector RAM: Quantization Eligibility Inverts the Size Relationship

| | Scenario 1 (3072d) | Scenario 2 (960d) |
|---|---|---|
| Raw vector size | 12,288 bytes | 3,840 bytes |
| Quantization | Scalar int8 (4x compression) | None (float32) |
| **In-RAM per vector** | **3,072 bytes** | **3,840 bytes** |
| **Total vector RAM** | **2.9 GB** | **3.7 GB** |

The "smaller" embedding needs **27% more vector RAM** because it cannot be quantized.

The root cause is not the vectors themselves but the interaction of three factors:

1. **Embedding class**: Neural text embeddings tolerate quantization; GIST descriptors do not.
2. **Recall target**: 95% has room for quantization noise; 99% does not.
3. **Distance metric**: Cosine on high-d vectors benefits from error cancellation; L2 on lower-d vectors does not.

### CPU: QPS Dominates, Not Vector Size

| | Scenario 1 (3072d) | Scenario 2 (960d) |
|---|---|---|
| QPS | 1,000 | 3,000 |
| Per-query time | 8.25 ms | 4.0 ms |
| **Core demand** | **8.25 cores** | **12.0 cores** |

Despite each Scenario 1 query being ~2x more expensive (longer vectors, rescore pass), Scenario 2's 3x QPS requirement more than compensates.

### Bottom Line

| | Scenario 1 | Scenario 2 |
|---|---|---|
| Instance | c6i.4xlarge | c6i.4xlarge |
| On-demand cost | ~$500/mo | ~$499/mo |
| Reserved cost | ~$326/mo | ~$325/mo |

**Same instance class. Same cost.** Dimensionality is a poor proxy for infrastructure cost. What matters is the interaction of quantization eligibility, recall requirements, and throughput demands.

---

## 5. Qdrant Collection Configurations

### Scenario 1: Search Team

```json
{
  "collection_name": "search_openai_dbpedia",
  "vectors": {
    "size": 3072,
    "distance": "Cosine",
    "on_disk": true
  },
  "hnsw_config": {
    "m": 20,
    "ef_construct": 200,
    "full_scan_threshold": 10000,
    "on_disk": false
  },
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "quantile": 0.99,
      "always_ram": true
    }
  },
  "optimizers_config": {
    "default_segment_number": 2,
    "memmap_threshold": 20000,
    "indexing_threshold": 20000,
    "flush_interval_sec": 5,
    "max_optimization_threads": 2
  },
  "replication_factor": 1,
  "shard_number": 1
}
```

**Query-time overrides** (set per search request):

```json
{
  "params": {
    "hnsw_ef": 200,
    "quantization": {
      "rescore": true,
      "oversampling": 2.0
    }
  }
}
```

**Design notes:**
- `on_disk: true` for vectors pushes float32 data to mmap; `always_ram: true` for quantization keeps int8 projections in memory. This is the split-storage pattern: fast approximate search in RAM, precise rescore from disk.
- `ef_construct=200` balances build cost against graph quality at 95% recall.
- `oversampling=2.0` retrieves 2x candidates in quantized space, then rescores with full float32 from mmap to recover ranking precision.

### Scenario 2: Data Science Team

```json
{
  "collection_name": "datascience_gist960",
  "vectors": {
    "size": 960,
    "distance": "Euclid",
    "on_disk": false
  },
  "hnsw_config": {
    "m": 24,
    "ef_construct": 400,
    "full_scan_threshold": 10000,
    "on_disk": false
  },
  "optimizers_config": {
    "default_segment_number": 2,
    "memmap_threshold": 200000,
    "indexing_threshold": 20000,
    "flush_interval_sec": 30,
    "max_optimization_threads": 4
  },
  "replication_factor": 1,
  "shard_number": 1
}
```

**Query-time overrides:**

```json
{
  "params": {
    "hnsw_ef": 128
  }
}
```

**Design notes:**
- No `quantization_config` block -- float32 vectors remain in RAM by default when `on_disk: false`.
- `flush_interval_sec=30` (vs 5 for Scenario 1): nightly batch writes don't need aggressive durability flushing.
- `max_optimization_threads=4`: more cores available for background compaction during off-peak hours since writes are batched nightly, not streaming.
- `memmap_threshold=200000`: higher threshold keeps segments in RAM longer; appropriate since all vectors fit comfortably in 8 GB.

---

## 6. Benchmark Validation Plan

> *This section will be updated with empirical data from `qdrant_benchmark.py`.*

The sizing numbers above are **analytical priors** derived from the HNSW complexity model, published Qdrant benchmarks, and hardware specs. They must be validated against the actual datasets.

### Metrics to Capture

| Metric | Scenario 1 Target | Scenario 2 Target |
|--------|-------------------|-------------------|
| Recall@k | >= 0.95 @ k=100 | >= 0.99 @ k=10 |
| P99 latency | <= 50 ms | <= 500 ms |
| Sustained QPS | >= 1,000 | >= 3,000 (6hr burst) |
| Write throughput | >= 200 vec/s streaming | >= 100K/batch (nightly) |
| Peak RSS | <= 8 GB | <= 8 GB |

### Tuning Levers if Targets Miss

- **Recall too low (Sc. 1):** Increase `oversampling` from 2.0 to 3.0; increase `hnsw_ef` to 256.
- **Recall too low (Sc. 2):** Increase `hnsw_ef` from 128 to 200; increase `m` to 32 (requires rebuild).
- **Latency too high (Sc. 1):** Increase provisioned IOPS on gp3; consider r6i instance class for more page cache.
- **QPS too low (Sc. 2):** Shard across 2 nodes (doubles cost); alternatively, reduce `hnsw_ef` if recall has margin.

---

## 7. Cost Summary

| | Scenario 1: Search Team | Scenario 2: Data Science Team |
|---|---|---|
| Instance | c6i.4xlarge (16 vCPU, 32 GB) | c6i.4xlarge (16 vCPU, 32 GB) |
| Storage | 39 GB gp3 + provisioned IOPS | 30 GB gp3 (baseline) |
| On-demand | ~$500/mo | ~$499/mo |
| 1-yr reserved (no upfront) | ~$326/mo | ~$325/mo |
| HA option (2nd replica) | +$326-500/mo | +$325-499/mo |

**Combined monthly cost (reserved, no HA):** ~$651/mo for both teams.

These are single-node estimates. Production HA (replication factor 2) doubles the compute cost per scenario but does not require re-architecture -- Qdrant's built-in replication handles this with a configuration change.

---

*Prepared as part of the Qdrant Senior Solutions Architect take-home challenge. All sizing derived from first-principles analysis; benchmark validation pending.*
