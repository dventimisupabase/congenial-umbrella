# Qdrant Vector Database Architecture Field Guide

## How to Use This Guide

Start at **Stage 1** and work through each stage sequentially. Each stage narrows your
design space. By the end, you'll have a concrete starting architecture you can benchmark
and refine.

The guide assumes you already know:
- Your dataset size (number of vectors)
- Your embedding dimensions and origin
- Your SLA requirements (QPS, latency, recall)
- Your read/write patterns

---

## Stage 1: Classify Your Embeddings

This determines your quantization options — the single biggest cost lever.

```
What produced your embeddings?
│
├─ Neural text model (OpenAI, Cohere, BGE, E5, etc.)
│  Dimensions typically: 768 - 4096
│  Properties: centered near zero, well-distributed, high intrinsic dimensionality
│  │
│  └─ Are dimensions >= 1024?
│     ├─ YES ──► EMBEDDING CLASS A: "Binary-friendly neural"
│     │          All quantization strategies available.
│     │          Binary quantization is viable and dramatically reduces cost.
│     │
│     └─ NO ───► EMBEDDING CLASS B: "Scalar-friendly neural"
│                Binary may lose too much signal.
│                Scalar quantization is your best compression tool.
│
├─ Neural vision model (CLIP, DINOv2, etc.)
│  Dimensions typically: 512 - 1024
│  Properties: often well-distributed but may have non-negative activations
│  │
│  └──────────► EMBEDDING CLASS B: "Scalar-friendly neural"
│               Test binary empirically; don't assume it works.
│
├─ Classical CV features (GIST, SIFT, HOG, etc.)
│  Dimensions typically: 128 - 960
│  Properties: non-negative, magnitude-heavy, not centered at zero
│  │
│  └──────────► EMBEDDING CLASS C: "Quantization-resistant"
│               Scalar quantization with caution. Binary likely fails.
│               May need full float32 in RAM for high recall.
│
└─ Other / Unknown
   │
   └──────────► EMBEDDING CLASS B (default assumption)
                Benchmark scalar quantization; fall back to no quantization
                if recall targets can't be met.
```

---

## Stage 2: Determine Your Recall Regime

This determines how hard the algorithm has to work and how much precision you need
in your distance calculations.

```
What is your recall target?
│
├─ 90-95% ────► RECALL REGIME: RELAXED
│               Aggressive quantization is viable.
│               Lower HNSW parameters acceptable.
│               Oversampling + rescoring can cover the gap.
│
├─ 95-98% ────► RECALL REGIME: MODERATE
│               Quantization possible but needs careful tuning.
│               Moderate HNSW parameters.
│               Oversampling required if quantizing.
│
└─ 98-100% ───► RECALL REGIME: STRICT
                Quantization may be infeasible (depends on embedding class).
                High HNSW parameters (m, ef_construct, ef).
                May need full-precision vectors in RAM.
```

### Quantization Feasibility Matrix

Cross-reference your Embedding Class (Stage 1) with your Recall Regime (Stage 2):

```
                    RELAXED (90-95%)    MODERATE (95-98%)    STRICT (98-100%)
                    ─────────────────   ─────────────────    ─────────────────
CLASS A             Binary              Binary + high        Scalar, or
(binary-friendly)   oversample: 1.5-2x  oversample: 3-5x    no quantization
                    rescore: ON         rescore: ON          rescore: ON
                    ─────────────────   ─────────────────    ─────────────────
CLASS B             Scalar              Scalar +             No quantization
(scalar-friendly)   oversample: 1.5x    oversample: 2-3x    (full float32 in RAM)
                    rescore: ON         rescore: ON
                    ─────────────────   ─────────────────    ─────────────────
CLASS C             Scalar (test it)    No quantization      No quantization
(quant-resistant)   rescore: ON         (full float32)       (full float32)
```

**Record your quantization strategy. You'll need it for memory and compute math later.**

---

## Stage 3: Set HNSW Parameters

You need these values before you can size memory or compute, so set them now.

### Step 3a: Choose m

```
RECALL REGIME    RECOMMENDED m    RULE
─────────────    ─────────────    ─────────────────────────────────────────
RELAXED          16               Use 16. Go to 12 only if memory-constrained
                                  after completing Stage 4 and needing to cut.
MODERATE         20               Use 20. Midpoint of range.
STRICT           28               Use 28. Midpoint of range (24-32).

Rarely go above 32 — diminishing returns.
```

### Step 3b: Choose ef_construct

```
ef_construct only affects index build time, not query time or RAM.
You pay this cost once. Use the high end of the range.

RECALL REGIME    ef_construct     RULE
─────────────    ────────────     ──────────────────────────────────
RELAXED          128              Fixed value — no range to decide.
MODERATE         256              Use the top of the 200-256 range.
                                  Build time is a one-time cost.
STRICT           384              Use the midpoint of 256-512.
                                  Go to 512 only if benchmarks show
                                  recall is still below target after
                                  tuning ef in Stage 5.
```

### Step 3c: Choose ef (search-time) — PRELIMINARY

Set a starting value now for compute estimation. You will refine this during
benchmarking.

```
ef must be >= top_k (hard constraint)

RECALL REGIME    STARTING ef                 RULE
─────────────    ──────────────────────────  ──────────────────────────
RELAXED          2x top_k                    Use the low end. Sufficient
                                             for 90-95% recall in most cases.
MODERATE         3x top_k                    Midpoint of 2-4x range.
STRICT           6x top_k                    Midpoint of 4-8x range.

Apply floor: ef = max(ef, 64) — very small top_k values (e.g. 10)
produce ef values too low for effective graph traversal.

ef is your primary runtime tuning knob. After benchmarking:
  - If recall is below target: increase ef.
  - If latency is above SLA: decrease ef (and accept lower recall,
    or improve quantization/graph quality instead).
```

### Step 3d: Choose Oversampling (if quantizing)

```
Only applies when using quantization + rescoring.

QUANTIZATION     RECALL REGIME    OVERSAMPLING FACTOR
────────────     ─────────────    ───────────────────
Binary           RELAXED          2.0x  (top of 1.5-2.0 range — binary is lossy)
Binary           MODERATE         4.0x  (midpoint of 3.0-5.0)
Binary           STRICT           Not recommended (use scalar or none)
Scalar           RELAXED          1.5x  (top of 1.0-1.5)
Scalar           MODERATE         2.5x  (midpoint of 2.0-3.0)
Scalar           STRICT           4.0x  (midpoint of 3.0-5.0, may not suffice)

rescore_candidates = oversampling_factor x top_k
These candidates are re-ranked using full-precision vectors.
```

---

## Stage 4: Assess Your Latency Budget

This determines what can live on disk vs. what must be in RAM.

```
What is your P99 latency SLA?
│
├─ < 20ms ────► LATENCY TIER: ULTRA-TIGHT
│               Everything in RAM: HNSW index, quantized vectors, full vectors.
│               No mmap. NVMe only as backup.
│               Consider: is this SLA realistic for your recall target?
│
├─ 20-100ms ──► LATENCY TIER: TIGHT
│               HNSW index + quantized vectors: RAM (mandatory)
│               Full vectors for rescoring: NVMe SSD via mmap (OK)
│               ~100us per disk read is acceptable within budget.
│               Disk IOPS become critical if rescoring many candidates.
│
├─ 100-500ms ─► LATENCY TIER: MODERATE
│               HNSW index: RAM (always)
│               Quantized vectors: RAM preferred, mmap acceptable
│               Full vectors: see decision rule below
│
└─ > 500ms ───► LATENCY TIER: RELAXED
                HNSW index: RAM (always, non-negotiable)
                Everything else: mmap from SSD is fine.
                Budget allows for very thorough graph traversal.
                Good for batch/offline workloads.
```

### Storage Placement Decision Table

Cross-reference Latency Tier with your components. **If your quantization strategy
from Stage 2 is "no quantization," there are no separate quantized vectors — the full
vectors ARE the search vectors, and every distance calculation reads them. In this
case, promote full vectors one tier toward RAM** (marked with * below).

```
Component          ULTRA-TIGHT    TIGHT          MODERATE         RELAXED
──────────────     ───────────    ─────          ────────         ───────
HNSW index         RAM            RAM            RAM              RAM
Quantized vectors  RAM            RAM            RAM or mmap      mmap OK
Full vectors       RAM            mmap (NVMe)    mmap (SSD)       mmap (SSD)
  *if no quant:    RAM            RAM            RAM preferred*   mmap (SSD)
Payload/metadata   RAM            RAM or mmap    mmap             mmap

* = When there is no quantization layer, every search query reads full vectors
    directly. mmap adds per-query I/O latency that compounds with high ef values.
    Promoting to RAM avoids this. If RAM is too expensive, mmap is still viable
    under MODERATE/RELAXED latency tiers — just account for the I/O cost in your
    per-query time estimate (add ~0.5-1ms for mmap access patterns under load).
```

---

## Stage 5: Size Your Memory

### Step 5a: Vector Memory

```
Pick your quantization strategy from Stage 2 and calculate:

NO QUANTIZATION (float32):
  vector_memory = num_vectors x dimensions x 4 bytes

SCALAR QUANTIZATION (int8):
  quantized_memory = num_vectors x dimensions x 1 byte
  full_vector_memory = num_vectors x dimensions x 4 bytes  (for rescore, may be on disk)

BINARY QUANTIZATION (1-bit):
  quantized_memory = num_vectors x dimensions / 8
  full_vector_memory = num_vectors x dimensions x 4 bytes  (for rescore, may be on disk)
```

### Step 5b: HNSW Index Memory

```
Use the m value you chose in Stage 3, Step 3a.

hnsw_memory = num_vectors x m x 2 x 8 bytes

Common values:
  m=16:  num_vectors x 256 bytes
  m=20:  num_vectors x 320 bytes
  m=28:  num_vectors x 448 bytes
  m=32:  num_vectors x 512 bytes
```

### Step 5c: Total RAM

```
Based on your Storage Placement from Stage 4, sum the components assigned to RAM:

total_ram_components = sum of (components placed in RAM from Stage 4 table)
overhead = total_ram_components x 0.20
total_ram = total_ram_components + overhead

The overhead covers:
  - Payload storage (metadata per vector)
  - Internal data structures (segment maps, id maps)
  - OS page cache / filesystem buffers
  - Qdrant process overhead
  - Temporary memory during segment merges
```

---

## Stage 6: Size Your Disk

Disk is sized separately from RAM. Everything persists to disk regardless of whether
it's also in RAM (Qdrant uses disk as the durable backing store).

### Step 6a: Base Disk

```
base_disk = full_vector_memory + quantized_memory + hnsw_on_disk

Where:
  full_vector_memory = num_vectors x dimensions x 4 bytes  (always stored on disk)
  quantized_memory   = size of quantized vectors (0 if no quantization)
  hnsw_on_disk       = hnsw_memory (index is persisted to disk even if loaded into RAM)
```

### Step 6b: WAL (Write-Ahead Log) Space

```
WAL accumulates writes before they're flushed to segments.

STREAMING writes:
  wal_space = write_rate_per_second x avg_vector_bytes x wal_flush_interval_seconds x 2
  Default wal_flush_interval is ~1 second, but segments may not merge immediately.
  Rule of thumb: wal_space = 2 GB (sufficient for most streaming workloads up to 500/s)

BATCH writes:
  wal_space = batch_size x avg_vector_bytes x 2
  The x2 accounts for WAL segments coexisting before compaction.

RARE/STATIC writes:
  wal_space = 1 GB (minimal, just for operational headroom)
```

### Step 6c: Total Disk

```
total_disk = (base_disk + wal_space) x 1.5

The 1.5x multiplier covers:
  - Segment merge operations (old + new segments coexist temporarily)
  - Snapshot storage (if enabled)
  - OS filesystem overhead
```

---

## Stage 7: Size Your Compute (vCPUs)

### Step 7a: Estimate Per-Query CPU Time

This varies with ef, dimensions, and quantization. Use these baselines:

```
                        Binary quant    Scalar quant    No quant (float32)
                        ────────────    ────────────    ──────────────────
dims < 512              ~1ms            ~2ms            ~3ms
dims 512-1024           ~1.5ms          ~3ms            ~5ms
dims 1024-2048          ~2ms            ~4ms            ~8ms
dims 2048-4096          ~2.5ms          ~5ms            ~12ms

These assume:
  - ef = 2x top_k
  - Includes rescoring time if quantized
  - Warm cache (data in RAM or OS page cache)
  - Single-threaded per query

Adjustments for ef (use the ef value from Stage 3, Step 3c):
  - ef = 2x top_k: no adjustment (baseline)
  - ef = 3x top_k: multiply by 1.25
  - ef = 4x top_k: multiply by 1.5
  - ef = 6x top_k: multiply by 2.0
  - ef = 8x top_k: multiply by 2.5

Additional adjustments:
  - If rescoring from disk (mmap): add ~0.5ms per 100 rescore candidates
  - If full vectors on mmap with no quantization: add ~1ms (random I/O during search)
```

### Step 7b: Calculate Required Cores for QPS

```
cores_for_queries = target_QPS x per_query_cpu_seconds

Then add headroom:
  cores_total = cores_for_queries x 1.3  (30% headroom for GC, OS, background tasks)
```

### Step 7c: Account for Write Load

```
If concurrent reads + writes (STREAMING pattern):
  write_cores = write_QPS x indexing_time_per_vector

  Indexing time per vector (rough):
    - Simple append: ~0.1ms
    - HNSW insertion with m=16: ~1-2ms
    - HNSW insertion with m=20: ~2-3ms
    - HNSW insertion with m=28: ~3-4ms
    - HNSW insertion with m=32: ~3-5ms

  cores_total = (cores_for_queries + write_cores) x 1.3

If batch writes (no concurrent reads):
  Size for peak read QPS only.
  Writes use the same cores during off-peak.
```

### Step 7d: Single Node or Multiple?

```
Does cores_total fit in a single machine AND does total_ram fit?
│
├─ YES to both ──► Single node is viable.
│  │
│  └─ Is the QPS bursty (e.g., peak window < 24h)?
│     ├─ YES ──► Single node is still the default recommendation.
│     │          Note: replicas could allow scaling down off-peak,
│     │          but this is a cost optimization to explore AFTER
│     │          validating the architecture works. Do not introduce
│     │          replicas at initial design stage unless the single
│     │          node option is clearly wasteful (>50% idle time at
│     │          sustained cost). Flag it as a future optimization.
│     │
│     └─ NO ───► Single node. Proceed.
│
└─ NO to either ──► You need multiple nodes.
   │
   ├─ RAM is the bottleneck (data doesn't fit in one node)?
   │  └─► Shards: partition data across nodes
   │      num_shards = ceil(total_ram / max_ram_per_node)
   │      Note: each query hits all shards, adding latency.
   │
   └─ CPU is the bottleneck (QPS too high for one node)?
      └─► Replicas: full copies of data, distribute QPS
          num_replicas = ceil(cores_total / max_cores_per_node)
          Each replica needs total_ram, so total cost = replicas x node_cost.
```

---

## Stage 8: Tune for Write Pattern

```
What is your write pattern?
│
├─ STREAMING (continuous writes, concurrent with reads)
│  │
│  ├─ Write rate < 500/s ──► Standard configuration
│  │  - Default segment sizes are fine
│  │  - Qdrant handles background merging automatically
│  │  - Ensure 20-30% CPU headroom above read requirements
│  │
│  └─ Write rate > 500/s ──► Optimize for write throughput
│     - Increase WAL capacity
│     - Consider larger memmap_threshold to batch more in memory
│     - Monitor segment count (too many small segments hurts read perf)
│     - May need dedicated write cores (add to Stage 7 calculation)
│
├─ BATCH (periodic bulk loads, no concurrent reads)
│  │
│  ├─ Can you take the collection offline during load?
│  │  ├─ YES ──► Optimal: disable indexing during load, rebuild after
│  │  │          - Set indexing_threshold very high (or disable)
│  │  │          - Bulk insert all vectors
│  │  │          - Trigger index rebuild
│  │  │          - Much faster than incremental indexing
│  │  │
│  │  └─ NO ───► Load with indexing enabled but size for write capacity
│  │
│  └─ Size disk for: current data + full batch + merge overhead (2-3x batch size)
│
└─ RARE / STATIC (load once, mostly read)
   │
   └──────────► Optimize entirely for reads
                - Compact segments after initial load
                - Consider on_disk: false for everything that fits in RAM
                - No write headroom needed in CPU calculation
```

---

## Stage 9: Build Your Architecture Summary

Fill in this template with your decisions from Stages 1-8:

```
╔══════════════════════════════════════════════════════════╗
║  ARCHITECTURE SUMMARY                                    ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Dataset: _______________________________________________║
║  Vectors: ____________  Dimensions: ____________________║
║  Embedding Class: ______________________________________║
║                                                          ║
║  SLA Requirements                                        ║
║  ─────────────────                                       ║
║  Target QPS:     ____________                            ║
║  P99 Latency:    ____________                            ║
║  Recall Target:  ____________                            ║
║  Top-k:          ____________                            ║
║                                                          ║
║  Regime Classification                                   ║
║  ─────────────────────                                   ║
║  Recall Regime:  ____________                            ║
║  Latency Tier:   ____________                            ║
║  Write Pattern:  ____________                            ║
║                                                          ║
║  Quantization Strategy (from Stage 2)                    ║
║  ─────────────────────                                   ║
║  Method:         ____________                            ║
║  Oversampling:   ____________                            ║
║  Rescore:        ____________                            ║
║                                                          ║
║  HNSW Parameters (from Stage 3)                          ║
║  ───────────────                                         ║
║  m:              ____________                            ║
║  ef_construct:   ____________                            ║
║  ef (search):    ____________  (preliminary — benchmark) ║
║                                                          ║
║  Memory Calculation (from Stage 5)                       ║
║  ──────────────────                                      ║
║  Quantized vectors:  ________ (location: RAM / disk)     ║
║  Full vectors:       ________ (location: RAM / disk)     ║
║  HNSW index:         ________ (location: RAM)            ║
║  Overhead (20%):     ________                            ║
║  ────────────────────────────                            ║
║  Total RAM needed:   ________                            ║
║                                                          ║
║  Disk Calculation (from Stage 6)                         ║
║  ────────────────                                        ║
║  Base disk:          ________                            ║
║  WAL space:          ________                            ║
║  Total (with 1.5x):  ________                           ║
║                                                          ║
║  Compute Calculation (from Stage 7)                      ║
║  ───────────────────                                     ║
║  Per-query CPU time: ________                            ║
║  Cores for QPS:      ________                            ║
║  Cores for writes:   ________                            ║
║  Headroom (30%):     ________                            ║
║  ────────────────────────────                            ║
║  Total vCPUs needed: ________                            ║
║                                                          ║
║  Topology (from Stage 7d)                                ║
║  ────────                                                ║
║  Nodes:          ____________                            ║
║  Shards:         ____________                            ║
║  Replicas:       ____________                            ║
║                                                          ║
║  Estimated Instance: ____________________________________║
║  Estimated Cost:     ____________________________________║
║                                                          ║
║  Future Optimizations to Explore                         ║
║  ───────────────────────────────                         ║
║  ________________________________________________________║
║  ________________________________________________________║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

---

## Quick Reference: Common Archetypes

For rapid pattern-matching, here are common scenarios you'll encounter:

### Archetype 1: "High-dim text search, good-enough recall"
- OpenAI / Cohere / BGE embeddings, 1536-4096 dims
- 95% recall, <100ms latency, moderate QPS
- **Recipe**: Binary quantization, m=16, ef=2x top_k, mmap full vectors on NVMe
- **Cost profile**: Very low RAM (quantized vectors are tiny), moderate disk

### Archetype 2: "High-dim text search, strict recall"
- Same embeddings as above
- 99% recall, <100ms latency
- **Recipe**: Scalar quantization (binary too lossy at 99%), m=28, ef=6x top_k, oversample 4x
- **Cost profile**: Moderate RAM, moderate disk

### Archetype 3: "Low-dim features, strict recall"
- CV features, scientific embeddings, 128-960 dims
- 99% recall
- **Recipe**: No quantization (full float32 in RAM), m=28, ef=6x top_k
- **Cost profile**: High RAM per vector (no compression), CPU-heavy if high QPS

### Archetype 4: "Massive scale, relaxed latency"
- Any embeddings, >10M vectors
- 95% recall, >200ms latency OK
- **Recipe**: Scalar or binary quantization, mmap everything except HNSW index, multiple shards
- **Cost profile**: Disk-heavy, RAM mostly for HNSW index, horizontally scaled

### Archetype 5: "Real-time recommendations"
- Medium-dim embeddings (256-768), moderate recall (90-95%)
- <20ms latency, high QPS (5000+)
- **Recipe**: Everything in RAM, scalar quantization, m=16, low ef, multiple replicas for QPS
- **Cost profile**: RAM-heavy but compressed, many CPU cores, replicas for throughput

---

## Decision Traps to Avoid

1. **"More dimensions = more resources"**
   Wrong. Quantization effectiveness and recall requirements dominate. 3072-dim with
   binary quantization can be cheaper than 960-dim without quantization.

2. **"Just throw more RAM at it"**
   RAM solves latency, not recall. If recall is below target, you need better HNSW
   parameters or less aggressive quantization — not more memory.

3. **"Set ef_construct high and ef low"**
   A great graph (high ef_construct) can't compensate for not searching it thoroughly
   (low ef). You need both, proportional to your recall target.

4. **"Sharding helps with QPS"**
   Sharding helps with data size, not throughput. Each query must touch all shards.
   For QPS scaling, use replicas.

5. **"Quantize everything the same way"**
   Different datasets have different quantization tolerance. Always benchmark
   quantization against your specific embeddings and recall target.

6. **"P99 = average x 2"**
   Tail latency in vector search is driven by unlucky graph traversals and segment
   merges, not by simple distribution. Measure P99 under load, don't estimate it.

7. **"Batch size doesn't matter"**
   Large batch inserts can trigger segment merges that temporarily spike query latency.
   If you have concurrent reads + writes, monitor merge impact on tail latency.
