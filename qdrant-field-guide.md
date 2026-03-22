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

**Record your quantization strategy. You'll need it for memory math in Stage 4.**

---

## Stage 3: Assess Your Latency Budget

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
│               Full vectors: disk (mmap) is fine
│               More room for higher ef values and larger oversample.
│
└─ > 500ms ───► LATENCY TIER: RELAXED
                HNSW index: RAM (always, non-negotiable)
                Everything else: mmap from SSD is fine.
                Budget allows for very thorough graph traversal.
                Good for batch/offline workloads.
```

### Storage Placement Decision Table

Cross-reference Latency Tier with your components:

```
Component          ULTRA-TIGHT    TIGHT          MODERATE       RELAXED
──────────────     ───────────    ─────          ────────       ───────
HNSW index         RAM            RAM            RAM            RAM
Quantized vectors  RAM            RAM            RAM or mmap    mmap OK
Full vectors       RAM            mmap (NVMe)    mmap (SSD)     mmap (SSD)
Payload/metadata   RAM            RAM or mmap    mmap           mmap
```

---

## Stage 4: Size Your Memory

Now you have everything needed to calculate RAM requirements.

### Step 4a: Vector Memory

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

### Step 4b: HNSW Index Memory

```
hnsw_memory = num_vectors x m x 2 x 8 bytes

Common values:
  m=16:  num_vectors x 256 bytes
  m=24:  num_vectors x 384 bytes
  m=32:  num_vectors x 512 bytes
```

### Step 4c: Overhead

```
overhead = (vector_memory + hnsw_memory) x 0.20

This covers:
  - Payload storage (metadata per vector)
  - Internal data structures (segment maps, id maps)
  - OS page cache / filesystem buffers
  - Qdrant process overhead
  - Temporary memory during segment merges
```

### Step 4d: Total RAM

```
Based on your Storage Placement from Stage 3:

total_ram = sum of components assigned to RAM + overhead

If TIGHT latency:
  total_ram = hnsw_memory + quantized_memory + overhead
  (full vectors on disk, need fast SSD)

If ULTRA-TIGHT:
  total_ram = hnsw_memory + quantized_memory + full_vector_memory + overhead

If MODERATE/RELAXED:
  total_ram = hnsw_memory + overhead  (minimum)
  (add quantized_memory to RAM for better performance)
```

### Step 4e: Disk Sizing

```
disk_needed = full_vector_memory + quantized_memory + hnsw_memory + wal_space

wal_space = estimated_daily_writes x avg_vector_size x 2
  (WAL segments before compaction; factor of 2 for safety)

Add 30-50% headroom for:
  - Segment merge operations (temporarily doubles space)
  - WAL accumulation during peak writes
  - Snapshot storage
```

---

## Stage 5: Size Your Compute (vCPUs)

### Step 5a: Estimate Per-Query CPU Time

This varies with ef, dimensions, and quantization. Use these rough baselines:

```
                        Binary quant    Scalar quant    No quant (float32)
                        ────────────    ────────────    ──────────────────
dims < 512              ~1ms            ~2ms            ~3ms
dims 512-1024           ~1.5ms          ~3ms            ~5ms
dims 1024-2048          ~2ms            ~4ms            ~8ms
dims 2048-4096          ~2.5ms          ~5ms            ~12ms

These assume:
  - ef = 2x top_k (adjust up for higher recall)
  - Includes rescoring time if quantized
  - Warm cache (data in RAM or OS page cache)
  - Single-threaded per query

Adjustments:
  - For ef = 4x top_k: multiply by 1.5
  - For ef = 8x top_k: multiply by 2.5
  - Add ~0.5ms per 100 rescore candidates if rescoring from disk
```

### Step 5b: Calculate Required Cores for QPS

```
cores_for_queries = target_QPS x per_query_cpu_seconds

Example: 3000 QPS x 0.005s = 15 cores

Then add headroom:
  cores_total = cores_for_queries x 1.3  (30% headroom for GC, OS, background tasks)
```

### Step 5c: Account for Write Load

```
If concurrent reads + writes:
  write_cores = write_QPS x indexing_time_per_vector

  Indexing time per vector (rough):
    - Simple append: ~0.1ms
    - HNSW insertion with m=16: ~1-2ms
    - HNSW insertion with m=32: ~3-5ms

  cores_total = (cores_for_queries + write_cores) x 1.3

If batch writes (no concurrent reads):
  Size for peak read QPS only.
  Writes use the same cores during off-peak.
```

### Step 5d: Single Node or Multiple?

```
Is cores_total <= 64?
├─ YES ──► Single node likely sufficient
│          Verify total_ram fits in one machine
│
└─ NO ───► Consider:
           ├─ Replicas: copies of the same data, distribute read QPS
           │  Good when: high QPS is the bottleneck, data fits in one node's RAM
           │  num_replicas = ceil(cores_total / cores_per_node)
           │
           └─ Shards: partition data across nodes
              Good when: data doesn't fit in one node's RAM
              num_shards = ceil(total_ram / ram_per_node)
              Note: each query must hit all shards, adding latency
```

---

## Stage 6: Set HNSW Parameters

### Step 6a: Choose m

```
RECALL REGIME    RECOMMENDED m    MEMORY IMPACT (per 1M vectors)
─────────────    ─────────────    ─────────────────────────────
RELAXED          12-16            192-256 MB
MODERATE         16-24            256-384 MB
STRICT           24-32            384-512 MB

Higher m = better recall, more memory, slower indexing.
Rarely go above 32 — diminishing returns.
```

### Step 6b: Choose ef_construct

```
Rule of thumb: ef_construct = max(2 x m, 128)

RECALL REGIME    RECOMMENDED ef_construct
─────────────    ────────────────────────
RELAXED          128
MODERATE         200-256
STRICT           256-512

Higher ef_construct = better graph quality = better recall at any ef.
Only affects index build time, not query time or memory.
Err on the high side — you pay this cost once.
```

### Step 6c: Choose ef (search-time)

```
ef must be >= top_k (hard constraint)

RECALL REGIME    RECOMMENDED ef (as multiple of top_k)
─────────────    ──────────────────────────────────────
RELAXED          1.5 - 2x top_k
MODERATE         2 - 4x top_k
STRICT           4 - 8x top_k

Higher ef = better recall, slower queries.
This is your primary runtime tuning knob.
Start at 2x top_k and increase until recall target is met.
```

### Step 6d: Choose Oversampling (if quantizing)

```
Only applies when using quantization + rescoring.

QUANTIZATION     RECALL REGIME    OVERSAMPLING FACTOR
────────────     ─────────────    ───────────────────
Binary           RELAXED          1.5 - 2.0x
Binary           MODERATE         3.0 - 5.0x
Binary           STRICT           Not recommended (use scalar or none)
Scalar           RELAXED          1.0 - 1.5x
Scalar           MODERATE         2.0 - 3.0x
Scalar           STRICT           3.0 - 5.0x (may not be enough)

rescore_candidates = oversampling_factor x top_k
These candidates are re-ranked using full-precision vectors.
```

---

## Stage 7: Tune for Write Pattern

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
│     - May need dedicated write cores (add to Stage 5 calculation)
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

## Stage 8: Build Your Architecture Summary

Fill in this template with your decisions from Stages 1-7:

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
║  Quantization Strategy                                   ║
║  ─────────────────────                                   ║
║  Method:         ____________                            ║
║  Oversampling:   ____________                            ║
║  Rescore:        ____________                            ║
║                                                          ║
║  HNSW Parameters                                         ║
║  ───────────────                                         ║
║  m:              ____________                            ║
║  ef_construct:   ____________                            ║
║  ef (search):    ____________                            ║
║                                                          ║
║  Memory Calculation                                      ║
║  ──────────────────                                      ║
║  Quantized vectors:  ________ (location: RAM / disk)     ║
║  Full vectors:       ________ (location: RAM / disk)     ║
║  HNSW index:         ________ (location: RAM)            ║
║  Overhead (20%):     ________                            ║
║  ────────────────────────────                            ║
║  Total RAM needed:   ________                            ║
║  Total disk needed:  ________                            ║
║                                                          ║
║  Compute Calculation                                     ║
║  ───────────────────                                     ║
║  Per-query CPU time: ________                            ║
║  Cores for QPS:      ________                            ║
║  Cores for writes:   ________                            ║
║  Headroom (30%):     ________                            ║
║  ────────────────────────────                            ║
║  Total vCPUs needed: ________                            ║
║                                                          ║
║  Topology                                                ║
║  ────────                                                ║
║  Nodes:          ____________                            ║
║  Shards:         ____________                            ║
║  Replicas:       ____________                            ║
║                                                          ║
║  Estimated Instance: ____________________________________║
║  Estimated Cost:     ____________________________________║
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
- **Recipe**: Scalar quantization (binary too lossy at 99%), m=24-32, ef=4-8x top_k, oversample 3-5x
- **Cost profile**: Moderate RAM, moderate disk

### Archetype 3: "Low-dim features, strict recall"
- CV features, scientific embeddings, 128-960 dims
- 99% recall
- **Recipe**: No quantization (full float32 in RAM), m=32, ef=4-8x top_k
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
