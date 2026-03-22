# Qdrant Vector Database Architecture Field Guide

## How to Use This Guide

Start at **Stage 1** and work through each stage sequentially. Each stage narrows your
design space. By the end, you'll have a concrete starting architecture you can benchmark
and refine.

### Design Philosophy

This guide produces a **conservative initial estimate** — a defensible starting point
for a customer conversation, not a final production configuration. The principles:

1. **Conservative over optimal.** Where expert opinion splits (e.g., quantization
   strategy, HNSW parameter tuning), this guide chooses the safer option. The result
   may over-provision slightly, but it will not under-deliver on SLAs. Aggressive
   optimizations (binary quantization, lower m, reduced ef) are flagged as future
   refinements to explore after benchmarking validates the baseline.

2. **Minimum viable infrastructure.** Topology decisions (replication, sharding)
   are customer decisions driven by business requirements (HA, compliance, cost
   tolerance), not sizing decisions. This guide produces a single-node architecture
   that meets the SLA, then presents multi-node options as recommendations.

3. **Benchmark, then optimize.** The output of this guide is a starting architecture
   to load-test. Real-world performance depends on data distribution, access patterns,
   and hardware specifics that no formula can fully capture. Expect to tune hnsw_ef,
   oversampling, and instance size after benchmarking. The guide's values are chosen
   to be good enough to pass initial benchmarks, not to be globally optimal.

### Prerequisites

The guide assumes you already know:
- Your dataset size (number of vectors)
- Your embedding dimensions and origin
- Your SLA requirements (QPS, latency, recall)
- Your read/write patterns

---

## Stage 1: Classify Your Embeddings

This determines your quantization options (the single biggest cost lever) and your
distance metric.

```mermaid
flowchart TD
    Start["What produced your embeddings?"] --> NT["Neural text model\n(OpenAI, Cohere, BGE, E5)"]
    Start --> NV["Neural vision model\n(CLIP, DINOv2)"]
    Start --> CV["Classical CV features\n(GIST, SIFT, HOG)"]
    Start --> Other["Other / Unknown"]

    NT --> DimCheck{"Dimensions >= 1024?"}
    DimCheck -->|Yes| ClassA["CLASS A\nBinary-friendly neural"]
    DimCheck -->|No| ClassB1["CLASS B\nScalar-friendly neural"]

    NV --> ClassB2["CLASS B\nScalar-friendly neural"]
    CV --> ClassC["CLASS C\nQuantization-resistant"]
    Other --> ClassB3["CLASS B\n(default assumption)"]

    style ClassA fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style ClassB1 fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style ClassB2 fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style ClassB3 fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style ClassC fill:#3b1f1f,stroke:#ef4444,color:#fca5a5
    style Start fill:#18181b,stroke:#3f3f46,color:#e4e4e7
```

**Embedding class details:**

| Class | Type | Distance Metric | Qdrant `distance` | Properties |
|-------|------|----------------|-------------------|------------|
| **A** | Neural text, dims >= 1024 | Cosine | `"Cosine"` | Centered near zero, well-distributed. All quantization strategies viable. Binary quantization dramatically reduces cost. _Some models (e.g., OpenAI text-embedding-3-large) support Matryoshka dimension reduction — test truncation against recall target._ |
| **B** | Neural text (dims < 1024), neural vision, or unknown | Cosine | `"Cosine"` | Binary may lose signal. Scalar quantization is the best compression tool. For vision models, verify distance metric with model docs. Alternative: use `"Dot"` if model was trained with dot-product objective. |
| **C** | Classical CV features (GIST, SIFT, HOG) | Euclidean (L2) | `"Euclid"` | Non-negative, magnitude-heavy, not centered at zero. Quantization-resistant. May need full float32 in RAM for high recall. Using Cosine on non-normalized features gives wrong results. Qdrant also supports `"Manhattan"` for L1. |

**Record: your Embedding Class and your Distance Metric (Qdrant `distance` value).**

---

## Stage 2: Determine Your Recall Regime

This determines how hard the algorithm has to work and how much precision you need
in your distance calculations.

```mermaid
flowchart LR
    Start["Recall target?"] --> R["90-95%\nRELAXED"]
    Start --> M["96-98%\nMODERATE"]
    Start --> S["99-100%\nSTRICT"]

    R --> RD["Aggressive quantization viable\nLower HNSW parameters OK\nOversampling + rescoring covers the gap"]
    M --> MD["Quantization needs careful tuning\nModerate HNSW parameters\nOversampling required if quantizing"]
    S --> SD["Quantization may be infeasible\nHigh HNSW parameters\nMay need full float32 in RAM"]

    style R fill:#14532d,stroke:#22c55e,color:#86efac
    style M fill:#422006,stroke:#f59e0b,color:#fde68a
    style S fill:#3b1f1f,stroke:#ef4444,color:#fca5a5
    style Start fill:#18181b,stroke:#3f3f46,color:#e4e4e7
```

**Boundary rule:** 95% is RELAXED. 96% is MODERATE. 99% is STRICT.

### Quantization Feasibility Matrix

Cross-reference your Embedding Class (Stage 1) with your Recall Regime (Stage 2).
This table tells you WHICH quantization method to use. The specific oversampling
factor is set later in Stage 3d — do not pick an oversampling value from this table.

| Embedding Class | RELAXED (90-95%) | MODERATE (96-98%) | STRICT (99-100%) |
|-----------------|------------------|-------------------|------------------|
| **CLASS A** (binary-friendly) | Scalar, rescore: ON | Scalar, rescore: ON | Scalar, or no quantization; rescore: ON (if scalar) |
| **CLASS B** (scalar-friendly) | Scalar, rescore: ON | Scalar, rescore: ON | No quantization (full float32 in RAM) |
| **CLASS C** (quant-resistant) | Scalar (test it), rescore: ON | No quantization (full float32) | No quantization (full float32) |

> **NOTE on Binary Quantization:** For CLASS A embeddings, binary quantization
> offers 32x compression (vs scalar's 4x) and can be effective for high-dim
> neural embeddings. However, expert opinion is split on whether binary is
> reliable enough for production use at top-k >= 50. This guide defaults to
> scalar quantization as the conservative choice. Consider binary quantization
> as a FUTURE OPTIMIZATION after validating scalar meets your requirements —
> especially when RAM savings are critical and top-k is small (< 20).

**Qdrant quantization_config examples:**

```json
{"scalar": {"type": "int8", "quantile": 0.99, "always_ram": true}}
```

```json
{"binary": {"always_ram": true}}
```

```json
{"product": {"compression": "x32", "always_ram": true}}
```

None: omit `quantization_config` entirely.

Qdrant also supports Product Quantization (up to 64x compression), but it has
higher indexing cost and lower recall than scalar. Use only when memory is the
absolute top priority and recall loss is acceptable.

**Record: your quantization method (Binary, Scalar, or None) and whether rescore is ON.**

---

## Stage 3: Set HNSW Parameters

You need these values before you can size memory or compute, so set them now.

**Qdrant parameter mapping:**
- `m` and `ef_construct` are set in `hnsw_config` at collection creation
- Search-time ef is called **`hnsw_ef`** in Qdrant (set per-query in `search_params`)
- Qdrant defaults: m=16, ef_construct=100, full_scan_threshold=10000

### Step 3a: Choose m

**Qdrant parameter:** `hnsw_config.m` (set at collection creation) — **Default:** 16

| RECALL REGIME | m | RULE |
|---------------|---|------|
| RELAXED | 16 | Fixed. Go to 12 only if memory-constrained after completing Stage 5 and needing to cut. |
| MODERATE | 20 | Fixed. |
| STRICT | 32 | Fixed. |

```mermaid
flowchart TD
    Start{"top_k value?"} -->|">= 50"| High["Increase m by one tier\n(denser graph needed for large result sets)"]
    Start -->|"21 - 49"| Mid["Keep m as-is"]
    Start -->|"<= 20"| Low["Decrease m by one tier\n(minimum m = 16)"]

    High --> H_R["RELAXED: 16 -> 20"]
    High --> H_M["MODERATE: 20 -> 28"]
    High --> H_S["STRICT: stays at 32"]

    Low --> L_S["STRICT: 32 -> 24"]
    Low --> L_M["MODERATE: 20 -> 16"]
    Low --> L_R["RELAXED: stays at 16"]

    Mid --> Note["Rarely go above 32\ndiminishing returns"]

    style Start fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style High fill:#422006,stroke:#f59e0b,color:#fde68a
    style Mid fill:#14532d,stroke:#22c55e,color:#86efac
    style Low fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style H_R fill:#27272a,stroke:#3f3f46,color:#a1a1aa
    style H_M fill:#27272a,stroke:#3f3f46,color:#a1a1aa
    style H_S fill:#27272a,stroke:#3f3f46,color:#a1a1aa
    style L_S fill:#27272a,stroke:#3f3f46,color:#a1a1aa
    style L_M fill:#27272a,stroke:#3f3f46,color:#a1a1aa
    style L_R fill:#27272a,stroke:#3f3f46,color:#a1a1aa
    style Note fill:#27272a,stroke:#3f3f46,color:#a1a1aa
```

### Step 3b: Choose ef_construct

**Qdrant parameter:** `hnsw_config.ef_construct` (set at collection creation) — **Default:** 100

ef_construct only affects index build time, not query time or RAM.
You pay this cost once.

| RECALL REGIME | ef_construct | RULE |
|---------------|-------------|------|
| RELAXED | 200 | Fixed. (Above Qdrant's default of 100.) |
| MODERATE | 256 | Fixed. |
| STRICT | 400 | Fixed. Go to 512 only if benchmarks show recall is still below target after tuning ef. |

```mermaid
flowchart TD
    Q{"Write pattern is BATCH?\n(no concurrent reads during writes)"} -->|"YES"| Use400["Use ef_construct = 400\nfor STRICT regardless"]
    Q -->|"NO"| Keep["Keep ef_construct\nfrom table above"]

    Use400 --> Note["Extra build time costs nothing\nwhen there are no queries to slow down\n(400 sufficient for 99% recall;\n512 available as benchmark-driven escalation)"]

    style Q fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style Use400 fill:#14532d,stroke:#22c55e,color:#86efac
    style Keep fill:#27272a,stroke:#3f3f46,color:#a1a1aa
    style Note fill:#27272a,stroke:#3f3f46,color:#a1a1aa
```

### Step 3c: Choose hnsw_ef (search-time) — PRELIMINARY

Set a starting value now for compute estimation. You will refine this during
benchmarking.

**Qdrant parameter:** `search_params.hnsw_ef` (set per-query, not at collection level)
Example: `client.query_points(..., search_params={"hnsw_ef": 200})`

hnsw_ef must be >= top_k (hard constraint)

| RECALL REGIME | STARTING hnsw_ef | RULE |
|---------------|-----------------|------|
| RELAXED | 2x top_k | Fixed multiplier. |
| MODERATE | 3x top_k | Fixed multiplier. |
| STRICT | 6x top_k | Fixed multiplier. |

```mermaid
flowchart TD
    Calc["calculated_ef = multiplier x top_k"] --> Floor{"Apply floor by regime"}

    Floor -->|"RELAXED"| FR["hnsw_ef = max(calculated_ef, 64)"]
    Floor -->|"MODERATE"| FM["hnsw_ef = max(calculated_ef, 64)"]
    Floor -->|"STRICT"| FS["hnsw_ef = max(calculated_ef, 128)"]

    FR --> Use["Use floor-adjusted ef\nfor all subsequent stages\nincluding compute estimation (Stage 7a)"]
    FM --> Use
    FS --> Use

    Use --> Tune["hnsw_ef is your primary runtime tuning knob:\n- Recall below target: increase hnsw_ef\n- Latency above SLA: decrease hnsw_ef"]

    Use --> FST["NOTE: Qdrant full_scan_threshold\n(default 10000 KB) — segments smaller\nthan this use brute-force scan, not HNSW.\nSmall mutable segments from streaming\nwrites (<20K vectors) are searched\nvia brute force; hnsw_ef does not apply."]

    style Calc fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style Floor fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style FR fill:#14532d,stroke:#22c55e,color:#86efac
    style FM fill:#422006,stroke:#f59e0b,color:#fde68a
    style FS fill:#3b1f1f,stroke:#ef4444,color:#fca5a5
    style Use fill:#27272a,stroke:#3f3f46,color:#a1a1aa
    style Tune fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style FST fill:#27272a,stroke:#3f3f46,color:#a1a1aa
```

### Step 3d: Choose Oversampling (if quantizing)

**Qdrant parameter:** `search_params.quantization.oversampling` (set per-query)
Must be paired with: `search_params.quantization.rescore = true`

Example:
```python
client.query_points(..., search_params={
  "hnsw_ef": 200,
  "quantization": {"rescore": true, "oversampling": 2.0}
})
```

> **NOTE:** Qdrant also supports `quantization.ignore = true` to bypass
> quantization entirely for a specific query (uses full vectors).

Only applies when using quantization + rescoring from Stage 2.
If your quantization method is "None," skip this step.

| QUANTIZATION | RECALL REGIME | OVERSAMPLING FACTOR |
|-------------|---------------|---------------------|
| Scalar | RELAXED | 2.0x |
| Scalar | MODERATE | 2.5x |
| Scalar | STRICT | 4.0x |

If exploring binary quantization as a future optimization (see Stage 2 NOTE):

| QUANTIZATION | RECALL REGIME | OVERSAMPLING FACTOR |
|-------------|---------------|---------------------|
| Binary | RELAXED | 3.0x |
| Binary | MODERATE | 4.0x |
| Binary | STRICT | Not recommended (use scalar or none) |

> **rescore_candidates** = `oversampling_factor` x `top_k` — these candidates are re-ranked using full-precision vectors.

---

## Stage 4: Assess Your Latency Budget

This determines what can live on disk vs. what must be in RAM.

```mermaid
flowchart LR
    Start["P99 latency SLA?"] --> UT["< 20ms\nULTRA-TIGHT"]
    Start --> T["20-99ms\nTIGHT"]
    Start --> MO["100-500ms\nMODERATE"]
    Start --> RE["> 500ms\nRELAXED"]

    UT --> UTD["Everything in RAM\nNo mmap\nIs this SLA realistic?"]
    T --> TD["HNSW + quant vectors: RAM\nFull vectors: mmap OK\nDisk IOPS critical"]
    MO --> MOD["HNSW: RAM always\nQuant vectors: RAM preferred\nFull vectors: see table below"]
    RE --> RED["HNSW: RAM always\nEverything else: mmap OK\nGood for batch/offline"]

    style UT fill:#3b1f1f,stroke:#ef4444,color:#fca5a5
    style T fill:#422006,stroke:#f59e0b,color:#fde68a
    style MO fill:#14532d,stroke:#22c55e,color:#86efac
    style RE fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style Start fill:#18181b,stroke:#3f3f46,color:#e4e4e7
```

**Boundary rule:** exact boundary values belong to the lower tier. 20ms = TIGHT. 100ms = MODERATE. 500ms = MODERATE.

### Storage Placement Decision Table

Cross-reference Latency Tier with your components.

**Qdrant storage configuration:**
- Vectors on disk: set `on_disk: true` inside `vectors_config` (per-vector parameter)
    Example: vectors={"size": 3072, "distance": "Cosine", "on_disk": true}
- HNSW index on disk: set `on_disk: true` inside `hnsw_config`
- Quantized vectors in RAM: set `always_ram: true` inside `quantization_config`
- Automatic mmap promotion: configure `memmap_threshold` in `optimizers_config`
    (segments larger than this threshold in KB are automatically converted to mmap)

**No-quantization promotion rule**: If your quantization strategy from Stage 2 is
"no quantization," there are no separate quantized vectors — the full vectors ARE
the search vectors, and every distance calculation reads them. In this case, promote
full vectors one tier toward RAM (marked with * below).

| Component | ULTRA-TIGHT | TIGHT | MODERATE | RELAXED |
|-----------|-------------|-------|----------|---------|
| HNSW index | RAM | RAM | RAM | RAM |
| Quantized vectors | RAM | RAM | RAM or mmap | mmap OK |
| Full vectors | RAM | mmap (NVMe) | mmap (SSD) | mmap (SSD) |
| *if no quant: | RAM | RAM | RAM* | mmap (SSD) |
| Payload/metadata | RAM | RAM or mmap | mmap | mmap |

**"RAM or mmap"** = use RAM. Only fall back to mmap if RAM budget is exceeded after completing Stage 5.

> **\* No-quantization promotion:** When there is no quantization layer, every search query reads full vectors directly. mmap adds per-query I/O latency that compounds with high ef values. Promoting to RAM avoids this.

**Qdrant implementation:**

| Placement | Qdrant Config |
|-----------|--------------|
| RAM (vectors) | `on_disk: false` (default) |
| RAM (quantized) | `always_ram: true` in quantization_config |
| mmap (vectors) | `on_disk: true`, or rely on `memmap_threshold` auto-promotion |
| HNSW in RAM | `hnsw_config.on_disk: false` (default) |

---

## Stage 5: Size Your Memory

### Step 5a: Vector Memory

```mermaid
flowchart TD
    Q{"Quantization strategy\n(from Stage 2)?"} -->|"NO QUANTIZATION\n(float32)"| NoQ["vector_memory =\nnum_vectors x dimensions x 4 bytes"]
    Q -->|"SCALAR\n(int8)"| Scalar["quantized_memory =\nnum_vectors x dimensions x 1 byte\n\nfull_vector_memory =\nnum_vectors x dimensions x 4 bytes\n(for rescore, may be on disk)"]
    Q -->|"BINARY\n(1-bit)"| Binary["quantized_memory =\nnum_vectors x dimensions / 8\n\nfull_vector_memory =\nnum_vectors x dimensions x 4 bytes\n(for rescore, may be on disk)"]

    style Q fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style NoQ fill:#14532d,stroke:#22c55e,color:#86efac
    style Scalar fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style Binary fill:#422006,stroke:#f59e0b,color:#fde68a
```

### Step 5b: HNSW Index Memory

Use the `m` value you chose in Stage 3, Step 3a.

```mermaid
flowchart LR
    M["m\n(from Stage 3a)"] --> Formula["hnsw_memory =\nnum_vectors x m x 2 x 8 bytes x 1.1"]
    V["num_vectors"] --> Formula
    Formula --> Output["hnsw_memory\n(bytes)"]

    style M fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style V fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style Formula fill:#27272a,stroke:#3f3f46,color:#a1a1aa
    style Output fill:#14532d,stroke:#22c55e,color:#86efac
```

The formula: each vector has up to `m` bidirectional links at layer 0
(so m x 2 link slots), each stored as an 8-byte reference. The 1.1
multiplier (10%) accounts for the multi-level HNSW structure: a small
fraction of nodes are promoted to higher layers with additional links.

NOTE: This is a theoretical approximation. Qdrant does not publish
their exact graph memory layout. The formula is derived from the
original HNSW paper and is a reasonable estimate for sizing purposes.

**Common values (per 1M vectors):**

| m | Calculation | Approximate HNSW Memory |
|---|-------------|------------------------|
| 16 | 1,000,000 x 256 x 1.1 | ~282 MB |
| 20 | 1,000,000 x 320 x 1.1 | ~352 MB |
| 32 | 1,000,000 x 512 x 1.1 | ~563 MB |

### Step 5c: Page Cache Budget

```mermaid
flowchart TD
    Q{"Which components\nare on mmap?"} -->|"Full vectors on mmap\nWITH quantization\n(rescore path only)"| A["page_cache =\n0.10 x full_vector_memory\n\n(10% — rescoring accesses a small,\nsomewhat random subset per query)"]
    Q -->|"Full vectors on mmap\nWITHOUT quantization\n(every search reads them)"| B["page_cache =\n0.30 x full_vector_memory\n\n(30% — search traversal\nhas some locality)"]
    Q -->|"Quantized vectors\non mmap"| C["page_cache =\n0.50 x quantized_memory\n\n(50% — quantized vectors are\nsmall and heavily accessed)"]
    Q -->|"Everything in RAM\n(no mmap components)"| D["page_cache = 0"]

    style Q fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style A fill:#14532d,stroke:#22c55e,color:#86efac
    style B fill:#422006,stroke:#f59e0b,color:#fde68a
    style C fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style D fill:#27272a,stroke:#3f3f46,color:#a1a1aa
```

### Step 5d: Process Overhead

> **process_overhead = 500 MB (fixed)**
>
> This covers:
> - Qdrant server process (heap, thread stacks, buffers)
> - Point ID mapping and internal data structures
> - Payload index (for moderate payloads; add more if payloads are large)
> - OS kernel and system services

### Step 5e: Total RAM

```mermaid
flowchart TD
    RC["ram_components\n(HNSW index + quantized vectors if RAM\n+ full vectors if RAM)"] --> Sum["total_ram =\nram_components + page_cache\n+ process_overhead + merge_headroom"]
    PC["page_cache\n(from Step 5c)"] --> Sum
    PO["process_overhead\n(500 MB)"] --> Sum
    MH["merge_headroom\n(ram_components x 0.10)"] --> Sum

    Sum --> Round["Round up to nearest\nstandard instance RAM tier:\n4, 8, 16, 32, 64, 128, 256 GB"]

    style RC fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style PC fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style PO fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style MH fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style Sum fill:#27272a,stroke:#3f3f46,color:#a1a1aa
    style Round fill:#14532d,stroke:#22c55e,color:#86efac
```

---

## Stage 6: Size Your Disk

Disk is sized separately from RAM. Everything persists to disk regardless of whether
it's also in RAM (Qdrant uses disk as the durable backing store).

### Step 6a: Base Disk

```mermaid
flowchart LR
    FV["full_vector_memory\n(num_vectors x dimensions x 4 bytes)"] --> Sum["base_disk =\nfull_vector_memory\n+ quantized_memory\n+ hnsw_on_disk"]
    QM["quantized_memory\n(0 if no quantization)"] --> Sum
    HN["hnsw_on_disk\n(= hnsw_memory, persisted\neven if loaded into RAM)"] --> Sum

    style FV fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style QM fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style HN fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style Sum fill:#14532d,stroke:#22c55e,color:#86efac
```

### Step 6b: WAL (Write-Ahead Log) Space

Qdrant uses a WAL for durability. Configuration (in qdrant config YAML):
`wal.wal_capacity_mb` (default 32 MB per segment), `wal.wal_segments_ahead` (default 0).
WAL accumulates writes before they're flushed to segments.

```mermaid
flowchart TD
    Start{"Write pattern?"} -->|"STREAMING"| Stream["wal_space = 2 GB\n\nSufficient for most streaming\nworkloads up to 500 writes/s.\nFor >500/s use:\npeak_write_rate x avg_vector_bytes\nx 60 seconds x 2"]
    Start -->|"BATCH"| Batch["wal_space =\nbatch_size x avg_vector_bytes x 2\n\nwhere avg_vector_bytes =\ndimensions x 4 (float32)\nThe x2 accounts for WAL segments\ncoexisting before compaction\n\nFloor: max(calculated, 1 GB)"]
    Start -->|"RARE / STATIC"| Static["wal_space = 1 GB\n(minimal, just for\noperational headroom)"]

    style Start fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style Stream fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style Batch fill:#422006,stroke:#f59e0b,color:#fde68a
    style Static fill:#14532d,stroke:#22c55e,color:#86efac
```

### Step 6c: Operational Headroom

Disk operations that temporarily require extra space:

```mermaid
flowchart TD
    BD["base_disk"] --> Merge["segment_merge_overhead =\nbase_disk x 0.50\n\nDuring merges, old and new segments\ncoexist. Worst case is a full rebuild\n= 1.0x, but 0.5x covers typical\nincremental merges."]
    BD --> Snap["snapshot_space =\nbase_disk x 1.0\n\nOne full snapshot for backup/recovery.\nSet to 0 if snapshots are stored\nexternally or not used."]

    style BD fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style Merge fill:#422006,stroke:#f59e0b,color:#fde68a
    style Snap fill:#422006,stroke:#f59e0b,color:#fde68a
```

### Step 6d: Total Disk

```mermaid
flowchart TD
    BD["base_disk"] --> Sum["sum =\nbase_disk + wal_space\n+ segment_merge_overhead\n+ snapshot_space"]
    WS["wal_space"] --> Sum
    MO["segment_merge_overhead"] --> Sum
    SS["snapshot_space"] --> Sum

    Sum --> Max["total_disk =\nmax(sum, 30 GB)\n\nMinimum 30 GB recommended\nto avoid filesystem overhead\non very small datasets"]

    style BD fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style WS fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style MO fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style SS fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style Sum fill:#27272a,stroke:#3f3f46,color:#a1a1aa
    style Max fill:#14532d,stroke:#22c55e,color:#86efac
```

### Step 6e: Disk Type

```mermaid
flowchart TD
    Q{"Are any components on mmap\nAND in the query path?"}
    Q -->|"YES, TIGHT latency tier"| NVMe["NVMe SSD required\n(mmap reads must be <100us for latency budget)"]
    Q -->|"YES, MODERATE/RELAXED"| SSD["Standard SSD sufficient\n(e.g., gp3)"]
    Q -->|"NO (everything in RAM)"| SSD2["Standard SSD sufficient\n(disk only for persistence and batch loads)"]

    style Q fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style NVMe fill:#3b1f1f,stroke:#ef4444,color:#fca5a5
    style SSD fill:#14532d,stroke:#22c55e,color:#86efac
    style SSD2 fill:#14532d,stroke:#22c55e,color:#86efac
```

---

## Stage 7: Size Your Compute (vCPUs)

### Step 7a: Estimate Per-Query CPU Time

Per-query time depends on the **absolute value of ef** (not the ratio ef/top_k)
and the cost of each distance computation (driven by dimensions and quantization).

**Step 1: Look up base distance computation cost per query.**

| Dimensions | Binary quant | Scalar quant | No quant (float32) |
|------------|-------------|-------------|-------------------|
| dims < 512 | ~0.3ms | ~0.5ms | ~1.0ms |
| dims 512-1024 | ~0.5ms | ~1.0ms | ~2.0ms |
| dims 1024-2048 | ~0.8ms | ~1.5ms | ~3.0ms |
| dims 2048-4096 | ~1.0ms | ~2.0ms | ~5.0ms |

These assume: ef = 64 (the reference point for all adjustments below), warm cache (data in RAM or OS page cache), single-threaded per query, includes graph traversal overhead (memory access, pointer chasing), does NOT include rescoring (added separately below).

**Step 2: Adjust for your actual ef value.**

The relationship between ef and query time is roughly linear (not
exactly, due to early termination, but close enough for sizing).

```mermaid
flowchart LR
    EF["your_ef"] --> Div["ef_adjustment =\nyour_ef / 64"] --> Mul["per_query_base =\ntable_value x ef_adjustment"]
    TV["table_value\n(from Step 1)"] --> Mul

    style EF fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style TV fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style Div fill:#27272a,stroke:#3f3f46,color:#a1a1aa
    style Mul fill:#14532d,stroke:#22c55e,color:#86efac
```

Examples: ef = 64 → adjustment = 1.0x (baseline), ef = 128 → 2.0x, ef = 200 → 3.1x, ef = 256 → 4.0x, ef = 512 → 8.0x.

**Step 3: Add rescoring cost (if quantizing).**

```mermaid
flowchart TD
    Q{"Using quantization\nwith rescoring?"} -->|"YES"| Rescore["rescore_candidates =\noversampling_factor x top_k\n(from Stage 3d)"]
    Q -->|"NO (no quant)"| NoRescore["rescore_time = 0"]

    Rescore --> Where{"Full vectors\nstored where?"}
    Where -->|"In RAM"| RAM["rescore_time =\nrescore_candidates x dims x 4 bytes\n/ 10,000,000,000\n\n(~10 GB/s effective throughput\nfor in-RAM float32 dot products)\nFloor: 0.1ms"]
    Where -->|"On mmap (disk)"| Disk["rescore_time =\nrescore_candidates x 0.01ms\n\n(per-candidate disk read + compute)\nExample: 200 candidates x 0.01ms = 2.0ms"]

    style Q fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style Rescore fill:#27272a,stroke:#3f3f46,color:#a1a1aa
    style NoRescore fill:#14532d,stroke:#22c55e,color:#86efac
    style Where fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style RAM fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style Disk fill:#422006,stroke:#f59e0b,color:#fde68a
```

**Step 4: Add mmap overhead (if applicable).**

```mermaid
flowchart TD
    Q{"Full vectors on mmap\nAND no quantization?\n(every search reads disk)"} -->|"YES"| Add["mmap_overhead = 1.0ms\n\n(Random I/O pattern during graph traversal;\npage cache helps but misses add up)"]
    Q -->|"NO"| None["mmap_overhead = 0"]

    style Q fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style Add fill:#422006,stroke:#f59e0b,color:#fde68a
    style None fill:#14532d,stroke:#22c55e,color:#86efac
```

**Step 5: Total per-query time.**

> **per_query_time** = `per_query_base` + `rescore_time` + `mmap_overhead`

### Step 7b: Calculate Required Cores for QPS

> **cores_for_queries** = `target_QPS` x `per_query_time_seconds`

Then add headroom based on latency tier and write pattern:

```mermaid
flowchart TD
    Q{"Latency tier?"} -->|"TIGHT"| Tight{"Write pattern?"}
    Q -->|"MODERATE, RELAXED,\nor ULTRA-TIGHT"| Other["headroom = 30%\nheadroom_cores = cores_for_queries x 0.30"]

    Tight -->|"STREAMING"| TS["headroom = 100%\nheadroom_cores = cores_for_queries x 1.0\n\n(Streaming writes cause segment merges\nand lock contention that amplify tail latency.\nTight P99 budget leaves no room for\nwrite-induced spikes.)"]
    Tight -->|"BATCH / STATIC"| TB["headroom = 50%\nheadroom_cores = cores_for_queries x 0.50"]

    style Q fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style Tight fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style TS fill:#3b1f1f,stroke:#ef4444,color:#fca5a5
    style TB fill:#422006,stroke:#f59e0b,color:#fde68a
    style Other fill:#14532d,stroke:#22c55e,color:#86efac
```

This headroom covers: GC pauses, OS scheduling, background tasks,
and the gap between average and P99 query times.

### Step 7c: Account for Write Load

```mermaid
flowchart TD
    Q{"Write pattern?"} -->|"STREAMING\n(concurrent reads + writes)"| Stream["Look up indexing time per vector by m:\n\nm=12: 1.0ms   m=16: 1.5ms   m=20: 2.5ms\nm=28: 3.5ms   m=32: 4.0ms\n\nwrite_cores =\npeak_write_rate x indexing_time_per_vector_seconds"]
    Q -->|"BATCH\n(no concurrent reads)"| Batch["write_cores = 0\n\nSize for peak read QPS only.\nWrites use the same cores during off-peak."]

    style Q fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style Stream fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style Batch fill:#14532d,stroke:#22c55e,color:#86efac
```

### Step 7d: Total vCPUs

> **total_vcpus** = `cores_for_queries` + `headroom_cores` + `write_cores` -- round up to nearest whole number, minimum 4 vCPUs (to allow OS, Qdrant, and at least 2 query threads)

### Step 7e: Single Node or Multiple?

```mermaid
flowchart TD
    Check{"total_vcpus fits in one machine?\nAND total_ram fits?"}
    Check -->|"YES to both"| Single["Single node\nnodes=1, shards=1, replicas=1"]
    Check -->|"NO to either"| Multi{"What's the bottleneck?"}

    Multi -->|"RAM doesn't fit"| Shards["Shards\nPartition data across nodes\nnum_shards = ceil(total_ram / max_ram)"]
    Multi -->|"CPU too high"| Replicas["Replicas\nFull copies, distribute QPS\nnum_replicas = ceil(total_vcpus / max_vcpus)"]

    Single --> HA["Present HA option\nto customer"]
    Single --> Burst["Flag bursty QPS\nas future optimization"]

    style Single fill:#14532d,stroke:#22c55e,color:#86efac
    style Shards fill:#422006,stroke:#f59e0b,color:#fde68a
    style Replicas fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style Check fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style Multi fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style HA fill:#27272a,stroke:#3f3f46,color:#a1a1aa
    style Burst fill:#27272a,stroke:#3f3f46,color:#a1a1aa
```

**Single node is the default.** Topology is a CUSTOMER DECISION, not a sizing decision.
This guide produces the minimum viable infrastructure. Replication is a business/SLA
decision the customer makes after reviewing your initial architecture.

- **HA option:** Present as a recommendation — "add `replication_factor=2` for failover.
  Doubles infrastructure cost but prevents SLA breach during node failure. Recommended
  if the P99 SLA is contractual." Record under Future Optimizations.
- **Bursty QPS:** If peak QPS occurs in a defined window (e.g., 6 hours/day), the node
  will be idle outside that window. This is expected. Scaling replicas up/down is a cost
  optimization to explore AFTER validating the single-node architecture works.
- **Shards:** Each query must hit all shards, adding ~2-5ms network latency. Use only
  when data doesn't fit in one node's RAM.
- **Replicas:** Each replica needs `total_ram`, so total cost = replicas x node cost.

**Qdrant collection creation parameters:**

| Parameter | Purpose |
|-----------|---------|
| `shard_number` | Number of shards (set at creation, manual choice) |
| `replication_factor` | Copies per shard (e.g., 2 = primary + 1 replica) |
| `write_consistency_factor` | Replicas that must ACK a write before responding (must be <= `replication_factor`; set to 1 for async replication) |

---

## Stage 8: Tune for Write Pattern

```mermaid
flowchart TD
    Start["Write pattern?"] --> S["STREAMING\nContinuous writes\nconcurrent with reads"]
    Start --> B["BATCH\nPeriodic bulk loads\nno concurrent reads"]
    Start --> R["RARE / STATIC\nLoad once, mostly read"]

    S --> SLow{"Write rate?"}
    SLow -->|"< 500/s"| SStd["Standard config\nDefault segments OK"]
    SLow -->|"> 500/s"| SOpt["Optimize throughput\nIncrease WAL capacity"]

    B --> BIdx["Disable indexing during load\nRebuild after"]
    B --> BAlias["Consider collection aliasing\nfor zero-downtime swaps"]

    R --> RRead["Optimize for reads\nCompact segments\non_disk: false"]

    style S fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style B fill:#422006,stroke:#f59e0b,color:#fde68a
    style R fill:#14532d,stroke:#22c55e,color:#86efac
    style Start fill:#18181b,stroke:#3f3f46,color:#e4e4e7
```

**STREAMING (< 500/s):** Standard configuration. Qdrant handles background merging
automatically via three optimizers: Merge (combines small segments), Vacuum (reclaims
deleted records), and Indexing (promotes segments to HNSW + mmap). Key config:
`optimizers_config.flush_interval_sec` (default: 5).

**STREAMING (> 500/s):** Increase WAL capacity (`wal.wal_capacity_mb`, default 32),
increase `wal.wal_segments_ahead` for write buffering, consider larger `memmap_threshold`.
Monitor segment count — too many small segments hurts read performance.

**BATCH:** "No concurrent reads" is functionally equivalent to "offline during load."

- **Preferred:** Set `hnsw_config.m = 0` at collection creation to disable HNSW graph
  construction during bulk load. After loading, update the collection to set m to your
  target value — Qdrant rebuilds the HNSW index automatically.
- **Alternative:** Set `optimizers_config.indexing_threshold = 0` to defer indexing.
  (NOTE: 0 = disabled. This is the opposite of what you might expect.)
- Set `max_optimization_threads` high during rebuild (null = dynamic CPU saturation).
- Consider **collection aliasing** for zero-downtime reindexing:

```json
{
  "actions": [
    {"delete_alias": {"alias_name": "production"}},
    {"create_alias": {"collection_name": "new_v2", "alias_name": "production"}}
  ]
}
```

Multiple actions in one request are executed atomically. Enables rollback.

**RARE / STATIC:** Optimize entirely for reads. Compact segments after initial load.
Set `on_disk: false` for everything that fits in RAM. No write headroom needed in CPU.

---

## Stage 9: Select Instance and Estimate Cost

### Step 9a: Map Requirements to Instance Type

```mermaid
flowchart TD
    Inputs["From your calculations:\ntotal_vcpus (Stage 7d)\ntotal_ram (Stage 5e)\ntotal_disk (Stage 6d)\ndisk_type (Stage 6e)"] --> Q{"Dominant constraint?"}
    Q -->|"total_ram > total_vcpus x 4 GB\n(Memory-bound)"| R["R-series\n(memory-optimized)\n\nAWS: r6i, r7i\nGCP: n2-highmem\nAzure: E-series"]
    Q -->|"total_vcpus x 4 GB > total_ram\n(Compute-bound)"| C["C-series\n(compute-optimized)\n\nAWS: c6i, c7i\nGCP: c3\nAzure: F-series"]
    Q -->|"Roughly balanced"| M["M-series\n(general-purpose)\n\nAWS: m6i, m7i\nGCP: n2-standard\nAzure: D-series"]

    R --> Pick["Pick smallest instance where:\ninstance_vcpus >= total_vcpus\ninstance_ram >= total_ram\ncan attach required disk type/size"]
    C --> Pick
    M --> Pick

    Pick --> Over{"Selected instance has\n> 1.5x your total_vcpus?"}
    Over -->|"YES"| Down["Consider next smaller instance\n\n1-2 vCPU shortfall is acceptable —\nStage 7 headroom already covers burst.\nExample: total_vcpus=17, choices are\nc6i.4xlarge (16) vs c6i.8xlarge (32)\n→ use c6i.4xlarge (88% overprovision avoided)"]
    Over -->|"NO"| OK["Instance size is appropriate"]

    Pick --> NVMe{"Stage 6e requires NVMe\nbut family lacks local NVMe?"}
    NVMe -->|"YES"| NVMeOpts["Option 1: High-IOPS EBS volume\n(io2 or gp3 w/ provisioned IOPS)\n\nOption 2: NVMe instance family\n(i3, i4i, c6id, m6id, r6id)\n\nDefault: gp3 with 16,000 IOPS provisioned.\nFlag NVMe as benchmark refinement."]
    NVMe -->|"NO"| Done["Proceed to cost estimate"]

    style Inputs fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style Q fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style R fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    style C fill:#422006,stroke:#f59e0b,color:#fde68a
    style M fill:#14532d,stroke:#22c55e,color:#86efac
    style Pick fill:#27272a,stroke:#3f3f46,color:#a1a1aa
    style Over fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style Down fill:#422006,stroke:#f59e0b,color:#fde68a
    style OK fill:#14532d,stroke:#22c55e,color:#86efac
    style NVMe fill:#18181b,stroke:#3f3f46,color:#e4e4e7
    style NVMeOpts fill:#27272a,stroke:#3f3f46,color:#a1a1aa
    style Done fill:#14532d,stroke:#22c55e,color:#86efac
```

### Step 9b: Estimate Monthly Cost

> **monthly_cost** = `instance_hourly_rate` x `730 hours` x `num_nodes` + `disk_gb` x `disk_price_per_gb_month` x `num_nodes`

For reserved instances (1-year commitment): multiply by 0.60-0.70.

| Instance | vCPU | RAM | Hourly Rate | ~Monthly Cost |
|----------|------|-----|------------|---------------|
| c6i.xlarge | 4 | 8 GB | $0.17/hr | $124/mo |
| c6i.2xlarge | 8 | 16 GB | $0.34/hr | $248/mo |
| c6i.4xlarge | 16 | 32 GB | $0.68/hr | $496/mo |
| c6i.8xlarge | 32 | 64 GB | $1.36/hr | $993/mo |
| c6i.12xlarge | 48 | 96 GB | $2.04/hr | $1489/mo |
| m6i.xlarge | 4 | 16 GB | $0.192/hr | $140/mo |
| m6i.2xlarge | 8 | 32 GB | $0.384/hr | $280/mo |
| m6i.4xlarge | 16 | 64 GB | $0.768/hr | $561/mo |
| r6i.xlarge | 4 | 32 GB | $0.252/hr | $184/mo |
| r6i.2xlarge | 8 | 64 GB | $0.504/hr | $368/mo |
| gp3 SSD | — | — | — | $0.08/GB/mo |
| NVMe (local) | — | — | — | included in i-series/d-series |

---

## Stage 10: Build Your Architecture Summary

Fill in this template with your decisions from Stages 1-9.

### Dataset & Classification

| Field | Value |
|-------|-------|
| Dataset | _______________ |
| Vectors | _______________ |
| Dimensions | _______________ |
| Embedding Class | _______________ |
| Distance Metric | _______________ |

### SLA Requirements

| Field | Value |
|-------|-------|
| Target QPS | _______________ |
| P99 Latency | _______________ |
| Recall Target | _______________ |
| Top-k | _______________ |
| Recall Regime | _______________ |
| Latency Tier | _______________ |
| Write Pattern | _______________ |

### Quantization & HNSW (from Stages 2-3)

| Field | Value |
|-------|-------|
| Quantization method | _______________ |
| Oversampling | _______________ |
| Rescore | _______________ |
| m | _______________ |
| ef_construct | _______________ |
| hnsw_ef (search) | _______________ _(preliminary — benchmark)_ |

### Memory (from Stage 5)

| Component | Size | Location |
|-----------|------|----------|
| Quantized vectors | _______________ | RAM / disk |
| Full vectors | _______________ | RAM / disk |
| HNSW index | _______________ | RAM |
| Page cache | _______________ | — |
| Process overhead | 500 MB | — |
| Merge headroom | _______________ | — |
| **Total RAM** | **_______________** | |

### Disk (from Stage 6)

| Component | Size |
|-----------|------|
| Base disk | _______________ |
| WAL space | _______________ |
| Segment merge overhead | _______________ |
| Snapshot space | _______________ |
| **Total disk** | **_______________** |
| Disk type | _______________ |

### Compute (from Stage 7)

| Component | Value |
|-----------|-------|
| Per-query CPU time | _______________ |
| Cores for QPS | _______________ |
| Cores for writes | _______________ |
| Headroom (___%) | _______________ |
| **Total vCPUs** | **_______________** |

### Topology & Cost (from Stages 7e, 9)

| Field | Value |
|-------|-------|
| Nodes / Shards / Replicas | ___ / ___ / ___ |
| Instance type | _______________ |
| Cost/month (on-demand) | _______________ |
| Cost/month (reserved) | _______________ |

### Future Optimizations

1. _______________
2. _______________
3. _______________

---

## Quick Reference: Common Archetypes

For rapid pattern-matching, here are common scenarios you'll encounter:

### Archetype 1: "High-dim text search, good-enough recall"
- OpenAI / Cohere / BGE embeddings, 1536-4096 dims
- 95% recall, <100ms latency, moderate QPS
- **Recipe**: Scalar quantization, m=16, ef=2x top_k, oversample 2x, mmap full vectors on NVMe
- **Cost profile**: Low RAM (quantized vectors are 4x smaller), moderate disk
- **Future optimization**: Test binary quantization for 32x compression if RAM is critical

### Archetype 2: "High-dim text search, strict recall"
- Same embeddings as above
- 99% recall, <100ms latency
- **Recipe**: Scalar quantization, m=32, ef=6x top_k, oversample 4x
- **Cost profile**: Moderate RAM, moderate disk

### Archetype 3: "Low-dim features, strict recall"
- CV features, scientific embeddings, 128-960 dims
- 99% recall
- **Recipe**: No quantization (full float32 in RAM), m=32, ef=6x top_k, Euclidean distance
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

8. **"Cosine distance works for everything"**
   Classical CV features (GIST, SIFT, HOG) use Euclidean (L2) distance. Neural
   embeddings typically use Cosine. Using the wrong metric silently degrades recall.
