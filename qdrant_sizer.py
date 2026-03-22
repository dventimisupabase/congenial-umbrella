#!/usr/bin/env python3
"""
Qdrant Cluster Sizer — Mechanical implementation of the Qdrant Field Guide.

Takes discovery inputs (embedding type, dimensions, vector count, QPS, latency SLA,
recall target, top-k, write pattern) and produces a complete architecture recommendation.

Usage:
    Interactive:  python qdrant_sizer.py
    Programmatic: python qdrant_sizer.py --json '{"embedding_type": "neural_text", ...}'
    From file:    python qdrant_sizer.py --file scenario.json
"""

import json
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Stage outputs — each stage produces a dataclass consumed by later stages
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingClassification:
    embedding_class: str          # "A", "B", "C"
    class_label: str              # "Binary-friendly neural", etc.
    distance_metric: str          # "Cosine", "Euclid", "Dot", "Manhattan"
    qdrant_distance: str          # Qdrant API value


@dataclass
class RecallRegime:
    regime: str                   # "RELAXED", "MODERATE", "STRICT"
    quantization: str             # "binary", "scalar", "none"
    rescore: bool


@dataclass
class HNSWParams:
    m: int
    ef_construct: int
    hnsw_ef: int
    hnsw_ef_note: str             # e.g., "floor-adjusted from 60"
    oversampling: Optional[float]
    rescore_candidates: Optional[int]


@dataclass
class LatencyAssessment:
    latency_tier: str             # "ULTRA-TIGHT", "TIGHT", "MODERATE", "RELAXED"
    hnsw_placement: str           # "RAM"
    quantized_placement: str      # "RAM", "mmap", "N/A"
    full_vector_placement: str    # "RAM", "mmap (NVMe)", "mmap (SSD)"
    no_quant_promotion: bool


@dataclass
class MemorySizing:
    quantized_memory_mb: float
    full_vector_memory_mb: float
    hnsw_memory_mb: float
    page_cache_mb: float
    process_overhead_mb: float
    merge_headroom_mb: float
    total_ram_mb: float
    total_ram_rounded_gb: int


@dataclass
class DiskSizing:
    base_disk_mb: float
    wal_space_mb: float
    segment_merge_overhead_mb: float
    snapshot_space_mb: float
    total_disk_mb: float
    total_disk_gb: int
    disk_type: str


@dataclass
class ComputeSizing:
    per_query_base_ms: float
    ef_adjustment: float
    rescore_time_ms: float
    mmap_overhead_ms: float
    per_query_time_ms: float
    cores_for_queries: float
    headroom_pct: int
    headroom_cores: float
    write_cores: float
    total_vcpus: int


@dataclass
class TopologyDecision:
    nodes: int
    shards: int
    replicas: int
    note: str


@dataclass
class InstanceSelection:
    instance_family: str          # "C-series", "M-series", "R-series"
    dominant_constraint: str      # "compute", "memory", "balanced"
    instance_type: str
    instance_vcpus: int
    instance_ram_gb: int
    cost_on_demand_mo: float
    cost_reserved_mo: float
    disk_cost_mo: float
    total_on_demand_mo: float
    total_reserved_mo: float
    overprovision_note: str


@dataclass
class ArchitectureSummary:
    # Inputs
    dataset_name: str
    num_vectors: int
    dimensions: int
    peak_qps: int
    write_pattern: str
    peak_write_rate: int
    batch_size: int
    p99_latency_ms: int
    recall_target: float
    top_k: int
    # Stage outputs
    classification: EmbeddingClassification
    recall_regime: RecallRegime
    hnsw: HNSWParams
    latency: LatencyAssessment
    memory: MemorySizing
    disk: DiskSizing
    compute: ComputeSizing
    topology: TopologyDecision
    instance: InstanceSelection
    future_optimizations: list


# ---------------------------------------------------------------------------
# Instance catalog
# ---------------------------------------------------------------------------

INSTANCES = [
    # (type, vcpus, ram_gb, family, hourly_rate)
    ("c6i.xlarge",   4,   8, "C", 0.170),
    ("c6i.2xlarge",  8,  16, "C", 0.340),
    ("c6i.4xlarge", 16,  32, "C", 0.680),
    ("c6i.8xlarge", 32,  64, "C", 1.360),
    ("c6i.12xlarge",48,  96, "C", 2.040),
    ("m6i.xlarge",   4,  16, "M", 0.192),
    ("m6i.2xlarge",  8,  32, "M", 0.384),
    ("m6i.4xlarge", 16,  64, "M", 0.768),
    ("r6i.xlarge",   4,  32, "R", 0.252),
    ("r6i.2xlarge",  8,  64, "R", 0.504),
]

RAM_TIERS = [4, 8, 16, 32, 64, 128, 256]


# ---------------------------------------------------------------------------
# Stage 1: Classify Embeddings
# ---------------------------------------------------------------------------

def classify_embeddings(embedding_type: str, dimensions: int) -> EmbeddingClassification:
    if embedding_type == "neural_text":
        distance = "Cosine"
        if dimensions >= 1024:
            return EmbeddingClassification("A", "Binary-friendly neural", distance, "Cosine")
        else:
            return EmbeddingClassification("B", "Scalar-friendly neural", distance, "Cosine")
    elif embedding_type == "neural_vision":
        return EmbeddingClassification("B", "Scalar-friendly neural", "Cosine", "Cosine")
    elif embedding_type == "classical_cv":
        return EmbeddingClassification("C", "Quantization-resistant", "Euclidean (L2)", "Euclid")
    else:
        return EmbeddingClassification("B", "Scalar-friendly neural (default)", "Cosine", "Cosine")


# ---------------------------------------------------------------------------
# Stage 2: Determine Recall Regime
# ---------------------------------------------------------------------------

QUANT_MATRIX = {
    # (class, regime) -> quantization method
    ("A", "RELAXED"):  "scalar",
    ("A", "MODERATE"): "scalar",
    ("A", "STRICT"):   "scalar",  # could be "none"; scalar is conservative default
    ("B", "RELAXED"):  "scalar",
    ("B", "MODERATE"): "scalar",
    ("B", "STRICT"):   "none",
    ("C", "RELAXED"):  "scalar",
    ("C", "MODERATE"): "none",
    ("C", "STRICT"):   "none",
}


def determine_recall_regime(recall_target: float, embedding_class: str) -> RecallRegime:
    if recall_target <= 0.95:
        regime = "RELAXED"
    elif recall_target <= 0.98:
        regime = "MODERATE"
    else:
        regime = "STRICT"

    quant = QUANT_MATRIX[(embedding_class, regime)]
    rescore = quant != "none"

    return RecallRegime(regime, quant, rescore)


# ---------------------------------------------------------------------------
# Stage 3: Set HNSW Parameters
# ---------------------------------------------------------------------------

M_BASE = {"RELAXED": 16, "MODERATE": 20, "STRICT": 32}
M_UP   = {"RELAXED": 20, "MODERATE": 28, "STRICT": 32}   # top_k >= 50
M_DOWN = {"RELAXED": 16, "MODERATE": 16, "STRICT": 24}    # top_k <= 20

EF_CONSTRUCT_BASE = {"RELAXED": 200, "MODERATE": 256, "STRICT": 400}
EF_CONSTRUCT_BATCH_STRICT = 400  # same as base now, but explicitly stated

EF_MULTIPLIER = {"RELAXED": 2, "MODERATE": 3, "STRICT": 6}
EF_FLOOR      = {"RELAXED": 64, "MODERATE": 64, "STRICT": 128}

OVERSAMPLING = {
    ("scalar", "RELAXED"):  2.0,
    ("scalar", "MODERATE"): 2.5,
    ("scalar", "STRICT"):   4.0,
    ("binary", "RELAXED"):  3.0,
    ("binary", "MODERATE"): 4.0,
}

# Indexing time per vector (ms) by m value
INDEXING_TIME_MS = {12: 1.0, 16: 1.5, 20: 2.5, 24: 3.0, 28: 3.5, 32: 4.0}


def set_hnsw_params(regime: str, top_k: int, quantization: str,
                    write_pattern: str) -> HNSWParams:
    # Step 3a: m with top-k adjustment
    if top_k >= 50:
        m = M_UP[regime]
    elif top_k <= 20:
        m = M_DOWN[regime]
    else:
        m = M_BASE[regime]

    # Step 3b: ef_construct
    if write_pattern == "batch" and regime == "STRICT":
        ef_construct = EF_CONSTRUCT_BATCH_STRICT
    else:
        ef_construct = EF_CONSTRUCT_BASE[regime]

    # Step 3c: hnsw_ef
    calculated_ef = EF_MULTIPLIER[regime] * top_k
    floor = EF_FLOOR[regime]
    hnsw_ef = max(calculated_ef, floor)
    if hnsw_ef > calculated_ef:
        ef_note = f"floor-adjusted from {calculated_ef} to {hnsw_ef}"
    else:
        ef_note = f"{EF_MULTIPLIER[regime]}x top_k"

    # Step 3d: oversampling
    if quantization == "none":
        oversampling = None
        rescore_candidates = None
    else:
        key = (quantization, regime)
        oversampling = OVERSAMPLING.get(key)
        rescore_candidates = int(oversampling * top_k) if oversampling else None

    return HNSWParams(m, ef_construct, hnsw_ef, ef_note, oversampling, rescore_candidates)


# ---------------------------------------------------------------------------
# Stage 4: Assess Latency Budget
# ---------------------------------------------------------------------------

def assess_latency(p99_ms: int, quantization: str) -> LatencyAssessment:
    if p99_ms < 20:
        tier = "ULTRA-TIGHT"
    elif p99_ms <= 99:
        tier = "TIGHT"
    elif p99_ms <= 500:
        tier = "MODERATE"
    else:
        tier = "RELAXED"

    no_quant = quantization == "none"

    # Storage placement lookup
    placement = {
        "ULTRA-TIGHT": ("RAM", "RAM", "RAM"),
        "TIGHT":       ("RAM", "RAM", "mmap (NVMe)"),
        "MODERATE":    ("RAM", "RAM or mmap", "mmap (SSD)"),
        "RELAXED":     ("RAM", "mmap", "mmap (SSD)"),
    }

    hnsw_p, quant_p, full_p = placement[tier]

    # No-quantization promotion
    promoted = False
    if no_quant:
        quant_p = "N/A"
        promotion_map = {
            "ULTRA-TIGHT": "RAM",
            "TIGHT":       "RAM",
            "MODERATE":    "RAM",
            "RELAXED":     "mmap (SSD)",
        }
        full_p = promotion_map[tier]
        promoted = tier in ("MODERATE",)  # only MODERATE gets promoted

    return LatencyAssessment(tier, hnsw_p, quant_p, full_p, promoted)


# ---------------------------------------------------------------------------
# Stage 5: Size Memory
# ---------------------------------------------------------------------------

def size_memory(num_vectors: int, dimensions: int, quantization: str,
                hnsw: HNSWParams, latency: LatencyAssessment) -> MemorySizing:
    # 5a: Vector memory
    full_vector_mb = num_vectors * dimensions * 4 / (1024 * 1024)

    if quantization == "scalar":
        quantized_mb = num_vectors * dimensions * 1 / (1024 * 1024)
    elif quantization == "binary":
        quantized_mb = num_vectors * dimensions / 8 / (1024 * 1024)
    else:
        quantized_mb = 0

    # 5b: HNSW index memory
    hnsw_mb = num_vectors * hnsw.m * 2 * 8 * 1.1 / (1024 * 1024)

    # Determine what's in RAM
    ram_components = hnsw_mb  # HNSW always in RAM

    if latency.quantized_placement in ("RAM",):
        ram_components += quantized_mb

    if latency.full_vector_placement == "RAM":
        ram_components += full_vector_mb

    # 5c: Page cache
    full_on_mmap = "mmap" in latency.full_vector_placement
    quant_on_mmap = "mmap" in (latency.quantized_placement or "")

    page_cache = 0
    if full_on_mmap and quantization != "none":
        page_cache = 0.10 * full_vector_mb
    elif full_on_mmap and quantization == "none":
        page_cache = 0.30 * full_vector_mb
    if quant_on_mmap:
        page_cache += 0.50 * quantized_mb

    # 5d: Process overhead
    process_overhead = 500  # MB

    # 5e: Total
    merge_headroom = ram_components * 0.10
    total_ram = ram_components + page_cache + process_overhead + merge_headroom

    # Round to nearest tier
    rounded_gb = 4
    for tier in RAM_TIERS:
        if tier * 1024 >= total_ram:
            rounded_gb = tier
            break

    return MemorySizing(
        quantized_memory_mb=round(quantized_mb, 1),
        full_vector_memory_mb=round(full_vector_mb, 1),
        hnsw_memory_mb=round(hnsw_mb, 1),
        page_cache_mb=round(page_cache, 1),
        process_overhead_mb=process_overhead,
        merge_headroom_mb=round(merge_headroom, 1),
        total_ram_mb=round(total_ram, 1),
        total_ram_rounded_gb=rounded_gb,
    )


# ---------------------------------------------------------------------------
# Stage 6: Size Disk
# ---------------------------------------------------------------------------

def size_disk(num_vectors: int, dimensions: int, memory: MemorySizing,
              write_pattern: str, peak_write_rate: int, batch_size: int,
              latency: LatencyAssessment) -> DiskSizing:
    # 6a: Base disk
    base_disk = memory.full_vector_memory_mb + memory.quantized_memory_mb + memory.hnsw_memory_mb

    # 6b: WAL space
    avg_vector_bytes = dimensions * 4
    if write_pattern == "streaming":
        wal_mb = 2048  # 2 GB
    elif write_pattern == "batch":
        wal_bytes = batch_size * avg_vector_bytes * 2
        wal_mb = max(wal_bytes / (1024 * 1024), 1024)  # floor 1 GB
    else:
        wal_mb = 1024  # 1 GB

    # 6c: Operational headroom
    merge_overhead = base_disk * 0.50
    snapshot_space = base_disk * 1.0

    # 6d: Total
    total_mb = base_disk + wal_mb + merge_overhead + snapshot_space
    total_gb = max(math.ceil(total_mb / 1024), 30)  # 30 GB minimum

    # 6e: Disk type
    mmap_in_query = ("mmap" in latency.full_vector_placement or
                     "mmap" in (latency.quantized_placement or ""))
    if mmap_in_query and latency.latency_tier == "TIGHT":
        disk_type = "NVMe SSD (gp3 w/ provisioned IOPS)"
    elif mmap_in_query:
        disk_type = "Standard SSD (gp3)"
    else:
        disk_type = "Standard SSD (gp3)"

    return DiskSizing(
        base_disk_mb=round(base_disk, 1),
        wal_space_mb=round(wal_mb, 1),
        segment_merge_overhead_mb=round(merge_overhead, 1),
        snapshot_space_mb=round(snapshot_space, 1),
        total_disk_mb=round(total_mb, 1),
        total_disk_gb=total_gb,
        disk_type=disk_type,
    )


# ---------------------------------------------------------------------------
# Stage 7: Size Compute
# ---------------------------------------------------------------------------

CPU_BASE_MS = {
    # (dim_bucket, quantization) -> ms at ef=64
    ("< 512",     "binary"):  0.3,
    ("< 512",     "scalar"):  0.5,
    ("< 512",     "none"):    1.0,
    ("512-1024",  "binary"):  0.5,
    ("512-1024",  "scalar"):  1.0,
    ("512-1024",  "none"):    2.0,
    ("1024-2048", "binary"):  0.8,
    ("1024-2048", "scalar"):  1.5,
    ("1024-2048", "none"):    3.0,
    ("2048-4096", "binary"):  1.0,
    ("2048-4096", "scalar"):  2.0,
    ("2048-4096", "none"):    5.0,
}


def _dim_bucket(dims: int) -> str:
    if dims < 512:
        return "< 512"
    elif dims <= 1024:
        return "512-1024"
    elif dims <= 2048:
        return "1024-2048"
    else:
        return "2048-4096"


def size_compute(dimensions: int, quantization: str, hnsw: HNSWParams,
                 latency: LatencyAssessment, peak_qps: int,
                 write_pattern: str, peak_write_rate: int) -> ComputeSizing:
    # 7a: Per-query CPU time
    bucket = _dim_bucket(dimensions)
    base_ms = CPU_BASE_MS[(bucket, quantization)]

    ef_adj = hnsw.hnsw_ef / 64
    per_query_base = base_ms * ef_adj

    # Rescoring cost
    rescore_ms = 0
    if hnsw.rescore_candidates and "mmap" in latency.full_vector_placement:
        rescore_ms = hnsw.rescore_candidates * 0.01
    elif hnsw.rescore_candidates:
        rescore_ms = 0.1  # floor for in-RAM rescore

    # mmap overhead (only if no quantization AND full vectors on mmap)
    mmap_overhead = 0
    if quantization == "none" and "mmap" in latency.full_vector_placement:
        mmap_overhead = 1.0

    per_query_ms = per_query_base + rescore_ms + mmap_overhead

    # 7b: Cores for QPS
    cores = peak_qps * (per_query_ms / 1000)

    # Headroom
    tight_streaming = (latency.latency_tier == "TIGHT" and write_pattern == "streaming")
    tight_batch = (latency.latency_tier == "TIGHT" and write_pattern != "streaming")
    if tight_streaming:
        headroom_pct = 100
    elif tight_batch:
        headroom_pct = 50
    else:
        headroom_pct = 30

    headroom_cores = cores * (headroom_pct / 100)

    # 7c: Write cores
    if write_pattern == "streaming":
        idx_time = INDEXING_TIME_MS.get(hnsw.m, 3.0)  # fallback 3.0ms
        write_cores = peak_write_rate * (idx_time / 1000)
    else:
        write_cores = 0

    # 7d: Total
    total = cores + headroom_cores + write_cores
    total_vcpus = max(math.ceil(total), 4)  # minimum 4

    return ComputeSizing(
        per_query_base_ms=round(per_query_base, 2),
        ef_adjustment=round(ef_adj, 2),
        rescore_time_ms=round(rescore_ms, 2),
        mmap_overhead_ms=mmap_overhead,
        per_query_time_ms=round(per_query_ms, 2),
        cores_for_queries=round(cores, 2),
        headroom_pct=headroom_pct,
        headroom_cores=round(headroom_cores, 2),
        write_cores=round(write_cores, 2),
        total_vcpus=total_vcpus,
    )


# ---------------------------------------------------------------------------
# Stage 7e + Stage 9: Topology & Instance Selection
# ---------------------------------------------------------------------------

def select_topology_and_instance(compute: ComputeSizing,
                                 memory: MemorySizing,
                                 disk: DiskSizing) -> tuple[TopologyDecision, InstanceSelection]:
    vcpus = compute.total_vcpus
    ram_gb = memory.total_ram_rounded_gb

    # Topology: single node by default
    topo = TopologyDecision(
        nodes=1, shards=1, replicas=1,
        note="Single node. Present HA option (replication_factor=2) to customer."
    )

    # Instance family selection
    if ram_gb > vcpus * 4:
        family = "R"
        constraint = "memory"
        family_label = "R-series (memory-optimized)"
    elif vcpus * 4 > ram_gb:
        family = "C"
        constraint = "compute"
        family_label = "C-series (compute-optimized)"
    else:
        family = "M"
        constraint = "balanced"
        family_label = "M-series (general-purpose)"

    # Find smallest fitting instance
    candidates = [i for i in INSTANCES if i[3] == family]
    candidates.sort(key=lambda x: x[1])  # sort by vcpus

    selected = None
    overprovision_note = ""

    for inst in candidates:
        name, inst_vcpus, inst_ram, _, hourly = inst
        if inst_vcpus >= vcpus and inst_ram >= ram_gb:
            selected = inst
            break

    # Overprovision check: if selected is >1.5x vcpus, try one smaller
    if selected and selected[1] > vcpus * 1.5:
        idx = candidates.index(selected)
        if idx > 0:
            smaller = candidates[idx - 1]
            if smaller[2] >= ram_gb:  # RAM still fits
                shortfall = vcpus - smaller[1]
                overprovision_note = (
                    f"Using {smaller[0]} ({smaller[1]} vCPU) instead of "
                    f"{selected[0]} ({selected[1]} vCPU) to avoid "
                    f"{round(selected[1]/vcpus*100-100)}% overprovision. "
                    f"{shortfall} vCPU shortfall is within headroom margin."
                )
                selected = smaller

    if not selected:
        # Fallback: pick largest available
        selected = candidates[-1] if candidates else INSTANCES[-1]
        overprovision_note = "No exact fit found; using largest available instance."

    name, inst_vcpus, inst_ram, _, hourly = selected
    monthly = hourly * 730
    reserved = monthly * 0.65
    disk_cost = disk.total_disk_gb * 0.08

    inst_sel = InstanceSelection(
        instance_family=family_label,
        dominant_constraint=constraint,
        instance_type=name,
        instance_vcpus=inst_vcpus,
        instance_ram_gb=inst_ram,
        cost_on_demand_mo=round(monthly, 0),
        cost_reserved_mo=round(reserved, 0),
        disk_cost_mo=round(disk_cost, 2),
        total_on_demand_mo=round(monthly + disk_cost, 0),
        total_reserved_mo=round(reserved + disk_cost, 0),
        overprovision_note=overprovision_note,
    )

    return topo, inst_sel


# ---------------------------------------------------------------------------
# Future optimizations
# ---------------------------------------------------------------------------

def suggest_optimizations(classification: EmbeddingClassification,
                          recall: RecallRegime, hnsw: HNSWParams,
                          latency: LatencyAssessment, dimensions: int,
                          top_k: int, write_pattern: str,
                          peak_qps: int) -> list[str]:
    opts = []

    # HA recommendation (always)
    opts.append(
        "HA: add replication_factor=2 for failover (doubles infra cost). "
        "Recommended if P99 SLA is contractual."
    )

    # Binary quantization for Class A
    if classification.embedding_class == "A" and recall.quantization == "scalar":
        if top_k < 20:
            opts.append(
                "Binary quantization: 32x compression vs scalar's 4x. "
                "Viable for top-k < 20; benchmark against recall target."
            )
        else:
            opts.append(
                "Binary quantization: expert opinion split at top-k >= 50. "
                "Test empirically; may not meet recall at high top-k."
            )

    # Matryoshka
    if classification.embedding_class == "A" and dimensions >= 1536:
        opts.append(
            f"Matryoshka dimension reduction: test {dimensions} -> {dimensions // 2} dims "
            f"against recall target. Could halve vector memory."
        )

    # Collection aliasing for batch
    if write_pattern == "batch":
        opts.append(
            "Collection aliasing: build new collection per batch, swap alias atomically "
            "for zero-downtime reindexing."
        )

    # Bursty QPS
    if write_pattern == "batch":
        opts.append(
            "Autoscaling: scale replicas up during peak QPS window, down during off-peak "
            "to reduce idle infrastructure cost."
        )

    # Scalar quantization test for Class C
    if classification.embedding_class == "C" and recall.quantization == "none":
        opts.append(
            "Scalar quantization: guide defaults to none for Class C + high recall, "
            "but test empirically — may work and save significant RAM."
        )

    return opts


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def size_cluster(config: dict) -> ArchitectureSummary:
    """Run the full sizing pipeline on a config dict."""

    # Extract inputs
    dataset_name = config.get("dataset_name", "unnamed")
    embedding_type = config["embedding_type"]
    dimensions = config["dimensions"]
    num_vectors = config["num_vectors"]
    peak_qps = config["peak_qps"]
    write_pattern = config.get("write_pattern", "streaming")
    peak_write_rate = config.get("peak_write_rate", 0)
    batch_size = config.get("batch_size", 0)
    p99_latency_ms = config["p99_latency_ms"]
    recall_target = config["recall_target"]
    top_k = config["top_k"]

    # Stage 1
    classification = classify_embeddings(embedding_type, dimensions)

    # Stage 2
    recall_regime = determine_recall_regime(recall_target, classification.embedding_class)

    # Stage 3
    hnsw = set_hnsw_params(recall_regime.regime, top_k,
                           recall_regime.quantization, write_pattern)

    # Stage 4
    latency = assess_latency(p99_latency_ms, recall_regime.quantization)

    # Stage 5
    memory = size_memory(num_vectors, dimensions, recall_regime.quantization,
                         hnsw, latency)

    # Stage 6
    disk = size_disk(num_vectors, dimensions, memory, write_pattern,
                     peak_write_rate, batch_size, latency)

    # Stage 7
    compute = size_compute(dimensions, recall_regime.quantization, hnsw,
                           latency, peak_qps, write_pattern, peak_write_rate)

    # Stage 7e + 9
    topology, instance = select_topology_and_instance(compute, memory, disk)

    # Future optimizations
    future = suggest_optimizations(classification, recall_regime, hnsw,
                                   latency, dimensions, top_k,
                                   write_pattern, peak_qps)

    return ArchitectureSummary(
        dataset_name=dataset_name,
        num_vectors=num_vectors,
        dimensions=dimensions,
        peak_qps=peak_qps,
        write_pattern=write_pattern,
        peak_write_rate=peak_write_rate,
        batch_size=batch_size,
        p99_latency_ms=p99_latency_ms,
        recall_target=recall_target,
        top_k=top_k,
        classification=classification,
        recall_regime=recall_regime,
        hnsw=hnsw,
        latency=latency,
        memory=memory,
        disk=disk,
        compute=compute,
        topology=topology,
        instance=instance,
        future_optimizations=future,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_summary(s: ArchitectureSummary) -> str:
    lines = []
    a = lines.append
    a("=" * 64)
    a(f"  ARCHITECTURE SUMMARY — {s.dataset_name}")
    a("=" * 64)
    a("")
    a(f"  Dataset:         {s.dataset_name}")
    a(f"  Vectors:         {s.num_vectors:,}")
    a(f"  Dimensions:      {s.dimensions}")
    a(f"  Embedding Class: {s.classification.embedding_class} ({s.classification.class_label})")
    a(f"  Distance Metric: {s.classification.distance_metric} (Qdrant: {s.classification.qdrant_distance})")
    a("")
    a("  SLA Requirements")
    a("  " + "-" * 30)
    a(f"  Target QPS:      {s.peak_qps:,}")
    a(f"  P99 Latency:     {s.p99_latency_ms}ms")
    a(f"  Recall Target:   {s.recall_target:.0%}")
    a(f"  Top-k:           {s.top_k}")
    a(f"  Write Pattern:   {s.write_pattern} ({s.peak_write_rate}/s peak)" if s.write_pattern == "streaming"
      else f"  Write Pattern:   {s.write_pattern} (batch size: {s.batch_size:,})")
    a("")
    a("  Regime Classification")
    a("  " + "-" * 30)
    a(f"  Recall Regime:   {s.recall_regime.regime}")
    a(f"  Latency Tier:    {s.latency.latency_tier}")
    a("")
    a("  Quantization Strategy")
    a("  " + "-" * 30)
    a(f"  Method:          {s.recall_regime.quantization}")
    a(f"  Oversampling:    {s.hnsw.oversampling or 'N/A'}")
    a(f"  Rescore:         {'ON' if s.recall_regime.rescore else 'N/A'}")
    a("")
    a("  HNSW Parameters")
    a("  " + "-" * 30)
    a(f"  m:               {s.hnsw.m}")
    a(f"  ef_construct:    {s.hnsw.ef_construct}")
    a(f"  hnsw_ef:         {s.hnsw.hnsw_ef}  ({s.hnsw.hnsw_ef_note})")
    a("")
    a("  Storage Placement")
    a("  " + "-" * 30)
    a(f"  HNSW index:      {s.latency.hnsw_placement}")
    a(f"  Quantized vecs:  {s.latency.quantized_placement}")
    a(f"  Full vectors:    {s.latency.full_vector_placement}"
      + (" (promoted: no-quant rule)" if s.latency.no_quant_promotion else ""))
    a("")
    a("  Memory Calculation")
    a("  " + "-" * 30)
    if s.memory.quantized_memory_mb > 0:
        a(f"  Quantized vecs:  {s.memory.quantized_memory_mb:,.0f} MB")
    a(f"  Full vectors:    {s.memory.full_vector_memory_mb:,.0f} MB")
    a(f"  HNSW index:      {s.memory.hnsw_memory_mb:,.0f} MB")
    a(f"  Page cache:      {s.memory.page_cache_mb:,.0f} MB")
    a(f"  Process overhead: {s.memory.process_overhead_mb:,.0f} MB")
    a(f"  Merge headroom:  {s.memory.merge_headroom_mb:,.0f} MB")
    a(f"  Total RAM:       {s.memory.total_ram_mb:,.0f} MB → {s.memory.total_ram_rounded_gb} GB")
    a("")
    a("  Disk Calculation")
    a("  " + "-" * 30)
    a(f"  Base disk:       {s.disk.base_disk_mb:,.0f} MB")
    a(f"  WAL space:       {s.disk.wal_space_mb:,.0f} MB")
    a(f"  Merge overhead:  {s.disk.segment_merge_overhead_mb:,.0f} MB")
    a(f"  Snapshot space:  {s.disk.snapshot_space_mb:,.0f} MB")
    a(f"  Total disk:      {s.disk.total_disk_gb} GB")
    a(f"  Disk type:       {s.disk.disk_type}")
    a("")
    a("  Compute Calculation")
    a("  " + "-" * 30)
    a(f"  Per-query time:  {s.compute.per_query_time_ms:.2f}ms")
    a(f"    base:          {s.compute.per_query_base_ms:.2f}ms (ef adjustment: {s.compute.ef_adjustment:.1f}x)")
    a(f"    rescore:       {s.compute.rescore_time_ms:.2f}ms")
    a(f"    mmap overhead: {s.compute.mmap_overhead_ms:.1f}ms")
    a(f"  Cores for QPS:   {s.compute.cores_for_queries:.1f}")
    a(f"  Headroom ({s.compute.headroom_pct}%):  {s.compute.headroom_cores:.1f}")
    a(f"  Write cores:     {s.compute.write_cores:.1f}")
    a(f"  Total vCPUs:     {s.compute.total_vcpus}")
    a("")
    a("  Topology")
    a("  " + "-" * 30)
    a(f"  Nodes: {s.topology.nodes}  Shards: {s.topology.shards}  Replicas: {s.topology.replicas}")
    a(f"  {s.topology.note}")
    a("")
    a("  Instance & Cost")
    a("  " + "-" * 30)
    a(f"  Instance:        {s.instance.instance_type} ({s.instance.instance_vcpus} vCPU, {s.instance.instance_ram_gb} GB)")
    a(f"  Constraint:      {s.instance.dominant_constraint} → {s.instance.instance_family}")
    if s.instance.overprovision_note:
        a(f"  Note:            {s.instance.overprovision_note}")
    a(f"  Cost (on-demand): ${s.instance.total_on_demand_mo:,.0f}/mo")
    a(f"  Cost (reserved):  ${s.instance.total_reserved_mo:,.0f}/mo")
    a("")
    a("  Future Optimizations")
    a("  " + "-" * 30)
    for i, opt in enumerate(s.future_optimizations, 1):
        a(f"  {i}. {opt}")
    a("")
    a("=" * 64)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

EMBEDDING_TYPES = {
    "1": ("neural_text",    "Neural text model (OpenAI, Cohere, BGE, E5, etc.)"),
    "2": ("neural_vision",  "Neural vision model (CLIP, DINOv2, etc.)"),
    "3": ("classical_cv",   "Classical CV features (GIST, SIFT, HOG, etc.)"),
    "4": ("other",          "Other / Unknown"),
}

WRITE_PATTERNS = {
    "1": ("streaming", "Streaming (continuous writes, concurrent with reads)"),
    "2": ("batch",     "Batch (periodic bulk loads, no concurrent reads during writes)"),
    "3": ("static",    "Rare / Static (load once, mostly read)"),
}


def _ask_choice(prompt: str, choices: dict[str, tuple], default_key: str) -> tuple:
    """Prompt user to pick from numbered choices. Re-prompts on invalid input."""
    while True:
        raw = input(prompt).strip()
        if raw in choices:
            return choices[raw]
        valid = ", ".join(choices.keys())
        print(f"    Invalid choice '{raw}'. Please enter one of: {valid}")


def _ask_int(prompt: str, min_val: int = 1, max_val: int = 10_000_000_000) -> int:
    """Prompt for a positive integer. Re-prompts on invalid input."""
    while True:
        raw = input(prompt).strip()
        try:
            val = int(raw)
        except ValueError:
            print(f"    '{raw}' is not a valid integer. Please enter a number.")
            continue
        if val < min_val:
            print(f"    Value must be at least {min_val}. Got {val}.")
            continue
        if val > max_val:
            print(f"    Value must be at most {max_val:,}. Got {val:,}.")
            continue
        return val


def _ask_float(prompt: str, min_val: float = 0, max_val: float = 100) -> float:
    """Prompt for a number. Re-prompts on invalid input."""
    while True:
        raw = input(prompt).strip()
        try:
            val = float(raw)
        except ValueError:
            print(f"    '{raw}' is not a valid number. Please enter a number (e.g., 95 or 0.95).")
            continue
        if val < min_val:
            print(f"    Value must be at least {min_val}. Got {val}.")
            continue
        if val > max_val:
            print(f"    Value must be at most {max_val}. Got {val}.")
            continue
        return val


def interactive():
    print("\n" + "=" * 64)
    print("  Qdrant Cluster Sizer — Interactive Mode")
    print("=" * 64 + "\n")

    dataset_name = input("  Dataset name: ").strip() or "unnamed"

    print("\n  Embedding type:")
    for k, (_, label) in EMBEDDING_TYPES.items():
        print(f"    {k}. {label}")
    _, embedding_type = _ask_choice("  Choice [1-4]: ", {
        k: (k, v[0]) for k, v in EMBEDDING_TYPES.items()
    }, "4")

    dimensions = _ask_int("  Dimensions: ", min_val=1, max_val=65536)
    num_vectors = _ask_int("  Number of vectors: ", min_val=1)
    peak_qps = _ask_int("  Peak QPS: ", min_val=1)

    print("\n  Write pattern:")
    for k, (_, label) in WRITE_PATTERNS.items():
        print(f"    {k}. {label}")
    _, write_pattern = _ask_choice("  Choice [1-3]: ", {
        k: (k, v[0]) for k, v in WRITE_PATTERNS.items()
    }, "1")

    peak_write_rate = 0
    batch_size = 0
    if write_pattern == "streaming":
        peak_write_rate = _ask_int("  Peak write rate (vectors/s): ", min_val=0)
    elif write_pattern == "batch":
        batch_size = _ask_int("  Batch size (vectors per load): ", min_val=1)

    p99_latency_ms = _ask_int("  P99 latency SLA (ms): ", min_val=1, max_val=60000)
    recall_pct = _ask_float("  Recall target (e.g., 95 for 95%): ", min_val=0.01, max_val=100)
    recall_target = recall_pct / 100 if recall_pct > 1 else recall_pct
    top_k = _ask_int("  Top-k: ", min_val=1, max_val=10000)

    config = {
        "dataset_name": dataset_name,
        "embedding_type": embedding_type,
        "dimensions": dimensions,
        "num_vectors": num_vectors,
        "peak_qps": peak_qps,
        "write_pattern": write_pattern,
        "peak_write_rate": peak_write_rate,
        "batch_size": batch_size,
        "p99_latency_ms": p99_latency_ms,
        "recall_target": recall_target,
        "top_k": top_k,
    }

    result = size_cluster(config)
    print("\n" + format_summary(result))
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) == 1:
        interactive()
    elif sys.argv[1] == "--json":
        config = json.loads(sys.argv[2])
        result = size_cluster(config)
        print(format_summary(result))
    elif sys.argv[1] == "--file":
        with open(sys.argv[2]) as f:
            configs = json.load(f)
        # Support single config or list of configs
        if isinstance(configs, list):
            for config in configs:
                result = size_cluster(config)
                print(format_summary(result))
                print()
        else:
            result = size_cluster(configs)
            print(format_summary(result))
    elif sys.argv[1] == "--json-out":
        config = json.loads(sys.argv[2])
        result = size_cluster(config)
        print(json.dumps(asdict(result), indent=2))
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
