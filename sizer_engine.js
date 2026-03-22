/**
 * Qdrant Cluster Sizer Engine — Single source of truth.
 *
 * This module contains ALL sizing logic and constants. It is consumed by:
 *   - index.html (browser wizard)
 *   - qdrant_sizer_cli.js (Node CLI)
 *
 * No logic should be duplicated outside this file.
 */

// ---------------------------------------------------------------------------
// Constants — All decision tables live here
// ---------------------------------------------------------------------------

export const QUANT_MATRIX = {
  'A-RELAXED': 'scalar', 'A-MODERATE': 'scalar', 'A-STRICT': 'scalar',
  'B-RELAXED': 'scalar', 'B-MODERATE': 'scalar', 'B-STRICT': 'none',
  'C-RELAXED': 'scalar', 'C-MODERATE': 'none',   'C-STRICT': 'none',
};

export const M_BASE = { RELAXED: 16, MODERATE: 20, STRICT: 32 };
export const M_UP   = { RELAXED: 20, MODERATE: 28, STRICT: 32 };
export const M_DOWN = { RELAXED: 16, MODERATE: 16, STRICT: 24 };

export const EF_CONSTRUCT_BASE = { RELAXED: 200, MODERATE: 256, STRICT: 400 };
export const EF_CONSTRUCT_BATCH_STRICT = 400;

export const EF_MULTIPLIER = { RELAXED: 2, MODERATE: 3, STRICT: 6 };
export const EF_FLOOR      = { RELAXED: 64, MODERATE: 64, STRICT: 128 };

export const OVERSAMPLING = {
  'scalar-RELAXED': 2.0, 'scalar-MODERATE': 2.5, 'scalar-STRICT': 4.0,
  'binary-RELAXED': 3.0, 'binary-MODERATE': 4.0,
};

export const INDEXING_TIME_MS = { 12: 1.0, 16: 1.5, 20: 2.5, 24: 3.0, 28: 3.5, 32: 4.0 };

export const CPU_BASE_MS = {
  '<512-binary': 0.3,    '<512-scalar': 0.5,    '<512-none': 1.0,
  '512-1024-binary': 0.5, '512-1024-scalar': 1.0, '512-1024-none': 2.0,
  '1024-2048-binary': 0.8,'1024-2048-scalar': 1.5,'1024-2048-none': 3.0,
  '2048-4096-binary': 1.0,'2048-4096-scalar': 2.0,'2048-4096-none': 5.0,
};

export const INSTANCES = [
  { name: 'c6i.xlarge',    vcpus: 4,  ram: 8,  family: 'C', hourly: 0.170 },
  { name: 'c6i.2xlarge',   vcpus: 8,  ram: 16, family: 'C', hourly: 0.340 },
  { name: 'c6i.4xlarge',   vcpus: 16, ram: 32, family: 'C', hourly: 0.680 },
  { name: 'c6i.8xlarge',   vcpus: 32, ram: 64, family: 'C', hourly: 1.360 },
  { name: 'c6i.12xlarge',  vcpus: 48, ram: 96, family: 'C', hourly: 2.040 },
  { name: 'm6i.xlarge',    vcpus: 4,  ram: 16, family: 'M', hourly: 0.192 },
  { name: 'm6i.2xlarge',   vcpus: 8,  ram: 32, family: 'M', hourly: 0.384 },
  { name: 'm6i.4xlarge',   vcpus: 16, ram: 64, family: 'M', hourly: 0.768 },
  { name: 'r6i.xlarge',    vcpus: 4,  ram: 32, family: 'R', hourly: 0.252 },
  { name: 'r6i.2xlarge',   vcpus: 8,  ram: 64, family: 'R', hourly: 0.504 },
];

export const RAM_TIERS = [4, 8, 16, 32, 64, 128, 256];

// ---------------------------------------------------------------------------
// Stage 1: Classify Embeddings
// ---------------------------------------------------------------------------

export function classifyEmbeddings(embeddingType, dimensions) {
  if (embeddingType === 'neural_text') {
    const cls = dimensions >= 1024 ? 'A' : 'B';
    const label = cls === 'A' ? 'Binary-friendly neural' : 'Scalar-friendly neural';
    return { cls, label, metric: 'Cosine', qdrant: 'Cosine' };
  }
  if (embeddingType === 'neural_vision')
    return { cls: 'B', label: 'Scalar-friendly neural', metric: 'Cosine', qdrant: 'Cosine' };
  if (embeddingType === 'classical_cv')
    return { cls: 'C', label: 'Quantization-resistant', metric: 'Euclidean (L2)', qdrant: 'Euclid' };
  return { cls: 'B', label: 'Scalar-friendly neural (default)', metric: 'Cosine', qdrant: 'Cosine' };
}

// ---------------------------------------------------------------------------
// Stage 2: Recall Regime
// ---------------------------------------------------------------------------

export function recallRegime(recallTarget, embeddingClass) {
  let regime;
  if (recallTarget <= 0.95) regime = 'RELAXED';
  else if (recallTarget <= 0.98) regime = 'MODERATE';
  else regime = 'STRICT';

  const quant = QUANT_MATRIX[`${embeddingClass}-${regime}`];
  return { regime, quant, rescore: quant !== 'none' };
}

// ---------------------------------------------------------------------------
// Stage 3: HNSW Parameters
// ---------------------------------------------------------------------------

export function hnswParams(regime, topK, quantization, writePattern) {
  // 3a: m with top-k adjustment
  let m;
  if (topK >= 50) m = M_UP[regime];
  else if (topK <= 20) m = M_DOWN[regime];
  else m = M_BASE[regime];

  // 3b: ef_construct
  let efConstruct;
  if (writePattern === 'batch' && regime === 'STRICT') efConstruct = EF_CONSTRUCT_BATCH_STRICT;
  else efConstruct = EF_CONSTRUCT_BASE[regime];

  // 3c: hnsw_ef
  const calculatedEf = EF_MULTIPLIER[regime] * topK;
  const floor = EF_FLOOR[regime];
  const hnswEf = Math.max(calculatedEf, floor);
  const efNote = hnswEf > calculatedEf
    ? `floor-adjusted from ${calculatedEf} to ${hnswEf}`
    : `${EF_MULTIPLIER[regime]}x top_k`;

  // 3d: oversampling
  let oversampling = null;
  let rescoreCandidates = null;
  if (quantization !== 'none') {
    oversampling = OVERSAMPLING[`${quantization}-${regime}`] || null;
    rescoreCandidates = oversampling ? Math.round(oversampling * topK) : null;
  }

  return { m, efConstruct, hnswEf, efNote, oversampling, rescoreCandidates };
}

// ---------------------------------------------------------------------------
// Stage 4: Latency Assessment
// ---------------------------------------------------------------------------

export function latencyAssess(p99Ms, quantization) {
  let tier;
  if (p99Ms < 20) tier = 'ULTRA-TIGHT';
  else if (p99Ms <= 99) tier = 'TIGHT';
  else if (p99Ms <= 500) tier = 'MODERATE';
  else tier = 'RELAXED';

  const noQuant = quantization === 'none';
  const placements = {
    'ULTRA-TIGHT': ['RAM', 'RAM', 'RAM'],
    'TIGHT':       ['RAM', 'RAM', 'mmap (NVMe)'],
    'MODERATE':    ['RAM', 'RAM or mmap', 'mmap (SSD)'],
    'RELAXED':     ['RAM', 'mmap', 'mmap (SSD)'],
  };

  let [hnswPlacement, quantPlacement, fullPlacement] = placements[tier];
  let promoted = false;

  if (noQuant) {
    quantPlacement = 'N/A';
    const promoMap = {
      'ULTRA-TIGHT': 'RAM', 'TIGHT': 'RAM', 'MODERATE': 'RAM', 'RELAXED': 'mmap (SSD)',
    };
    fullPlacement = promoMap[tier];
    promoted = tier === 'MODERATE';
  }

  return { tier, hnswPlacement, quantPlacement, fullPlacement, promoted };
}

// ---------------------------------------------------------------------------
// Stage 5: Memory Sizing
// ---------------------------------------------------------------------------

export function sizeMemory(numVectors, dimensions, quantization, hnsw, latency) {
  const fullMB = numVectors * dimensions * 4 / (1024 * 1024);
  let quantMB = 0;
  if (quantization === 'scalar') quantMB = numVectors * dimensions / (1024 * 1024);
  else if (quantization === 'binary') quantMB = numVectors * dimensions / 8 / (1024 * 1024);

  const hnswMB = numVectors * hnsw.m * 2 * 8 * 1.1 / (1024 * 1024);

  let ramComponents = hnswMB;
  if (latency.quantPlacement === 'RAM') ramComponents += quantMB;
  if (latency.fullPlacement === 'RAM') ramComponents += fullMB;

  let pageCache = 0;
  if (latency.fullPlacement.includes('mmap') && quantization !== 'none')
    pageCache = 0.10 * fullMB;
  else if (latency.fullPlacement.includes('mmap') && quantization === 'none')
    pageCache = 0.30 * fullMB;
  if (latency.quantPlacement && latency.quantPlacement.includes('mmap'))
    pageCache += 0.50 * quantMB;

  const processOverhead = 500;
  const mergeHeadroom = ramComponents * 0.10;
  const totalMB = ramComponents + pageCache + processOverhead + mergeHeadroom;

  let roundedGB = 256;
  for (const tier of RAM_TIERS) {
    if (tier * 1024 >= totalMB) { roundedGB = tier; break; }
  }

  return {
    quantMB: Math.round(quantMB),
    fullMB: Math.round(fullMB),
    hnswMB: Math.round(hnswMB),
    pageCacheMB: Math.round(pageCache),
    processOverheadMB: processOverhead,
    mergeHeadroomMB: Math.round(mergeHeadroom),
    totalMB: Math.round(totalMB),
    roundedGB,
  };
}

// ---------------------------------------------------------------------------
// Stage 6: Disk Sizing
// ---------------------------------------------------------------------------

export function sizeDisk(numVectors, dimensions, memory, writePattern, peakWriteRate, batchSize, latency) {
  const baseDisk = memory.fullMB + memory.quantMB + memory.hnswMB;

  let walMB;
  if (writePattern === 'streaming') walMB = 2048;
  else if (writePattern === 'batch') walMB = Math.max(batchSize * dimensions * 4 * 2 / (1024 * 1024), 1024);
  else walMB = 1024;

  const mergeOverhead = baseDisk * 0.50;
  const snapshotSpace = baseDisk * 1.0;
  const totalMB = baseDisk + walMB + mergeOverhead + snapshotSpace;
  const totalGB = Math.max(Math.ceil(totalMB / 1024), 30);

  const mmapInQuery = latency.fullPlacement.includes('mmap') ||
    (latency.quantPlacement && latency.quantPlacement.includes('mmap'));
  let diskType;
  if (mmapInQuery && latency.tier === 'TIGHT') diskType = 'NVMe SSD (gp3 w/ provisioned IOPS)';
  else diskType = 'Standard SSD (gp3)';

  return {
    baseMB: Math.round(baseDisk),
    walMB: Math.round(walMB),
    mergeOverheadMB: Math.round(mergeOverhead),
    snapshotMB: Math.round(snapshotSpace),
    totalGB,
    diskType,
  };
}

// ---------------------------------------------------------------------------
// Stage 7: Compute Sizing
// ---------------------------------------------------------------------------

function dimBucket(dims) {
  if (dims < 512) return '<512';
  if (dims <= 1024) return '512-1024';
  if (dims <= 2048) return '1024-2048';
  return '2048-4096';
}

export function sizeCompute(dimensions, quantization, hnsw, latency, peakQPS, writePattern, peakWriteRate) {
  const baseMs = CPU_BASE_MS[`${dimBucket(dimensions)}-${quantization}`];
  const efAdjustment = hnsw.hnswEf / 64;
  const perQueryBase = baseMs * efAdjustment;

  let rescoreMs = 0;
  if (hnsw.rescoreCandidates && latency.fullPlacement.includes('mmap'))
    rescoreMs = hnsw.rescoreCandidates * 0.01;
  else if (hnsw.rescoreCandidates)
    rescoreMs = 0.1;

  let mmapOverhead = 0;
  if (quantization === 'none' && latency.fullPlacement.includes('mmap'))
    mmapOverhead = 1.0;

  const perQueryMs = perQueryBase + rescoreMs + mmapOverhead;
  const coresForQueries = peakQPS * (perQueryMs / 1000);

  const tightStreaming = latency.tier === 'TIGHT' && writePattern === 'streaming';
  const tightBatch = latency.tier === 'TIGHT' && writePattern !== 'streaming';
  const headroomPct = tightStreaming ? 100 : tightBatch ? 50 : 30;
  const headroomCores = coresForQueries * (headroomPct / 100);

  let writeCores = 0;
  if (writePattern === 'streaming') {
    const idxTime = INDEXING_TIME_MS[hnsw.m] || 3.0;
    writeCores = peakWriteRate * (idxTime / 1000);
  }

  const totalVcpus = Math.max(Math.ceil(coresForQueries + headroomCores + writeCores), 4);

  return {
    perQueryBaseMs: +perQueryBase.toFixed(2),
    efAdjustment: +efAdjustment.toFixed(1),
    rescoreMs: +rescoreMs.toFixed(2),
    mmapOverheadMs: mmapOverhead,
    perQueryMs: +perQueryMs.toFixed(2),
    coresForQueries: +coresForQueries.toFixed(1),
    headroomPct,
    headroomCores: +headroomCores.toFixed(1),
    writeCores: +writeCores.toFixed(1),
    totalVcpus,
  };
}

// ---------------------------------------------------------------------------
// Stage 9: Instance Selection
// ---------------------------------------------------------------------------

export function selectInstance(compute, memory, disk) {
  const vcpus = compute.totalVcpus;
  const ramGB = memory.roundedGB;

  let family, constraint, familyLabel;
  if (ramGB > vcpus * 4) {
    family = 'R'; constraint = 'memory'; familyLabel = 'R-series (memory-optimized)';
  } else if (vcpus * 4 > ramGB) {
    family = 'C'; constraint = 'compute'; familyLabel = 'C-series (compute-optimized)';
  } else {
    family = 'M'; constraint = 'balanced'; familyLabel = 'M-series (general-purpose)';
  }

  const candidates = INSTANCES.filter(i => i.family === family).sort((a, b) => a.vcpus - b.vcpus);
  let selected = candidates.find(i => i.vcpus >= vcpus && i.ram >= ramGB);
  let overprovisionNote = '';

  if (selected && selected.vcpus > vcpus * 1.5) {
    const idx = candidates.indexOf(selected);
    if (idx > 0 && candidates[idx - 1].ram >= ramGB) {
      const smaller = candidates[idx - 1];
      overprovisionNote =
        `Using ${smaller.name} (${smaller.vcpus} vCPU) instead of ` +
        `${selected.name} (${selected.vcpus} vCPU) to avoid ` +
        `${Math.round(selected.vcpus / vcpus * 100 - 100)}% overprovision. ` +
        `${vcpus - smaller.vcpus} vCPU shortfall is within headroom.`;
      selected = smaller;
    }
  }

  if (!selected) {
    selected = candidates[candidates.length - 1] || INSTANCES[INSTANCES.length - 1];
    overprovisionNote = 'No exact fit; using largest available.';
  }

  const monthly = selected.hourly * 730;
  const reserved = monthly * 0.65;
  const diskCost = disk.totalGB * 0.08;

  return {
    instanceFamily: familyLabel,
    dominantConstraint: constraint,
    instanceType: selected.name,
    instanceVcpus: selected.vcpus,
    instanceRamGB: selected.ram,
    costOnDemandMo: Math.round(monthly + diskCost),
    costReservedMo: Math.round(reserved + diskCost),
    overprovisionNote,
  };
}

// ---------------------------------------------------------------------------
// Future Optimizations
// ---------------------------------------------------------------------------

export function suggestOptimizations(classification, recall, hnsw, latency, dimensions, topK, writePattern) {
  const opts = [
    'HA: add replication_factor=2 for failover (doubles infra cost). Recommended if P99 SLA is contractual.',
  ];

  if (classification.cls === 'A' && recall.quant === 'scalar') {
    opts.push(topK < 20
      ? 'Binary quantization: 32x compression. Viable for top-k < 20; benchmark against recall target.'
      : 'Binary quantization: expert opinion split at top-k >= 50. Test empirically.');
  }

  if (classification.cls === 'A' && dimensions >= 1536)
    opts.push(`Matryoshka dimension reduction: test ${dimensions} -> ${Math.floor(dimensions / 2)} dims against recall target.`);

  if (writePattern === 'batch') {
    opts.push('Collection aliasing: build new collection per batch, swap alias atomically.');
    opts.push('Autoscaling: scale replicas up during peak QPS window, down during off-peak.');
  }

  if (classification.cls === 'C' && recall.quant === 'none')
    opts.push('Scalar quantization: guide defaults to none for Class C + high recall. Test empirically.');

  return opts;
}

// ---------------------------------------------------------------------------
// Full Pipeline
// ---------------------------------------------------------------------------

export function sizeCluster(config) {
  const {
    datasetName = 'unnamed',
    embeddingType,
    dimensions,
    numVectors,
    peakQPS,
    writePattern = 'streaming',
    peakWriteRate = 0,
    batchSize = 0,
    p99LatencyMs,
    recallTarget,
    topK,
  } = config;

  const classification = classifyEmbeddings(embeddingType, dimensions);
  const recall = recallRegime(recallTarget, classification.cls);
  const hnsw = hnswParams(recall.regime, topK, recall.quant, writePattern);
  const latency = latencyAssess(p99LatencyMs, recall.quant);
  const memory = sizeMemory(numVectors, dimensions, recall.quant, hnsw, latency);
  const disk = sizeDisk(numVectors, dimensions, memory, writePattern, peakWriteRate, batchSize, latency);
  const compute = sizeCompute(dimensions, recall.quant, hnsw, latency, peakQPS, writePattern, peakWriteRate);
  const instance = selectInstance(compute, memory, disk);
  const optimizations = suggestOptimizations(classification, recall, hnsw, latency, dimensions, topK, writePattern);

  return {
    // Inputs
    datasetName, dimensions, numVectors, peakQPS, writePattern,
    peakWriteRate, batchSize, p99LatencyMs, recallTarget, topK,
    // Stage outputs
    classification, recall, hnsw, latency, memory, disk, compute, instance, optimizations,
    // Topology (always single-node per design philosophy)
    topology: { nodes: 1, shards: 1, replicas: 1 },
  };
}

// ---------------------------------------------------------------------------
// Text Formatter
// ---------------------------------------------------------------------------

export function formatSummary(r) {
  const lines = [];
  const a = (s) => lines.push(s);

  a('=' .repeat(64));
  a(`  ARCHITECTURE SUMMARY — ${r.datasetName}`);
  a('='.repeat(64));
  a('');
  a(`  Dataset:         ${r.datasetName}`);
  a(`  Vectors:         ${r.numVectors.toLocaleString()}`);
  a(`  Dimensions:      ${r.dimensions}`);
  a(`  Embedding Class: ${r.classification.cls} (${r.classification.label})`);
  a(`  Distance Metric: ${r.classification.metric} (Qdrant: ${r.classification.qdrant})`);
  a('');
  a('  SLA Requirements');
  a('  ' + '-'.repeat(30));
  a(`  Target QPS:      ${r.peakQPS.toLocaleString()}`);
  a(`  P99 Latency:     ${r.p99LatencyMs}ms`);
  a(`  Recall Target:   ${Math.round(r.recallTarget * 100)}%`);
  a(`  Top-k:           ${r.topK}`);
  a(r.writePattern === 'streaming'
    ? `  Write Pattern:   ${r.writePattern} (${r.peakWriteRate}/s peak)`
    : `  Write Pattern:   ${r.writePattern} (batch size: ${r.batchSize.toLocaleString()})`);
  a('');
  a('  Regime Classification');
  a('  ' + '-'.repeat(30));
  a(`  Recall Regime:   ${r.recall.regime}`);
  a(`  Latency Tier:    ${r.latency.tier}`);
  a('');
  a('  Quantization Strategy');
  a('  ' + '-'.repeat(30));
  a(`  Method:          ${r.recall.quant}`);
  a(`  Oversampling:    ${r.hnsw.oversampling || 'N/A'}`);
  a(`  Rescore:         ${r.recall.rescore ? 'ON' : 'N/A'}`);
  a('');
  a('  HNSW Parameters');
  a('  ' + '-'.repeat(30));
  a(`  m:               ${r.hnsw.m}`);
  a(`  ef_construct:    ${r.hnsw.efConstruct}`);
  a(`  hnsw_ef:         ${r.hnsw.hnswEf}  (${r.hnsw.efNote})`);
  a('');
  a('  Storage Placement');
  a('  ' + '-'.repeat(30));
  a(`  HNSW index:      ${r.latency.hnswPlacement}`);
  a(`  Quantized vecs:  ${r.latency.quantPlacement}`);
  a(`  Full vectors:    ${r.latency.fullPlacement}${r.latency.promoted ? ' (promoted: no-quant rule)' : ''}`);
  a('');
  a('  Memory Calculation');
  a('  ' + '-'.repeat(30));
  if (r.memory.quantMB > 0) a(`  Quantized vecs:  ${r.memory.quantMB.toLocaleString()} MB`);
  a(`  Full vectors:    ${r.memory.fullMB.toLocaleString()} MB`);
  a(`  HNSW index:      ${r.memory.hnswMB.toLocaleString()} MB`);
  a(`  Page cache:      ${r.memory.pageCacheMB.toLocaleString()} MB`);
  a(`  Process overhead: ${r.memory.processOverheadMB.toLocaleString()} MB`);
  a(`  Merge headroom:  ${r.memory.mergeHeadroomMB.toLocaleString()} MB`);
  a(`  Total RAM:       ${r.memory.totalMB.toLocaleString()} MB → ${r.memory.roundedGB} GB`);
  a('');
  a('  Disk Calculation');
  a('  ' + '-'.repeat(30));
  a(`  Total disk:      ${r.disk.totalGB} GB`);
  a(`  Disk type:       ${r.disk.diskType}`);
  a('');
  a('  Compute Calculation');
  a('  ' + '-'.repeat(30));
  a(`  Per-query time:  ${r.compute.perQueryMs}ms`);
  a(`    base:          ${r.compute.perQueryBaseMs}ms (ef adjustment: ${r.compute.efAdjustment}x)`);
  a(`    rescore:       ${r.compute.rescoreMs}ms`);
  a(`    mmap overhead: ${r.compute.mmapOverheadMs}ms`);
  a(`  Cores for QPS:   ${r.compute.coresForQueries}`);
  a(`  Headroom (${r.compute.headroomPct}%):  ${r.compute.headroomCores}`);
  a(`  Write cores:     ${r.compute.writeCores}`);
  a(`  Total vCPUs:     ${r.compute.totalVcpus}`);
  a('');
  a('  Topology');
  a('  ' + '-'.repeat(30));
  a(`  Nodes: ${r.topology.nodes}  Shards: ${r.topology.shards}  Replicas: ${r.topology.replicas}`);
  a('  Single node. Present HA option (replication_factor=2) to customer.');
  a('');
  a('  Instance & Cost');
  a('  ' + '-'.repeat(30));
  a(`  Instance:        ${r.instance.instanceType} (${r.instance.instanceVcpus} vCPU, ${r.instance.instanceRamGB} GB)`);
  a(`  Constraint:      ${r.instance.dominantConstraint} → ${r.instance.instanceFamily}`);
  if (r.instance.overprovisionNote) a(`  Note:            ${r.instance.overprovisionNote}`);
  a(`  Cost (on-demand): $${r.instance.costOnDemandMo.toLocaleString()}/mo`);
  a(`  Cost (reserved):  $${r.instance.costReservedMo.toLocaleString()}/mo`);
  a('');
  a('  Future Optimizations');
  a('  ' + '-'.repeat(30));
  r.optimizations.forEach((opt, i) => a(`  ${i + 1}. ${opt}`));
  a('');
  a('='.repeat(64));

  return lines.join('\n');
}
