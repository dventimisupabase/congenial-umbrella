/**
 * pgvector Cluster Sizer Engine — Single source of truth.
 *
 * This module contains ALL sizing logic and constants for PostgreSQL + pgvector
 * deployments. It is consumed by:
 *   - index.html (browser wizard)
 *
 * No logic should be duplicated outside this file.
 */

// ---------------------------------------------------------------------------
// Constants — All decision tables live here
// ---------------------------------------------------------------------------

// Storage bytes per dimension by vector type
export const BYTES_PER_DIM = {
  vector: 4,     // float32
  halfvec: 2,    // float16
  bit: 1 / 8,    // 1 bit per dimension
};

// Per-vector fixed overhead in Postgres (tuple header + alignment + ItemPointer)
export const TUPLE_OVERHEAD_BYTES = 36;

// pgvector index types
export const INDEX_TYPES = ['hnsw', 'ivfflat'];

// Distance operators
export const DISTANCE_OPS = {
  l2:      { operator: '<->', opsClass: 'vector_l2_ops',     halfvecOps: 'halfvec_l2_ops',     label: 'L2 (Euclidean)' },
  cosine:  { operator: '<=>', opsClass: 'vector_cosine_ops', halfvecOps: 'halfvec_cosine_ops', label: 'Cosine' },
  ip:      { operator: '<#>', opsClass: 'vector_ip_ops',     halfvecOps: 'halfvec_ip_ops',     label: 'Inner Product' },
};

// HNSW default parameters from pgvector source
export const HNSW_DEFAULTS = { m: 16, ef_construction: 64 };

// HNSW m recommendations by recall regime
export const HNSW_M = { RELAXED: 16, MODERATE: 24, STRICT: 32 };

// HNSW ef_construction recommendations
export const HNSW_EF_CONSTRUCTION = { RELAXED: 64, MODERATE: 128, STRICT: 256 };

// hnsw.ef_search recommendations (query-time parameter)
export const HNSW_EF_SEARCH_MULTIPLIER = { RELAXED: 2, MODERATE: 4, STRICT: 8 };
export const HNSW_EF_SEARCH_FLOOR = { RELAXED: 40, MODERATE: 100, STRICT: 200 };

// IVFFlat: lists rule of thumb
// <= 1M rows: rows/1000; > 1M rows: sqrt(rows)
export const IVFFLAT_PROBES_MULTIPLIER = { RELAXED: 1, MODERATE: 2, STRICT: 4 };

// Whether halfvec indexing is viable given embedding type and recall
export const HALFVEC_MATRIX = {
  'neural_text-RELAXED': true,  'neural_text-MODERATE': true,  'neural_text-STRICT': true,
  'neural_vision-RELAXED': true,'neural_vision-MODERATE': true, 'neural_vision-STRICT': false,
  'classical_cv-RELAXED': true, 'classical_cv-MODERATE': false, 'classical_cv-STRICT': false,
  'other-RELAXED': true,        'other-MODERATE': false,        'other-STRICT': false,
};

// CPU base ms per query estimate by dimension bucket and storage type
export const CPU_BASE_MS = {
  '<512-halfvec': 0.3,     '<512-vector': 0.5,
  '512-1024-halfvec': 0.6, '512-1024-vector': 1.2,
  '1024-2048-halfvec': 1.0,'1024-2048-vector': 2.0,
  '2048-4096-halfvec': 1.5,'2048-4096-vector': 3.5,
};

// Supabase compute tier catalog
export const INSTANCES = [
  { name: 'Micro',  vcpus: 2,  ram: 1,   hourly: 0.01370, platform: 'Supabase', dedicated: false },
  { name: 'Small',  vcpus: 2,  ram: 2,   hourly: 0.02055, platform: 'Supabase', dedicated: false },
  { name: 'Medium', vcpus: 2,  ram: 4,   hourly: 0.08220, platform: 'Supabase', dedicated: false },
  { name: 'Large',  vcpus: 2,  ram: 8,   hourly: 0.15070, platform: 'Supabase', dedicated: true },
  { name: 'XL',     vcpus: 4,  ram: 16,  hourly: 0.28770, platform: 'Supabase', dedicated: true },
  { name: '2XL',    vcpus: 8,  ram: 32,  hourly: 0.56165, platform: 'Supabase', dedicated: true },
  { name: '4XL',    vcpus: 16, ram: 64,  hourly: 1.09590, platform: 'Supabase', dedicated: true },
  { name: '8XL',    vcpus: 32, ram: 128, hourly: 2.19180, platform: 'Supabase', dedicated: true },
  { name: '12XL',   vcpus: 48, ram: 192, hourly: 3.28770, platform: 'Supabase', dedicated: true },
  { name: '16XL',   vcpus: 64, ram: 256, hourly: 4.38360, platform: 'Supabase', dedicated: true },
];

export const RAM_TIERS = [8, 16, 32, 64, 128, 256, 512];

// ---------------------------------------------------------------------------
// Stage 1: Classify Embeddings
// ---------------------------------------------------------------------------

export function classifyEmbeddings(embeddingType, dimensions) {
  if (embeddingType === 'neural_text') {
    return { cls: 'neural_text', label: 'Neural text embeddings', metric: 'Cosine', pgOps: 'vector_cosine_ops' };
  }
  if (embeddingType === 'neural_vision') {
    return { cls: 'neural_vision', label: 'Neural vision embeddings', metric: 'Cosine', pgOps: 'vector_cosine_ops' };
  }
  if (embeddingType === 'classical_cv') {
    return { cls: 'classical_cv', label: 'Classical CV features', metric: 'L2 (Euclidean)', pgOps: 'vector_l2_ops' };
  }
  return { cls: 'other', label: 'General embeddings', metric: 'Cosine', pgOps: 'vector_cosine_ops' };
}

// ---------------------------------------------------------------------------
// Stage 2: Recall Regime & Storage Type
// ---------------------------------------------------------------------------

export function recallRegime(recallTarget, embeddingClass) {
  let regime;
  if (recallTarget <= 0.95) regime = 'RELAXED';
  else if (recallTarget <= 0.98) regime = 'MODERATE';
  else regime = 'STRICT';

  const useHalfvec = HALFVEC_MATRIX[`${embeddingClass}-${regime}`] || false;
  const storageType = useHalfvec ? 'halfvec' : 'vector';
  const bytesPerDim = BYTES_PER_DIM[storageType];
  const compression = useHalfvec ? '2x (float32 → float16)' : 'None (full float32)';

  return { regime, storageType, bytesPerDim, compression, useHalfvec };
}

// ---------------------------------------------------------------------------
// Stage 3: Index Strategy
// ---------------------------------------------------------------------------

export function indexStrategy(regime, topK, numVectors, indexType) {
  if (indexType === 'hnsw') {
    const m = HNSW_M[regime];
    const efConstruction = HNSW_EF_CONSTRUCTION[regime];
    const efSearchCalc = HNSW_EF_SEARCH_MULTIPLIER[regime] * topK;
    const efSearchFloor = HNSW_EF_SEARCH_FLOOR[regime];
    const efSearch = Math.max(efSearchCalc, efSearchFloor);
    const efNote = efSearch > efSearchCalc
      ? `floor-adjusted from ${efSearchCalc} to ${efSearch}`
      : `${HNSW_EF_SEARCH_MULTIPLIER[regime]}x top_k`;

    return { indexType: 'hnsw', m, efConstruction, efSearch, efNote };
  }

  // IVFFlat
  const lists = numVectors <= 1_000_000
    ? Math.max(Math.round(numVectors / 1000), 10)
    : Math.round(Math.sqrt(numVectors));
  const probesBase = Math.max(Math.round(Math.sqrt(lists)), 1);
  const probes = probesBase * IVFFLAT_PROBES_MULTIPLIER[regime];

  return { indexType: 'ivfflat', lists, probes };
}

// ---------------------------------------------------------------------------
// Stage 4: Memory Sizing (PostgreSQL-aware)
// ---------------------------------------------------------------------------

export function sizeMemory(numVectors, dimensions, recall, index) {
  const bytesPerDim = recall.bytesPerDim;

  // Table size: vectors + tuple overhead
  const vectorBytes = dimensions * bytesPerDim;
  const rowBytes = vectorBytes + TUPLE_OVERHEAD_BYTES;
  const tableMB = numVectors * rowBytes / (1024 * 1024);

  // Index size
  let indexMB;
  if (index.indexType === 'hnsw') {
    // HNSW graph: each node stores ~(2 * m) neighbor pointers (8 bytes each) + vector copy
    const graphMB = numVectors * (index.m * 2 * 8 + vectorBytes) / (1024 * 1024);
    indexMB = graphMB * 1.2; // 20% overhead for internal structures
  } else {
    // IVFFlat: stores vectors in list buckets + centroid data
    const listOverhead = index.lists * vectorBytes / (1024 * 1024);
    indexMB = tableMB * 1.1 + listOverhead; // vectors re-stored in index + centroids
  }

  // shared_buffers: ideally holds the full index + hot table data
  // Rule of thumb: 25% of total RAM, but must fit index
  const idealSharedBuffersMB = indexMB + tableMB * 0.3; // index + 30% of table for hot rows

  // effective_cache_size: total memory Postgres can use (incl. OS page cache)
  // Should be ~75% of total RAM
  const idealEffectiveCacheMB = indexMB + tableMB;

  // maintenance_work_mem: needed for index builds
  let maintenanceWorkMemMB;
  if (index.indexType === 'hnsw') {
    maintenanceWorkMemMB = Math.max(indexMB * 1.5, 1024); // HNSW builds need significant memory
  } else {
    maintenanceWorkMemMB = Math.max(tableMB * 0.5, 512); // IVFFlat is lighter
  }

  // Total RAM needed
  const pgOverheadMB = 512; // Postgres process overhead, WAL buffers, etc.
  const osReserveMB = 1024; // OS + filesystem cache breathing room
  const totalRequiredMB = idealEffectiveCacheMB + pgOverheadMB + osReserveMB;

  // Round up to nearest RAM tier
  let roundedGB = RAM_TIERS[RAM_TIERS.length - 1];
  for (const tier of RAM_TIERS) {
    if (tier * 1024 >= totalRequiredMB) { roundedGB = tier; break; }
  }

  // Postgres config recommendations
  const sharedBuffersGB = Math.max(Math.round(roundedGB * 0.25), 1);
  const effectiveCacheSizeGB = Math.max(Math.round(roundedGB * 0.75), 2);

  return {
    tableMB: Math.round(tableMB),
    indexMB: Math.round(indexMB),
    idealSharedBuffersMB: Math.round(idealSharedBuffersMB),
    idealEffectiveCacheMB: Math.round(idealEffectiveCacheMB),
    maintenanceWorkMemMB: Math.round(maintenanceWorkMemMB),
    pgOverheadMB,
    osReserveMB,
    totalRequiredMB: Math.round(totalRequiredMB),
    roundedGB,
    sharedBuffersGB,
    effectiveCacheSizeGB,
  };
}

// ---------------------------------------------------------------------------
// Stage 5: Disk Sizing
// ---------------------------------------------------------------------------

export function sizeDisk(numVectors, dimensions, recall, memory, index) {
  const vectorBytes = dimensions * recall.bytesPerDim;
  const rowBytes = vectorBytes + TUPLE_OVERHEAD_BYTES;

  // Table on disk
  const tableGB = numVectors * rowBytes / (1024 * 1024 * 1024);
  // Index on disk
  const indexGB = memory.indexMB / 1024;
  // WAL: estimate 2x table size for write headroom
  const walGB = Math.max(tableGB * 2, 2);
  // TOAST + overhead
  const overheadGB = (tableGB + indexGB) * 0.3;
  // Snapshot / backup headroom
  const backupGB = tableGB + indexGB;

  const totalGB = Math.max(Math.ceil(tableGB + indexGB + walGB + overheadGB + backupGB), 30);

  // Disk type recommendation
  const needsHighIOPS = memory.totalRequiredMB > memory.roundedGB * 1024 * 0.8; // tight on RAM = more disk reads
  const diskType = needsHighIOPS ? 'High-performance SSD (provisioned IOPS recommended)' : 'SSD (standard Supabase disk)';

  return {
    tableGB: +tableGB.toFixed(1),
    indexGB: +indexGB.toFixed(1),
    walGB: +walGB.toFixed(1),
    overheadGB: +overheadGB.toFixed(1),
    backupGB: +backupGB.toFixed(1),
    totalGB,
    diskType,
  };
}

// ---------------------------------------------------------------------------
// Stage 6: Compute Sizing
// ---------------------------------------------------------------------------

function dimBucket(dims) {
  if (dims < 512) return '<512';
  if (dims <= 1024) return '512-1024';
  if (dims <= 2048) return '1024-2048';
  return '2048-4096';
}

export function sizeCompute(dimensions, recall, index, peakQPS, writePattern, peakWriteRate) {
  const storageKey = `${dimBucket(dimensions)}-${recall.storageType}`;
  const baseMs = CPU_BASE_MS[storageKey] || 2.0;

  let efAdjustment = 1.0;
  if (index.indexType === 'hnsw') {
    efAdjustment = index.efSearch / 40; // normalize against baseline ef_search=40
  } else {
    efAdjustment = index.probes / 10; // normalize against baseline probes=10
  }
  efAdjustment = Math.max(efAdjustment, 0.5);

  const perQueryMs = baseMs * efAdjustment;
  const coresForQueries = peakQPS * (perQueryMs / 1000);

  // Headroom
  const streamingWrites = writePattern === 'streaming';
  const headroomPct = streamingWrites ? 80 : 30;
  const headroomCores = coresForQueries * (headroomPct / 100);

  // Write cores
  let writeCores = 0;
  if (writePattern === 'streaming') {
    writeCores = peakWriteRate * 0.002; // ~2ms per insert with index maintenance
  }

  // Postgres background processes: autovacuum, checkpointer, WAL writer
  const bgCores = 2;

  const totalVcpus = Math.max(Math.ceil(coresForQueries + headroomCores + writeCores + bgCores), 2);

  return {
    perQueryBaseMs: +baseMs.toFixed(2),
    efAdjustment: +efAdjustment.toFixed(1),
    perQueryMs: +perQueryMs.toFixed(2),
    coresForQueries: +coresForQueries.toFixed(1),
    headroomPct,
    headroomCores: +headroomCores.toFixed(1),
    writeCores: +writeCores.toFixed(1),
    bgCores,
    totalVcpus,
  };
}

// ---------------------------------------------------------------------------
// Stage 7: Instance Selection
// ---------------------------------------------------------------------------

export function selectInstance(compute, memory, disk) {
  const vcpus = compute.totalVcpus;
  const ramGB = memory.roundedGB;

  // Find the smallest Supabase tier that meets both vCPU and RAM requirements
  const candidates = INSTANCES.sort((a, b) => a.vcpus - b.vcpus || a.ram - b.ram);

  let selected = candidates.find(i => i.vcpus >= vcpus && i.ram >= ramGB);
  let overprovisionNote = '';

  if (!selected) {
    selected = candidates[candidates.length - 1];
    overprovisionNote = 'Workload exceeds largest available Supabase tier; consider read replicas or partitioning.';
  }

  const monthly = selected.hourly * 730;
  const diskCost = disk.totalGB * 0.08;

  return {
    tierName: selected.name,
    tierVcpus: selected.vcpus,
    tierRamGB: selected.ram,
    dedicated: selected.dedicated,
    monthlyCost: Math.round(monthly + diskCost),
    overprovisionNote,
  };
}

// ---------------------------------------------------------------------------
// Stage 8: Postgres Configuration
// ---------------------------------------------------------------------------

export function postgresConfig(memory, index, compute, writePattern) {
  const config = {};

  config.shared_buffers = `${memory.sharedBuffersGB}GB`;
  config.effective_cache_size = `${memory.effectiveCacheSizeGB}GB`;
  config.maintenance_work_mem = `${Math.round(memory.maintenanceWorkMemMB)}MB`;
  config.work_mem = `${Math.max(Math.round(memory.roundedGB * 1024 / (compute.totalVcpus * 4)), 64)}MB`;

  // Parallel workers
  config.max_parallel_workers_per_gather = Math.min(Math.max(Math.floor(compute.totalVcpus / 4), 1), 4);
  config.max_parallel_maintenance_workers = Math.min(Math.max(Math.floor(compute.totalVcpus / 2), 1), 7);

  // pgvector-specific
  if (index.indexType === 'hnsw') {
    config['hnsw.ef_search'] = index.efSearch;
  } else {
    config['ivfflat.probes'] = index.probes;
  }

  // WAL tuning for write-heavy workloads
  if (writePattern === 'streaming') {
    config.wal_buffers = '64MB';
    config.checkpoint_completion_target = 0.9;
  }

  return config;
}

// ---------------------------------------------------------------------------
// Stage 9: Optimization Suggestions
// ---------------------------------------------------------------------------

export function suggestOptimizations(classification, recall, index, dimensions, topK, numVectors) {
  const opts = [];

  opts.push('Read replicas: Supabase supports read replicas on paid plans for HA and load distribution.');

  if (recall.storageType === 'vector' && classification.cls === 'neural_text') {
    opts.push('halfvec indexing: cast to halfvec for 2x index compression. Test recall impact: CREATE INDEX ON tbl USING hnsw ((embedding::halfvec(N)) halfvec_cosine_ops)');
  }

  if (index.indexType === 'hnsw' && numVectors > 5_000_000) {
    opts.push('Partitioning: for >5M vectors, consider range or hash partitioning with per-partition indexes.');
  }

  if (index.indexType === 'ivfflat') {
    opts.push('Rebuild IVFFlat indexes periodically: centroid quality degrades as data distribution shifts.');
    opts.push('Consider HNSW: better recall at similar cost, no retraining needed. Slower builds but faster queries.');
  }

  if (index.indexType === 'hnsw' && index.efSearch > 200) {
    opts.push(`Reduce hnsw.ef_search from ${index.efSearch} if recall has margin — lower ef_search = higher QPS.`);
  }

  if (dimensions >= 1536 && classification.cls === 'neural_text') {
    opts.push(`Matryoshka reduction: test ${dimensions} → ${Math.floor(dimensions / 2)} dims against recall target.`);
  }

  opts.push('PLAIN storage: ALTER TABLE ... ALTER COLUMN embedding SET STORAGE PLAIN to avoid TOAST overhead for vectors.');

  return opts;
}

// ---------------------------------------------------------------------------
// Stage 10: SQL Generation
// ---------------------------------------------------------------------------

export function generateSQL(datasetName, dimensions, classification, recall, index) {
  const tableName = datasetName.toLowerCase().replace(/[^a-z0-9_]/g, '_').replace(/_+/g, '_');
  const vecType = recall.useHalfvec ? `halfvec(${dimensions})` : `vector(${dimensions})`;
  const opsClass = recall.useHalfvec
    ? DISTANCE_OPS[classification.metric === 'Cosine' ? 'cosine' : classification.metric === 'L2 (Euclidean)' ? 'l2' : 'ip'].halfvecOps
    : classification.pgOps;
  const distOp = classification.metric === 'Cosine' ? '<=>' : classification.metric === 'L2 (Euclidean)' ? '<->' : '<#>';

  const lines = [];
  lines.push(`-- Table`);
  lines.push(`CREATE TABLE ${tableName} (`);
  lines.push(`    id bigserial PRIMARY KEY,`);
  lines.push(`    content text,`);
  lines.push(`    embedding ${vecType}`);
  lines.push(`);`);
  lines.push(``);
  lines.push(`-- Optimize storage (avoid TOAST overhead)`);
  lines.push(`ALTER TABLE ${tableName} ALTER COLUMN embedding SET STORAGE PLAIN;`);
  lines.push(``);

  if (index.indexType === 'hnsw') {
    const colExpr = recall.useHalfvec ? `embedding` : `embedding`;
    lines.push(`-- HNSW index`);
    lines.push(`CREATE INDEX ON ${tableName} USING hnsw (${colExpr} ${opsClass}) WITH (m = ${index.m}, ef_construction = ${index.efConstruction});`);
    lines.push(``);
    lines.push(`-- Query-time settings`);
    lines.push(`SET hnsw.ef_search = ${index.efSearch};`);
  } else {
    lines.push(`-- IVFFlat index`);
    lines.push(`CREATE INDEX ON ${tableName} USING ivfflat (embedding ${opsClass}) WITH (lists = ${index.lists});`);
    lines.push(``);
    lines.push(`-- Query-time settings`);
    lines.push(`SET ivfflat.probes = ${index.probes};`);
  }

  lines.push(``);
  lines.push(`-- Example query`);
  lines.push(`SELECT id, content, embedding ${distOp} $1 AS distance`);
  lines.push(`FROM ${tableName}`);
  lines.push(`ORDER BY embedding ${distOp} $1`);
  lines.push(`LIMIT 10;`);

  return lines.join('\n');
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
    indexType = 'hnsw',
  } = config;

  const classification = classifyEmbeddings(embeddingType, dimensions);
  const recall = recallRegime(recallTarget, classification.cls);
  const index = indexStrategy(recall.regime, topK, numVectors, indexType);
  const memory = sizeMemory(numVectors, dimensions, recall, index);
  const disk = sizeDisk(numVectors, dimensions, recall, memory, index);
  const compute = sizeCompute(dimensions, recall, index, peakQPS, writePattern, peakWriteRate);
  const instance = selectInstance(compute, memory, disk);
  const pgConfig = postgresConfig(memory, index, compute, writePattern);
  const optimizations = suggestOptimizations(classification, recall, index, dimensions, topK, numVectors);
  const sql = generateSQL(datasetName, dimensions, classification, recall, index);

  return {
    datasetName, dimensions, numVectors, peakQPS, writePattern,
    peakWriteRate, batchSize, p99LatencyMs, recallTarget, topK,
    classification, recall, index, memory, disk, compute, instance,
    pgConfig, optimizations, sql,
    topology: { nodes: 1, readReplicas: 0 },
  };
}

// ---------------------------------------------------------------------------
// Text Formatter
// ---------------------------------------------------------------------------

export function formatSummary(r) {
  const lines = [];
  const a = (s) => lines.push(s);

  a('='.repeat(64));
  a(`  ARCHITECTURE SUMMARY — ${r.datasetName}`);
  a('='.repeat(64));
  a('');
  a(`  Dataset:         ${r.datasetName}`);
  a(`  Vectors:         ${r.numVectors.toLocaleString()}`);
  a(`  Dimensions:      ${r.dimensions}`);
  a(`  Embedding Class: ${r.classification.label}`);
  a(`  Distance Metric: ${r.classification.metric} (${r.classification.pgOps})`);
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
  a('  Storage Strategy');
  a('  ' + '-'.repeat(30));
  a(`  Recall Regime:   ${r.recall.regime}`);
  a(`  Vector Type:     ${r.recall.storageType}`);
  a(`  Compression:     ${r.recall.compression}`);
  a('');
  a(`  Index Configuration (${r.index.indexType.toUpperCase()})`);
  a('  ' + '-'.repeat(30));
  if (r.index.indexType === 'hnsw') {
    a(`  m:               ${r.index.m}`);
    a(`  ef_construction: ${r.index.efConstruction}`);
    a(`  hnsw.ef_search:  ${r.index.efSearch}  (${r.index.efNote})`);
  } else {
    a(`  lists:           ${r.index.lists}`);
    a(`  probes:          ${r.index.probes}`);
  }
  a('');
  a('  Memory Sizing');
  a('  ' + '-'.repeat(30));
  a(`  Table data:       ${r.memory.tableMB.toLocaleString()} MB`);
  a(`  Index:            ${r.memory.indexMB.toLocaleString()} MB`);
  a(`  PG overhead:      ${r.memory.pgOverheadMB.toLocaleString()} MB`);
  a(`  OS reserve:       ${r.memory.osReserveMB.toLocaleString()} MB`);
  a(`  Total required:   ${r.memory.totalRequiredMB.toLocaleString()} MB → ${r.memory.roundedGB} GB`);
  a('');
  a('  PostgreSQL Configuration');
  a('  ' + '-'.repeat(30));
  for (const [k, v] of Object.entries(r.pgConfig)) {
    a(`  ${k} = ${v}`);
  }
  a('');
  a('  Disk');
  a('  ' + '-'.repeat(30));
  a(`  Total disk:      ${r.disk.totalGB} GB`);
  a(`  Disk type:       ${r.disk.diskType}`);
  a('');
  a('  Compute');
  a('  ' + '-'.repeat(30));
  a(`  Per-query time:  ${r.compute.perQueryMs}ms`);
  a(`  Cores for QPS:   ${r.compute.coresForQueries}`);
  a(`  Headroom (${r.compute.headroomPct}%):  ${r.compute.headroomCores}`);
  a(`  Write cores:     ${r.compute.writeCores}`);
  a(`  BG processes:    ${r.compute.bgCores}`);
  a(`  Total vCPUs:     ${r.compute.totalVcpus}`);
  a('');
  a('  Compute Tier & Cost');
  a('  ' + '-'.repeat(30));
  a(`  Compute Tier:    ${r.instance.tierName} (${r.instance.tierVcpus} vCPU, ${r.instance.tierRamGB} GB)`);
  a(`  CPU:             ${r.instance.dedicated ? 'Dedicated' : 'Shared'}`);
  if (r.instance.overprovisionNote) a(`  Note:            ${r.instance.overprovisionNote}`);
  a(`  Monthly Cost:    $${r.instance.monthlyCost.toLocaleString()}/mo`);
  a('');
  a('  Optimizations');
  a('  ' + '-'.repeat(30));
  r.optimizations.forEach((opt, i) => a(`  ${i + 1}. ${opt}`));
  a('');
  a('='.repeat(64));

  return lines.join('\n');
}
