#!/usr/bin/env node
/**
 * Qdrant Cluster Sizer CLI — Node wrapper around sizer_engine.js
 *
 * Usage:
 *   Interactive:  node qdrant_sizer_cli.js
 *   JSON input:   node qdrant_sizer_cli.js --json '{"embeddingType":"neural_text",...}'
 *   File input:   node qdrant_sizer_cli.js --file scenarios.json
 *   JSON output:  node qdrant_sizer_cli.js --json-out '{"embeddingType":"neural_text",...}'
 */

import { createInterface } from 'node:readline';
import { readFileSync } from 'node:fs';
import { sizeCluster, formatSummary } from './sizer_engine.js';

// ---------------------------------------------------------------------------
// Interactive helpers
// ---------------------------------------------------------------------------

function createPrompt() {
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  return (q) => new Promise(resolve => rl.question(q, resolve));
}

const EMBEDDING_TYPES = {
  '1': { value: 'neural_text',   label: 'Neural text model (OpenAI, Cohere, BGE, E5, etc.)' },
  '2': { value: 'neural_vision', label: 'Neural vision model (CLIP, DINOv2, etc.)' },
  '3': { value: 'classical_cv',  label: 'Classical CV features (GIST, SIFT, HOG, etc.)' },
  '4': { value: 'other',         label: 'Other / Unknown' },
};

const WRITE_PATTERNS = {
  '1': { value: 'streaming', label: 'Streaming (continuous writes, concurrent with reads)' },
  '2': { value: 'batch',     label: 'Batch (periodic bulk loads, no concurrent reads)' },
  '3': { value: 'static',    label: 'Rare / Static (load once, mostly read)' },
};

async function askChoice(ask, prompt, choices) {
  while (true) {
    const raw = (await ask(prompt)).trim();
    if (choices[raw]) return choices[raw].value;
    const valid = Object.keys(choices).join(', ');
    console.log(`    Invalid choice '${raw}'. Please enter one of: ${valid}`);
  }
}

async function askInt(ask, prompt, min = 1, max = 1e10) {
  while (true) {
    const raw = (await ask(prompt)).trim();
    const val = parseInt(raw, 10);
    if (isNaN(val)) { console.log(`    '${raw}' is not a valid integer.`); continue; }
    if (val < min) { console.log(`    Value must be at least ${min}. Got ${val}.`); continue; }
    if (val > max) { console.log(`    Value must be at most ${max}. Got ${val}.`); continue; }
    return val;
  }
}

async function askFloat(ask, prompt, min = 0, max = 100) {
  while (true) {
    const raw = (await ask(prompt)).trim();
    const val = parseFloat(raw);
    if (isNaN(val)) { console.log(`    '${raw}' is not a valid number.`); continue; }
    if (val < min) { console.log(`    Value must be at least ${min}.`); continue; }
    if (val > max) { console.log(`    Value must be at most ${max}.`); continue; }
    return val;
  }
}

// ---------------------------------------------------------------------------
// Interactive mode
// ---------------------------------------------------------------------------

async function interactive() {
  const ask = createPrompt();

  console.log('\n' + '='.repeat(64));
  console.log('  Qdrant Cluster Sizer — Interactive Mode');
  console.log('='.repeat(64) + '\n');

  const datasetName = (await ask('  Dataset name: ')).trim() || 'unnamed';

  console.log('\n  Embedding type:');
  for (const [k, v] of Object.entries(EMBEDDING_TYPES)) console.log(`    ${k}. ${v.label}`);
  const embeddingType = await askChoice(ask, '  Choice [1-4]: ', EMBEDDING_TYPES);

  const dimensions = await askInt(ask, '  Dimensions: ', 1, 65536);
  const numVectors = await askInt(ask, '  Number of vectors: ', 1);
  const peakQPS = await askInt(ask, '  Peak QPS: ', 1);

  console.log('\n  Write pattern:');
  for (const [k, v] of Object.entries(WRITE_PATTERNS)) console.log(`    ${k}. ${v.label}`);
  const writePattern = await askChoice(ask, '  Choice [1-3]: ', WRITE_PATTERNS);

  let peakWriteRate = 0, batchSize = 0;
  if (writePattern === 'streaming')
    peakWriteRate = await askInt(ask, '  Peak write rate (vectors/s): ', 0);
  else if (writePattern === 'batch')
    batchSize = await askInt(ask, '  Batch size (vectors per load): ', 1);

  const p99LatencyMs = await askInt(ask, '  P99 latency SLA (ms): ', 1, 60000);
  const recallPct = await askFloat(ask, '  Recall target (e.g., 95 for 95%): ', 0.01, 100);
  const recallTarget = recallPct > 1 ? recallPct / 100 : recallPct;
  const topK = await askInt(ask, '  Top-k: ', 1, 10000);

  const result = sizeCluster({
    datasetName, embeddingType, dimensions, numVectors, peakQPS,
    writePattern, peakWriteRate, batchSize, p99LatencyMs, recallTarget, topK,
  });

  console.log('\n' + formatSummary(result));
  process.exit(0);
}

// ---------------------------------------------------------------------------
// JSON/File modes
// ---------------------------------------------------------------------------

function normalizeConfig(raw) {
  // Accept both camelCase (JS) and snake_case (Python compat) keys
  return {
    datasetName:   raw.datasetName   || raw.dataset_name   || 'unnamed',
    embeddingType: raw.embeddingType  || raw.embedding_type,
    dimensions:    raw.dimensions,
    numVectors:    raw.numVectors     || raw.num_vectors,
    peakQPS:       raw.peakQPS        || raw.peak_qps,
    writePattern:  raw.writePattern   || raw.write_pattern  || 'streaming',
    peakWriteRate: raw.peakWriteRate  || raw.peak_write_rate || 0,
    batchSize:     raw.batchSize      || raw.batch_size      || 0,
    p99LatencyMs:  raw.p99LatencyMs   || raw.p99_latency_ms,
    recallTarget:  raw.recallTarget   || raw.recall_target,
    topK:          raw.topK           || raw.top_k,
  };
}

// ---------------------------------------------------------------------------
// CLI entry point
// ---------------------------------------------------------------------------

const args = process.argv.slice(2);

if (args.length === 0) {
  interactive();
} else if (args[0] === '--json' && args[1]) {
  const config = normalizeConfig(JSON.parse(args[1]));
  console.log(formatSummary(sizeCluster(config)));
} else if (args[0] === '--json-out' && args[1]) {
  const config = normalizeConfig(JSON.parse(args[1]));
  console.log(JSON.stringify(sizeCluster(config), null, 2));
} else if (args[0] === '--file' && args[1]) {
  const data = JSON.parse(readFileSync(args[1], 'utf8'));
  const configs = Array.isArray(data) ? data : [data];
  for (const raw of configs) {
    console.log(formatSummary(sizeCluster(normalizeConfig(raw))));
    console.log();
  }
} else {
  console.log(`
Qdrant Cluster Sizer CLI

Usage:
  node qdrant_sizer_cli.js                          Interactive mode
  node qdrant_sizer_cli.js --json '{"...": ...}'    JSON input, text output
  node qdrant_sizer_cli.js --json-out '{"...": ...}' JSON input, JSON output
  node qdrant_sizer_cli.js --file scenarios.json    File input (array or single)

Config keys (camelCase or snake_case):
  embeddingType / embedding_type:  neural_text | neural_vision | classical_cv | other
  dimensions, numVectors / num_vectors, peakQPS / peak_qps
  writePattern / write_pattern:    streaming | batch | static
  peakWriteRate / peak_write_rate, batchSize / batch_size
  p99LatencyMs / p99_latency_ms, recallTarget / recall_target, topK / top_k
`);
  process.exit(1);
}
