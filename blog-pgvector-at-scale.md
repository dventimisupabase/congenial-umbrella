# pgvector at 1 Million Vectors: What We Actually Measured

There's a growing body of opinion that pgvector isn't ready for production. The arguments are familiar: index builds are expensive, filtered search is tricky, you need a dedicated vector database for anything serious.

Some of these arguments are valid. Some are outdated. And some confuse limitations of vector search in general with limitations of pgvector specifically.

We decided to find out empirically. We loaded 1,000,000 vectors into pgvector on Supabase, ran benchmarks across multiple compute tiers, measured recall and latency with EXPLAIN ANALYZE, and compared the results against what ~60,000 Supabase organizations with vector workloads actually look like in production.

Here's what we found.

## Most vector workloads are smaller than you think

Before talking about 1M-vector benchmarks, it's worth looking at what real production vector workloads actually look like. We pulled data from every Supabase organization running pgvector:

| Vector rows | Organizations | Median DB size |
|-------------|---------------|----------------|
| < 10K       | 46,691        | 40 MB          |
| 10K–100K    | 9,216         | 310 MB         |
| 100K–1M     | 3,483         | 2.1 GB         |
| 1M–10M      | 811           | 19.8 GB        |
| 10M–100M    | 164           | 116 GB         |
| 100M+       | 20            | 1.4 TB         |

**93% of organizations with vectors have fewer than 1 million rows.** The median database at 100K–1M vector rows is 2.1 GB — small enough to fit entirely in memory on a $15/month instance.

More interesting is the instance distribution. At the 1M–10M vector tier, 60% of organizations run on Micro or Small instances. They're not running 1,000 queries per second. They're doing 10–100 queries/second with relaxed latency budgets, alongside their regular relational workload.

This is the reality the "you need a dedicated vector database" argument misses: **for most applications, vectors are a feature, not the workload.** The database is already there for authentication, business logic, and relational data. pgvector adds vector search without adding a service.

## What happens at 1 million vectors

We loaded two datasets into pgvector on Supabase Cloud:

- **dbpedia**: 1M vectors at 3,072 dimensions (OpenAI text-embedding-3-large), cosine distance, halfvec expression index
- **gist-960**: 1M vectors at 960 dimensions (classical CV features), L2 distance, full float32

We tested across three Supabase compute tiers and measured server-side query latency using `EXPLAIN (ANALYZE, BUFFERS)`.

### The single most important finding

The HNSW index must fit entirely in PostgreSQL's `shared_buffers`. When it does, queries are fast. When it doesn't, they're 40–60x slower.

| Cache state                        | Buffer activity              | Server-side latency |
|------------------------------------|------------------------------|---------------------|
| **Warm** (index in shared_buffers) | `shared hit=4839, read=0`    | **21 ms**           |
| **Cold** (index partially on disk) | `shared hit=3261, read=1578` | **834 ms**          |

This isn't unique to pgvector — any database that stores an HNSW index needs that index in memory for fast queries. The difference is that PostgreSQL requires you to configure `shared_buffers` explicitly, while dedicated vector databases manage their own memory.

The practical implication: set `shared_buffers >= index_size`, not the standard PostgreSQL recommendation of 25% of RAM. On Supabase, this is a one-line CLI command (`supabase postgres-config update --config shared_buffers=20GB`). After any restart, warm the cache with `SELECT pg_prewarm('your_index_name')`.

### Recall depends on ef_search, and ef_search depends on scale

`hnsw.ef_search` is the query-time parameter that controls the recall/latency trade-off. Higher ef_search means more of the HNSW graph is explored, which means better recall but slower queries.

We swept ef_search from 20 to 800 on both datasets:

**gist-960** (960d, L2, 99% recall target):

| ef_search | Recall@10 | P50 latency |
|-----------|-----------|-------------|
| 100       | 95.2%     | 91 ms       |
| 200       | 98.5%     | 96 ms       |
| **300**   | **99.1%** | **100 ms**  |
| 400       | 99.6%     | 105 ms      |
| 500       | 99.8%     | 182 ms      |

**dbpedia** (3072d, cosine, 95% recall target):

| ef_search | Recall@100 | P50 latency |
|-----------|------------|-------------|
| 128       | 98.2%      | 127 ms      |
| 200       | 98.7%      | 130 ms      |
| 400       | 99.1%      | 140 ms      |
| 800       | 99.4%      | 160 ms      |

Two things stand out. First, the right ef_search depends on your dataset and scale — there's no universal formula. At 1M vectors, gist-960 needed ef_search=300 for 99% recall, while dbpedia hit 98.2% at just ef_search=128.

Second, there are diminishing returns. Going from 95% to 99% recall might double your ef_search, but going from 99% to 99.5% might double it again. The last few percentage points of recall are expensive.

### HNSW vs IVFFlat: the numbers

We tested both index types on gist-960 at 1M vectors:

|            | HNSW (m=24)       | IVFFlat (lists=1000) |
|------------|-------------------|----------------------|
| Index size | 7,678 MB          | 3,912 MB             |
| Build time | ~35 min           | ~5 min               |
| 99% recall | ef=300, P50=100ms | probes=80, P50=479ms |

HNSW is 4.8x faster at the same recall level, but IVFFlat builds 7x faster and uses half the space. For workloads where build time or memory is the hard constraint, IVFFlat has a role. For everything else, HNSW is the better default — and it doesn't require periodic rebuilding as your data changes.

## The arithmetic argument

A common criticism is that pgvector's HNSW indexes require "10+ GB of RAM for a few million vectors." This is true — our 1M×960d index was 7.7 GB. But this is not a pgvector problem. It's an HNSW problem.

The HNSW algorithm stores, for each vector:
- The vector data (dimensions × bytes per element)
- Graph connectivity (~2 × m neighbor pointers at 8 bytes each)
- Internal overhead

For 1M vectors at 960 dimensions with float32 and m=24, the arithmetic gives roughly 7 GB regardless of which database engine you use. We calibrated our sizing formula against measured values and found pgvector's actual overhead is about 1.75x the theoretical minimum for float32 vectors — resulting in predictions within 2–9% of measured index sizes.

A dedicated vector database needs the same memory for the same index. The difference is operational, not fundamental: pgvector requires you to configure `shared_buffers` and think about cache warming, while managed vector databases handle memory allocation internally.

## What pgvector actually gets right

### Transactional consistency

With pgvector, vectors and metadata live in the same transaction:

```sql
INSERT INTO products (name, price, embedding)
VALUES ('Widget', 29.99, $1);
-- Atomic. No sync layer. No eventual consistency.
```

With a separate vector database, every write is a distributed transaction:

```python
# What if step 2 fails?
db.execute("INSERT INTO products (id, name, price) VALUES (%s, %s, %s)", ...)
vector_db.upsert(id=product_id, vector=embedding)
```

Every failure between the relational write and the vector write creates an inconsistency that requires background reconciliation to fix. This is a real operational cost that "just use a dedicated vector DB" arguments rarely acknowledge.

### SQL composability

Vector search composes with the full power of SQL:

```sql
SELECT p.name, p.price, p.embedding <=> $1 AS distance
FROM products p
JOIN categories c ON p.category_id = c.id
WHERE c.name = 'electronics'
  AND p.price BETWEEN 50 AND 200
  AND p.in_stock = true
ORDER BY p.embedding <=> $1
LIMIT 10;
```

JOINs, WHERE filters, and vector similarity in one query, one round trip. With a separate vector database, you'd search for IDs, then query PostgreSQL for the metadata, handling pagination and consistency across two systems.

### Operational simplicity

One database means one backup strategy, one monitoring stack, one set of credentials, one connection pool, one failure domain, and one team that understands the system. For organizations that already run PostgreSQL — which is most of them — adding pgvector is adding an extension, not adding a service.

## What pgvector gets wrong (and how to deal with it)

We're not going to pretend pgvector is perfect. Here are the real limitations and what you can do about them.

### Index builds are expensive

Building an HNSW index on 1M×3072d vectors took 50 minutes on a 16-vCPU instance with 4 parallel workers. A smaller instance (2 vCPU, 8 GB) couldn't complete the build at all.

**Mitigations:** Use `CREATE INDEX CONCURRENTLY` to avoid blocking writes during builds. Set `maintenance_work_mem` and `max_parallel_maintenance_workers` appropriately. For large indexes, build on a replica and promote.

### Filtered search requires care

Combining `WHERE` clauses with vector `ORDER BY` can produce unexpected results — a post-filter on the top-K results may return fewer rows than requested.

**Mitigations:** pgvector supports iterative scan (`SET hnsw.iterative_scan = relaxed_order`) which re-scans the HNSW graph with increasing depth until enough filtered results are found. For heavily-filtered workloads, consider partial indexes on the filter columns.

### Hybrid search is DIY

pgvector doesn't have built-in Reciprocal Rank Fusion for combining vector and full-text search. You need to implement it in SQL.

**Mitigation:** It's about 10 lines of SQL using CTEs and `ROW_NUMBER()`. Not ideal, but workable. We provide an example in our [field guide](https://congenial-umbrella-ashy.vercel.app/field-guide.html).

### No vector-specific monitoring

PostgreSQL's built-in monitoring doesn't distinguish vector I/O from relational I/O. There's no recall metric, no index quality dashboard.

**Mitigation:** Use `EXPLAIN (ANALYZE, BUFFERS)` and check for `shared read=0`. Periodically benchmark recall with a test query set against brute-force search.

## When to use pgvector

pgvector is the right choice when:

1. **Vectors are a feature, not the product.** Your app has authentication, relational data, business logic — and also does vector search. This is the vast majority of real use cases.

2. **You need transactional consistency** between vectors and metadata without building a sync layer.

3. **You already run PostgreSQL.** Adding an extension is simpler than adding a service.

4. **Your vector count is under 10M** and QPS is under 1,000. This covers >99% of production workloads we see on Supabase.

## When to consider a dedicated vector database

1. **Billions of vectors** requiring horizontal sharding across nodes.

2. **Vector search is your entire product**, not a feature within a larger application, and you need specialized operational tooling.

3. **You need real-time adaptive indexing** with automatic parameter tuning and your team doesn't have PostgreSQL expertise.

## Tools we built

This analysis produced a set of open-source tools for sizing pgvector deployments:

- **[Sizing Wizard](https://congenial-umbrella-ashy.vercel.app/sizer.html)** — input your workload parameters, get a Supabase compute tier, PostgreSQL configuration, and ready-to-run SQL
- **[Architecture Field Guide](https://congenial-umbrella-ashy.vercel.app/field-guide.html)** — 15-stage decision-tree reference with empirical benchmark data
- **[Benchmark Pipeline](https://github.com/dventimisupabase/congenial-umbrella)** — Python script for measuring recall and latency against your own data

The sizing formulas are calibrated against our 1M-vector benchmarks (index size predictions within 2–9% of measured, CPU costs from EXPLAIN ANALYZE). The field guide includes the full recall curves, HNSW vs IVFFlat comparisons, and cold-vs-warm cache measurements presented in this post.

---

*All benchmark data in this post was measured on Supabase Cloud (4XL: 16 vCPU, 64 GB RAM, shared_buffers=20 GB) running PostgreSQL 17.6 with pgvector. The benchmark pipeline, sizing tools, and field guide are open source at [github.com/dventimisupabase/congenial-umbrella](https://github.com/dventimisupabase/congenial-umbrella).*
