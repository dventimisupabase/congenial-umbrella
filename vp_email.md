**To:** [VP of Platform Engineering]
**From:** [Solutions Architect]
**Subject:** Vector Search Sizing Analysis — Surprising Parity Between Your Two Scenarios

---

Hi [Name],

Thanks for the conversation about your two Qdrant deployments. I wanted to follow up with the sizing analysis, because the results challenged my own initial assumptions as well.

**Your intuition is sound at the raw data level.** A million 960-dimensional vectors in float32 is about 3.8 GB, while a million 3072-dimensional vectors is 12.3 GB — more than 3x larger. If raw vector size were the only cost driver, Scenario 2 would indeed be the lighter workload.

However, three factors combine to invert that relationship in production.

**1. Recall targets determine whether quantization is viable.** Scenario 1 (3072d OpenAI embeddings, 95% recall) is an excellent candidate for scalar quantization, which compresses each float32 component down to int8 — a 4x reduction. That brings working vector RAM from ~12.3 GB down to roughly 2.9 GB. Scenario 2 (960d GIST features, 99% recall) cannot safely use the same technique. At 99% recall on classical CV features, quantization introduces too much approximation error, so we must keep full float32 precision — all 3.7 GB of it. The smaller embedding actually requires *more* RAM in practice.

**2. Embedding type affects quantization tolerance.** Modern transformer-based embeddings like OpenAI's tend to be robust to quantization; the information is distributed across many dimensions and tolerates rounding. Classical dense features (like GIST descriptors) concentrate information more tightly, making them less forgiving when precision is reduced.

**3. QPS drives compute independently of vector size.** Scenario 2 targets 3,000 QPS versus 1,000 QPS for Scenario 1. Even though Scenario 2 has a more relaxed latency budget (500ms vs. 50ms P99), the threefold throughput requirement adds compute pressure that offsets the smaller search space.

**The bottom line in dollars:** both scenarios land at approximately $500/month. They are, for practical purposes, identical in cost — not "significantly less" for Scenario 2.

| | Scenario 1 (OpenAI 3072d) | Scenario 2 (GIST 960d) |
|---|---|---|
| Raw vector footprint | 12.3 GB | 3.8 GB |
| Working RAM (after quantization decisions) | ~2.9 GB | ~3.7 GB |
| Estimated monthly cost | ~$500 | ~$499 |

**Where I see optimization opportunities going forward:**

- **HA replicas.** If either workload requires high availability, factor in replica overhead early rather than retrofitting later.
- **Autoscaling for burst windows.** If the 3,000 QPS target in Scenario 2 is peak rather than sustained, autoscaling could meaningfully reduce baseline cost.
- **Binary quantization exploration.** For Scenario 1, binary quantization (1-bit) could push RAM even lower if your team can tolerate a rescoring step. Worth benchmarking.

I'd recommend we walk through these numbers together with both teams so everyone is calibrating on the same cost model. Happy to set up a 30-minute session whenever works for you.

Best,
[Your Name]
