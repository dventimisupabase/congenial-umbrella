**To:** [VP of Platform Engineering]
**From:** [Solutions Architect]
**Subject:** Vector Search Sizing Analysis — Surprising Parity Between Your Two Scenarios

---

Hi [Name],

Following up on the sizing analysis for your two Qdrant deployments. The results challenged my own initial assumptions as well.

**Your intuition is sound at the raw data level.** A million 960-dimensional vectors is about 3.8 GB, while a million 3072-dimensional vectors is 12.3 GB. If raw vector size were the only cost driver, Scenario 2 would indeed be lighter. However, three factors invert that relationship in production.

**1. Recall targets determine quantization eligibility.** Scenario 1 (95% recall) can use scalar quantization, compressing vectors 4x — from ~12.3 GB down to ~2.9 GB in RAM. Scenario 2 (99% recall on classical CV features) cannot safely quantize, so all 3.7 GB stays as full-precision float32. The smaller embedding needs more RAM.

**2. Embedding type affects quantization tolerance.** OpenAI embeddings distribute information broadly across dimensions and tolerate rounding. GIST descriptors concentrate information more tightly and do not.

**3. QPS drives compute independently of vector size.** Scenario 2 targets 3,000 QPS versus 1,000 for Scenario 1, adding compute pressure that offsets the smaller search space.

**The bottom line:** both scenarios land at approximately $500/month — identical in cost, not significantly less for Scenario 2. Our benchmarks confirm both architectures meet their SLA targets:

| | Scenario 1 (OpenAI 3072d) | Scenario 2 (GIST 960d) |
|---|---|---|
| Working RAM | ~2.9 GB (quantized) | ~3.7 GB (full precision) |
| Recall (measured) | 99.6% @100 (target: 95%) | 99.2% @10 (target: 99%) |
| Estimated monthly cost | ~$500 | ~$499 |

Happy to walk through these numbers with both teams. Let me know if a 30-minute session works.

Best,
[Your Name]
