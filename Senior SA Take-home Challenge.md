# **Take-Home Architecture Brief**

## *Senior Solutions Architect*

# **Context**

You are working with a customer that has two distinct internal teams deploying vector search. Both teams have datasets of exactly 1 Million vectors, but they use different embedding models. The VP of Platform Engineering has asked you to design a cluster for each dataset that meets strict throughput and recall requirements while absolutely minimizing infrastructure costs. Below are the requirements.

## **Scenario 1 \- The Search Team**

Using [OpenAI dbpedia dataset](https://huggingface.co/datasets/Supabase/dbpedia-openai-3-large-1M) (3072 dimensions, neural-trained text embeddings)

* Peak QPS: 1000
* Writes: Streaming, 200 per second peak, Average 100
* P99 SLA: 50ms
* Recall target: 95%
* Top k: 100

## **Scenario 2 \- The Data Science Team**

Using the [gist-960 datase](https://huggingface.co/datasets/open-vdb/gist-960-euclidean)t (960 dimensions, computer vision feature descriptors)

* Peak QPS: 3000 during a 6 hour time window midday
* Peak writes: Nightly batches of 100k, no concurrent reads
* P99 SLA: 500ms
* Recall target: 99%
* Top k: 10

## **Instructions**

Please spend no more than **3 \- 4hours on this**. We are evaluating the clarity of your systems engineering logic and your architectural intuition, not your ability to write boilerplate code from scratch.

**Use AI.** We strongly encourage you to use Cursor, GitHub Copilot, ChatGPT, Claude, or whatever AI tooling you prefer. In the field, we expect you to move fast and solve complex problems efficiently. Use AI to write your ingestion loops and format your scripts. We care about what you build and why you configured it that way, not your memorization of the Python SDK.

### **Expected Deliverables**

#### **Architectural Write-Up**

Provide the exact cluster sizing (RAM, vCPUs, Disk type/size) you would provision for both scenarios.

Define the specific Qdrant collection configurations (Storage type, Quantization strategy, HNSW parameters like m and ef\_construct, etc). Prove mathematically why the setups work for the respective SLAs.

#### **GitHub Repository**

We want you to actually upload the data and prove your setup works. You do not need to start from scratch.

You can use this [pipeline script](https://github.com/brian-ogrady/qdrant-hybrid-pipeline-example/blob/main/src/scripts/wiki_cohere.py) as a baseline for how to structure a data upload to Qdrant.

Review the [Qdrant Alloy repository](https://github.com/qdrant/qdrant-alloy) to see how we handle search execution.

Adapt these references to download and ingest the OpenAI dbpedia and gist-960 datasets. You must write the code to create the collections using your optimized configurations, execute the uploads, and run sample searches.

**Note**: While scenario 1 requires streaming writes and scenario 2 is a nightly batch, your ingestion code does not need to reflect these different realities, but you must defend your architecture choices.

#### **Executive Summary**

The VP of Platform Engineering has looked at the raw dimensions and made a firm assumption: "Scenario 2 will obviously require significantly less memory and compute because 960 dimensions is much smaller than 3072 dimensions."

Validate this thesis against your architectures. Write a summary email explaining your findings.

## **Evaluation Criteria**

* **Infrastructure Economics & Cluster Sizing**: Your ability to mathematically justify your hardware choices (RAM, vCPU, Disk). We are looking for the optimal use of memory-mapped storage (mmap) and quantization to hit strict SLAs without over-provisioning expensive cloud resources.
* **Algorithmic Intuition**: We expect you to articulate why the OpenAI embeddings may have a different RAM footprint than the gist-960 dataset despite having the same number of embeddings.
* **Workload-Specific Tuning**: How well your architecture accounts for the differing write/read profiles.
* **Executive Communication**: The clarity, authority, and commercial awareness of your VP summary email. Can you gracefully correct an executive's flawed technical assumption and translate complex graph traversal mechanics into Total Cost of Ownership (TCO)?
