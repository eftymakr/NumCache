# NumCache — KV Cache Compression and Retrieval for Financial Document QA

NumCache compresses each SEC filing into a number-preserving KV cache (4× compression by default), trains the cache directly on financial QA pairs, and at query time retrieves the most relevant caches via a contrastive retriever that operates in KV-cache space.

This repository contains the retrieval baselines (text retrievers + the contrastive KV-cache retriever) and the number-preserving text compressor used at cache-initialization time.

## Repository layout

```
.
├── numcache_init/      Number-preserving text compressor used at cache init
├── contrastive/        Contrastive retriever over pooled KV-cache vectors (train + eval)
└── retrieval/          Text retrievers: BM25, Elasticsearch, vector, NumBM25
```

Each subfolder has its own README with per-script usage.

## Setup

```bash
pip install -r requirements.txt
# Elasticsearch 7.17.x is needed for the ES backend.
# CUDA-capable GPU (we used RTX A6000 48 GB) for contrastive training.
```

### Cartridges package (required for the contrastive retriever)

The contrastive train + eval scripts depend on the `cartridges` package for the **model classes** — specifically the patched `FlexQwen3ForCausalLM` (used to embed questions and to attend over injected KV caches at runtime) and the `TrainableCache` container around the trained KV tensors.

`cartridges` is not on PyPI — install it from the upstream repo:

```bash
git clone https://github.com/HazyResearch/cartridges /path/to/cartridges
export CARTRIDGES_DIR=/path/to/cartridges
```

The text retrievers in `retrieval/` and the number-preserving compressor in `numcache_init/` have **no cartridges dependency** — only the contrastive train/eval scripts need it.

### Environment variables

| Variable | What it points to |
| --- | --- |
| `CARTRIDGES_DIR` | Path to the cartridges package (see above). Required for `contrastive/{prepare_pooled_kv,train_static_pool,train_mlp_pool,eval_contrastive_recall}.py`. |
| `NUMCACHE_CACHE_DIR` | Directory of trained per-doc KV caches (defaults to `<repo>/trained_caches`). |

## Quick start

### 1. Text retrievers (no cartridges required)

```bash
python retrieval/setup_elasticsearch_index.py \
  --corpus_path corpus.jsonl --index_name financial_corpus

python retrieval/retrieve_doc.py \
  --qa_path qa/chunk_based_qa_VLO_PSX.json \
  --corpus_path corpus.jsonl \
  --output_dir retrieval_results/ \
  --retrieval_method bm25 --top_k 10

python retrieval/bm25_number_aware_v2.py \
  --corpus pool_826 \
  --qa_file qa/chunk_based_qa_VLO_PSX.json
```

### 2. Contrastive retriever (requires `CARTRIDGES_DIR`)

```bash
# Pool KV vectors from the trained caches into a single .pt file.
python contrastive/prepare_pooled_kv.py \
  --runs-root $NUMCACHE_CACHE_DIR \
  --train-json qa/train_all.json

# Static-pool projection heads
python contrastive/train_static_pool.py \
  --pooled-kv retrieval_results/pooled_kv_combined.pt \
  --qa-json qa/chunk_based_qa_VLO_PSX.json \
  --out-dir retrieval_results/static_pool

# Query-adaptive MLP-pool variant
python contrastive/train_mlp_pool.py \
  --pooled-kv retrieval_results/pooled_kv_with_layers.pt \
  --qa-json qa/chunk_based_qa_VLO_PSX.json \
  --out-dir retrieval_results/mlp_pool

# Recall@K + MRR
python contrastive/eval_contrastive_recall.py

# Score the retriever to dump {qid: [doc_id, ...]} JSONs for downstream use
python contrastive/mlp_retrieve_topk.py \
  --heads retrieval_results/static_pool/projection_heads.pt \
  --pooled-kv retrieval_results/pooled_kv_combined.pt \
  --out-dir retrieval_results/topk_for_baselines
```

### 3. Number-preserving compressor (utility, no cartridges required)

Used at cache-initialization time to compress a document down to a token budget while keeping all numeric facts intact:

```bash
python numcache_init/number_preserving_compressor.py \
  --document path/to/filing.txt \
  --budget 8192 \
  --output compressed.txt
```

## Notes

- Trained cache files, projection heads, and pooled-KV `.pt` files are produced by the contrastive training pipeline and are not committed.
- The `cartridges` package is not on PyPI. Set `CARTRIDGES_DIR` before running any contrastive train/eval script.
