# NumCache — KV Cache Compression and Retrieval for Financial Document QA

Reference implementation accompanying our paper *NumCache: KV Cache Compression and Retrieval for Financial Document QA* (KDD '26). This repository contains the **core training, retrieval, and inference code** used to reproduce the Fin-RATE results, plus the text-RAG baselines we compare against.

NumCache compresses each SEC filing into a number-preserving KV cache (4× compression by default), trains the cache directly on financial QA pairs, and at query time retrieves the most relevant caches via a contrastive retriever that operates in KV-cache space.

## Repository layout

```
.
├── numcache_init/
│   ├── cartridges_initialization_text.py     # Algorithm 1: number-preserving cache init
│   └── automated_training_pipeline.py        # End-to-end cache training (NumCache + first-p baseline)
├── contrastive/
│   ├── prepare_pooled_kv.py                  # Build pooled-KV .pt files from trained caches
│   ├── train_static_pool.py                  # Contrastive retriever — static-pool variant (Table 2)
│   ├── train_mlp_pool.py                     # Contrastive retriever — query-adaptive MLP-pool variant (Appendix E)
│   ├── mlp_retrieve_topk.py                  # Score a trained retriever to produce top-K doc lists
│   ├── eval_contrastive_recall.py            # Recall@K / MRR evaluation
│   ├── inference_with_contrastive_retrieval.py  # Sec. 3.3 inference: retrieve + cache concat + generate
│   └── inference_with_cache.py               # Helper: cache loading + concatenation + generation
├── inference/
│   ├── baseline_fullcontext.py               # Full-context baseline (Qwen3-4B over raw filing text)
│   ├── baseline_with_correct_cache.py        # NumCache gold-cache inference (top-1 oracle)
│   ├── text_baseline_vllm.py                 # Text RAG baseline via vLLM (Qwen3-4B + retriever)
│   └── gen_gpt51.py                          # GPT-5.1 baseline (Responses API)
└── retrieval/
    ├── retrieve_doc.py                       # Unified BM25 / Elasticsearch / Vector (±reranker) runner
    ├── bm25_number_aware_v2.py               # NumBM25 (number-aware BM25)
    ├── retrieval_methods/                    # Per-backend implementations
    ├── setup_elasticsearch_index.py          # Build ES index from corpus.jsonl
    └── run_all_retrieval_methods.sh          # Wrapper to run all text retrievers
```

## Paper-to-code map

| Paper component | Code |
| --- | --- |
| Sec. 3.1.1 / Algorithm 1 — number-preserving initialization | `numcache_init/cartridges_initialization_text.py` |
| Sec. 3.1.3 — cache training (CE loss) | `numcache_init/automated_training_pipeline.py` |
| Sec. 3.2 — contrastive retrieval (dual heads, static pooling) | `contrastive/prepare_pooled_kv.py` + `contrastive/train_static_pool.py` |
| Sec. 3.2.2 / Appendix E — query-adaptive layer pooling | `contrastive/train_mlp_pool.py` |
| Sec. 3.2 — Recall@K / MRR evaluation | `contrastive/eval_contrastive_recall.py` |
| Sec. 3.3 — inference (cache load + concat + generation) | `contrastive/inference_with_contrastive_retrieval.py`, `inference/baseline_with_correct_cache.py` |
| Table 2 — Contrastive KV Retrieval row (R@1 = 68.3 on DR-QA) | `contrastive/train_static_pool.py` + `eval_contrastive_recall.py` |
| Table 3 — Full Context baselines (Qwen3-4B / GPT-5.1) | `inference/baseline_fullcontext.py`, `inference/gen_gpt51.py` |
| Table 3 — Qwen3-4B / GPT-5.1 + text RAG | `inference/text_baseline_vllm.py`, `inference/gen_gpt51.py` |
| Table 2 — text retrievers (BM25, BM25+R, ES, ES+R, Vector ±R) | `retrieval/retrieve_doc.py` |
| Table 2 — NumBM25 | `retrieval/bm25_number_aware_v2.py` |
| Table 6 / 7 — first-p initialization ablation | `automated_training_pipeline.py --init-method pinit` |

## Setup

```bash
pip install -r requirements.txt
# Elasticsearch 7.17.x (for ES / hybrid backends)
# CUDA-capable GPU (RTX A6000 48 GB used in the paper) for cache training,
# contrastive retrieval, and vLLM
```

### Environment variables

All scripts default to repo-relative paths. Override these only if your data lives elsewhere:

| Variable | Default | What it points to |
| --- | --- | --- |
| `CARTRIDGES_DIR` | *(none — must be set)* | Directory containing the `cartridges/` Python package (the internal Fin-RATE training package, not on PyPI). Required for any cache-loading script. |
| `NUMCACHE_REPO_ROOT` | repo root | Used by scripts that resolve sibling data files. |
| `NUMCACHE_CACHE_DIR` | `<repo>/trained_caches` | Directory of trained per-doc KV caches (one subdir per `doc_id`, each containing `cache-step*.pt`). |
| `NUMCACHE_QA_DIR` | `<repo>/qa` | Directory containing the Fin-RATE QA JSONs. |
| `NUMCACHE_HEADS_PATH` | `<repo>/retrieval_results/projection_heads.pt` | Path to trained contrastive projection heads. |
| `NUMCACHE_POOLED_KV_PATH` | `<repo>/retrieval_results/pooled_kv_combined.pt` | Path to pooled KV vectors over the corpus. |
| `NUMCACHE_BASE_DIR` | repo root | Used by the training pipeline to find the cartridges workspace + corpus. |
| `NUMCACHE_PYTHON` | `python` | Interpreter used by the training pipeline to launch sub-runs. |
| `OPENAI_API_KEY` | — | GPT-5.1 baseline. |
| `AZURE_OPENAI_API_KEY` | — | GPT-4.1 LLM-as-judge (not included in this repo). |

## Quick start

### 1. Build the Elasticsearch index

```bash
python retrieval/setup_elasticsearch_index.py \
  --corpus_path corpus.jsonl --index_name financial_corpus
```

### 2. Run text retrievers

```bash
# BM25 / Elasticsearch / Vector (±reranker)
python retrieval/retrieve_doc.py \
  --qa_path qa/chunk_based_qa_VLO_PSX.json \
  --corpus_path corpus.jsonl \
  --output_dir retrieval_results/ \
  --retrieval_method bm25 \
  --top_k 10

# NumBM25 (number-aware BM25)
python retrieval/bm25_number_aware_v2.py \
  --corpus pool_826 \
  --qa_file qa/chunk_based_qa_VLO_PSX.json
```

### 3. Train NumCache caches

```bash
# Number-preserving initialization + CE training (paper default, 4× compression)
python numcache_init/automated_training_pipeline.py \
  --doc-id doc_000000 --compression-ratio 4 --init-method numcache

# First-p baseline (Tables 6 / 7)
python numcache_init/automated_training_pipeline.py \
  --doc-id doc_000000 --compression-ratio 4 --init-method pinit
```

### 4. Train the contrastive retriever

The contrastive retriever (Sec. 3.2) needs (a) pooled KV vectors built once from your trained caches, and (b) the projection-head training itself.

```bash
# (a) Pool KV vectors from the trained caches into a single .pt file.
python contrastive/prepare_pooled_kv.py \
  --runs-root $NUMCACHE_CACHE_DIR \
  --train-json qa/train_all.json

# (b-i) Static-pool projection heads (paper Table 2, headline R@1 = 68.3)
python contrastive/train_static_pool.py \
  --pooled-kv retrieval_results/pooled_kv_combined.pt \
  --qa-json qa/chunk_based_qa_VLO_PSX.json \
  --out-dir retrieval_results/static_pool

# (b-ii) Query-adaptive MLP-pool variant (Appendix E ablation)
python contrastive/train_mlp_pool.py \
  --pooled-kv retrieval_results/pooled_kv_with_layers.pt \
  --qa-json qa/chunk_based_qa_VLO_PSX.json \
  --out-dir retrieval_results/mlp_pool
```

### 5. Evaluate retriever recall + run cache-concat inference

```bash
# Recall@K / MRR (reproduces Table 2)
python contrastive/eval_contrastive_recall.py

# Score the retriever to produce {qid: [doc_id, ...]} JSONs
python contrastive/mlp_retrieve_topk.py \
  --heads retrieval_results/static_pool/projection_heads.pt \
  --pooled-kv retrieval_results/pooled_kv_combined.pt \
  --out-dir retrieval_results/topk_for_baselines

# End-to-end: retrieve top-K caches + concatenate + generate
python contrastive/inference_with_contrastive_retrieval.py \
  --pooled_kv_path retrieval_results/pooled_kv_combined.pt \
  --heads_path    retrieval_results/static_pool/projection_heads.pt \
  --cache_dir     $NUMCACHE_CACHE_DIR \
  --qa_pairs_file qa/chunk_based_qa_VLO_PSX.json \
  --top_k 5 --batch_mode --output_file results.json
```

### 6. Text RAG via vLLM (Qwen3-4B baseline)

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4b --port 8200 --max-model-len 32768

python inference/text_baseline_vllm.py \
  --retrieval_file retrieval_results/bm25/.../retrieved_doc_ids.json \
  --qa_file qa/chunk_based_qa_VLO_PSX.json \
  --output bm25_topk5_results.json --top_k 5
```

### 7. GPT-5.1 baseline

```bash
export OPENAI_API_KEY=sk-...
# Full-context
python inference/gen_gpt51.py --mode fullctx --qa_file qa/chunk_based_qa_VLO_PSX.json
# Top-K text RAG
python inference/gen_gpt51.py --mode topk5 --retrieval_file <retrieval.json>
```

## Inference prompt

NumCache inference uses the minimal prompt described in Sec. 3.3 and Appendix F.1 of the paper:

```
System: You are a helpful assistant. Based on the knowledge base, answer questions accurately.
User: {question}
```

The trained KV cache already encodes the document context, so no document text is passed in the prompt — the model attends to the cached key/value pairs.

## Notes

- Trained cache files (`cache_last.pt`), projection heads (`projection_heads.pt`), and pooled-KV `.pt` files are produced by the training pipeline and are **not** included in this repo.
- The `cartridges/` Python package (the internal Fin-RATE training package) is not on PyPI; set `CARTRIDGES_DIR` to point at the directory containing it before running any cache-loading script.
- API keys are never committed. Set `OPENAI_API_KEY` (GPT-5.1) and `AZURE_OPENAI_API_KEY` (GPT-4.1 judge) via environment variables.

## Citation

```bibtex
@inproceedings{makri2026numcache,
  title     = {NumCache: KV Cache Compression and Retrieval for Financial Document QA},
  author    = {Makri, Eftychia and Li, Peiwen and Jiang, Yidong and Chen, Junrong and Chen, Jialin and Maatouk, Ali and Tassiulas, Leandros and Brenner, Eliot and Xiang, Bing and Ying, Rex},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year      = {2026},
}
```
