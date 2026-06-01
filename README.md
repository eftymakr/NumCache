# NumCache — Fin-RATE Retrieval & Inference Baselines

This repository contains the retrieval pipelines, contrastive KV retriever, and inference scripts used to produce the baseline results in the Fin-RATE benchmark paper. It covers:

- 9 **text retrievers** (BM25, BM25+R, NumBM25, Elasticsearch, Elasticsearch+R, Vector FinLang/MiniLM ±R)
- **Contrastive KV Retrieval** (static pool + MLP variants)
- **Inference scripts** for full-context, text-RAG, cache-concat, and GPT-5.1 baselines

## Layout

```
.
├── retrieval/
│   ├── retrieve_doc.py                  # Unified runner: BM25, ES, vector, hybrid + rerankers
│   ├── retrieval_methods/               # Per-method implementations
│   ├── bm25_number_aware_v2.py          # NumBM25 (number-aware BM25 via rank_bm25)
│   ├── setup_elasticsearch_index.py     # Build the ES index from corpus.jsonl
│   └── run_all_retrieval_methods.sh     # Wrapper that runs all methods
├── contrastive/
│   ├── inference_with_contrastive_retrieval.py  # Cartridges retriever + cache concat inference
│   ├── eval_contrastive_recall.py               # R@K / hit@K / MRR evaluation
│   └── mlp_retrieve_topk.py                     # MLP-pool retriever (top-K extraction)
├── numcache_init/
│   ├── automated_training_pipeline.py           # End-to-end NumCache training pipeline
│   ├── run_cartridges_numcache_init_training.py # NumCache initialization runner
│   ├── run_cartridges_numcache_ce_training.py   # NumCache + CE loss training
│   ├── run_cartridges_pinit_ce_training.py      # P-init (first-p tokens) baseline
│   ├── cartridges_initialization_text.py        # Text-based initialization (NumCache core)
│   ├── cartridges_initialization_attention_select.py  # Attention-select init variant
│   ├── cartridges_initialization_from_trimkv.py # TrimKV init variant
│   └── trimkv_init_caches.py                    # TrimKV cache initialization
└── inference/
    ├── baseline_fullcontext.py                  # Qwen3-4B full-context baseline (configurable prompt)
    ├── baseline_with_correct_cache.py           # Single-cache (NumCache gold) baseline
    ├── text_baseline_vllm.py                    # Text RAG via vLLM (concurrent)
    ├── inference_eclt_fullcontext.py            # Full-context for EC/LT (multi-doc gold concat)
    ├── inference_eclt_cache.py                  # NumCache concat for EC/LT
    ├── inference_43tickers_cache.py             # NumCache concat across 4 cache dirs
    ├── inference_with_cache_training_prompt.py  # Helper: generate_answer with training prompt
    ├── contrastive_cache_concat_inference.py    # Contrastive retriever + cache concat (training prompt)
    └── gen_gpt51.py                             # GPT-5.1 baseline (Responses API)
```

## NumCache initialization

The `numcache_init/` directory contains the code that trains and initializes the compressed KV caches used as the "cartridges" for cache-concat inference. The `automated_training_pipeline.py` accepts `--init-method {numcache, pinit}` to switch between number-aware compressed-text initialization (the headline NumCache method) and the first-p raw-tokens baseline (`pinit`). At 4x compression, NumCache replaces 5K input tokens with ~1.25K trained cache tokens while preserving numerical information.

## Prereqs

```bash
pip install -r requirements.txt
# Elasticsearch 7.17.x (for ES/hybrid backends)
# CUDA + a GPU for the contrastive/cache scripts and vLLM
```

## Quick start

### 1. Build the Elasticsearch index

```bash
python retrieval/setup_elasticsearch_index.py \
  --corpus_path corpus.jsonl --index_name financial_corpus
```

### 2. Run all text retrievers

```bash
python retrieval/retrieve_doc.py \
  --qa_path qa/chunk_based_qa_VLO_PSX.json \
  --corpus_path corpus.jsonl \
  --output_dir retrieval_results/ \
  --retrieval_method bm25            # or elasticsearch, vector, hybrid
  --top_k 10

# NumBM25 (number-aware)
python retrieval/bm25_number_aware_v2.py --corpus pool_826 --qa_file qa/chunk_based_qa_VLO_PSX.json
```

### 3. Contrastive KV retriever + cache concat inference

```bash
# Evaluate retriever recall
python contrastive/eval_contrastive_recall.py

# Generate answers using contrastive retrieval + cache concat
python contrastive/inference_with_contrastive_retrieval.py \
  --pooled_kv_path retrieval_results/pooled_kv_combined.pt \
  --heads_path    retrieval_results/projection_heads.pt \
  --cache_dir     automated_runs_combined /ext/peiwenfiles/automated_runs_merged \
  --qa_pairs_file qa/chunk_based_qa_VLO_PSX.json \
  --top_k 5 --batch_mode --output_file results.json
```

### 4. Text RAG via vLLM

```bash
# Start vLLM
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4b --port 8200 --max-model-len 32768

# Run text RAG
python inference/text_baseline_vllm.py \
  --retrieval_file retrieval_results/bm25/.../retrieved_doc_ids.json \
  --qa_file qa/chunk_based_qa_VLO_PSX.json \
  --output bm25_topk5_results.json --top_k 5
```

### 5. GPT-5.1 baseline

```bash
export OPENAI_API_KEY=sk-...
# Full-context, single gold doc per QA
python inference/gen_gpt51.py --mode fullctx --qa_file qa/chunk_based_qa_VLO_PSX.json
# Top-K retrieval text input
python inference/gen_gpt51.py --mode topk5 --retrieval_file <retrieval.json>
```

## System prompt

All recent inference scripts standardize on the simple training prompt:

```
"Please answer the user's question based on your knowledge."
```

Pre-2026 results that used `"You are a financial analyst expert. …"` or `"You are a helpful assistant. Based on the knowledge base, …"` are NOT comparable to the training-prompt numbers.

## Notes

- The `BASE_DIR` paths in some scripts are hardcoded to the original development location. Update before running.
- API keys are not committed — set `OPENAI_API_KEY` (for GPT-5.1) and `AZURE_OPENAI_API_KEY` (for the GPT-4.1 judge, not included here) via env vars.
- The cache files (`cache_last.pt`) and trained projection heads (`projection_heads.pt`) are NOT in this repo — they're produced by the training pipeline in the Fin-RATE benchmark side.
