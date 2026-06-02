# Contrastive KV Retrieval

Contrastive question-to-cache retriever trained on top of precomputed NumCache KV caches. Produces per-query top-K cache rankings that can drive any downstream consumer (cache-concat inference, text RAG, evaluation).

## Method summary
- **Pool KV vectors** from the trained per-doc caches at `$NUMCACHE_CACHE_DIR/doc_*/cache-step*.pt`. For each cache and each retained transformer layer, mean-pool the keys across tokens and flatten across heads to get a single vector per layer per doc.
- **Static-pool variant**: average the last *L* layers into a single per-doc vector before training.
- **Query-adaptive MLP-pool variant**: keep the per-layer vectors and learn a lightweight MLP that predicts layer weights from the question embedding, so different questions can attend to different layers.
- **Projection heads** (dual MLPs) align the question embedding with the pooled-cache vector. Trained with **InfoNCE** using in-batch negatives, temperature `τ = 0.05`, batch size 64, 100 epochs.
- **Evaluation** retrieves the top-K caches for each query over the full corpus and reports Recall@K and MRR.

## Files

| File | Purpose |
| --- | --- |
| `prepare_pooled_kv.py` | Walk `--runs-root` (default `$NUMCACHE_CACHE_DIR`) and produce a single `pooled_kv*.pt` with one vector per (doc, layer). Run this once after cache training. |
| `train_static_pool.py` | Train the static-pool projection heads. |
| `train_mlp_pool.py` | Train the query-adaptive MLP-pool projection heads. |
| `eval_contrastive_recall.py` | Compute Recall@K and MRR on DR-QA, EC-QA, and LT-QA. |
| `mlp_retrieve_topk.py` | Use trained heads to score the corpus per query and dump `{qid: [doc_id, ...]}` JSONs for the downstream baselines. |


## Quick start

```bash
# 1. Build pooled-KV vectors from the trained caches (one-off).
python contrastive/prepare_pooled_kv.py \
  --runs-root "$NUMCACHE_CACHE_DIR" \
  --train-json qa/train_all.json \
  --out retrieval_results/pooled_kv_combined.pt

# 2. Train the static-pool projection heads.
python contrastive/train_static_pool.py \
  --pooled-kv retrieval_results/pooled_kv_combined.pt \
  --qa-json qa/chunk_based_qa_VLO_PSX.json \
  --out-dir retrieval_results/static_pool

# 3. Compute Recall@K + MRR on DR-QA, EC-QA, LT-QA.
NUMCACHE_HEADS_PATH=retrieval_results/static_pool/projection_heads.pt \
NUMCACHE_POOLED_KV_PATH=retrieval_results/pooled_kv_combined.pt \
python contrastive/eval_contrastive_recall.py
```

Output (`retrieval_results/evals_our_heads/`):
- `eval_chunk_based.json`, `eval_tracking.json`, `eval_comparison.json` — per-query top-K + verdicts
- `summary.json` — aggregate Recall@K and MRR

## MLP-pool (query-adaptive) variant

```bash
python contrastive/train_mlp_pool.py \
  --pooled-kv retrieval_results/pooled_kv_with_layers.pt \
  --qa-json qa/chunk_based_qa_VLO_PSX.json \
  --out-dir retrieval_results/mlp_pool
```

Point `eval_contrastive_recall.py` at the MLP-pool heads to evaluate this variant.

## Producing top-K JSONs for downstream consumers

```bash
python contrastive/mlp_retrieve_topk.py \
  --heads retrieval_results/static_pool/projection_heads.pt \
  --pooled-kv retrieval_results/pooled_kv_combined.pt \
  --out-dir retrieval_results/topk_for_baselines
```

This script reads only the trained heads + pooled KV vectors — no cartridges dependency.
