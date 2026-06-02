# Contrastive KV Retrieval

This subdirectory contains the contrastive retriever described in Sec. 3.2 of the paper: a question-to-cache retriever trained on top of precomputed NumCache KV caches, used to select the top-K caches for cache-concat generation (Sec. 3.3).

## Method summary
- **Pool KV vectors** from the trained per-doc caches at `$NUMCACHE_CACHE_DIR/doc_*/cache-step*.pt`. For each cache and each retained transformer layer, mean-pool the keys across tokens and flatten across heads to get a single vector per layer per doc.
- **Static-pool variant** (paper Table 2 headline): average the last *L* layers into a single per-doc vector before training.
- **Query-adaptive MLP-pool variant** (Appendix E ablation): keep the per-layer vectors and learn a lightweight MLP that predicts layer weights from the question embedding, so different questions can attend to different layers.
- **Projection heads** (dual MLPs) align the question embedding with the pooled-cache vector. Trained with **InfoNCE** using in-batch negatives, temperature `τ = 0.05`, batch size 64, 100 epochs.
- **Evaluation** retrieves the top-K caches for each query over the full corpus and reports Recall@K and MRR.

## Files

| File | Purpose |
| --- | --- |
| `prepare_pooled_kv.py` | Walk `--runs-root` (default `$NUMCACHE_CACHE_DIR`) and produce a single `pooled_kv*.pt` with one vector per (doc, layer). Run this once after cache training. |
| `train_static_pool.py` | Train the static-pool projection heads (Sec. 3.2 / Table 2). |
| `train_mlp_pool.py` | Train the query-adaptive MLP-pool projection heads (Appendix E / Table 10). |
| `eval_contrastive_recall.py` | Compute Recall@K and MRR on DR-QA, EC-QA, and LT-QA. Reproduces the bottom row of Table 2. |
| `mlp_retrieve_topk.py` | Use trained heads to score the corpus per query and dump `{qid: [doc_id, ...]}` JSONs for the downstream baselines. |
| `inference_with_contrastive_retrieval.py` | End-to-end Sec. 3.3 inference: retrieve top-K caches, concatenate, generate. |
| `inference_with_cache.py` | Helper module — cache loading, `concatenate_caches`, `generate_answer`. Imported by the inference script. |

## Reproduce Table 2 (Contrastive KV Retrieval row)

The commands below assume the env vars in the top-level README are set: `CARTRIDGES_DIR`, `NUMCACHE_CACHE_DIR`, and that the Fin-RATE QA JSONs live in `$NUMCACHE_REPO_ROOT/qa/`.

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

## Reproduce Appendix E ablation (query-adaptive MLP pool)

```bash
# Train the MLP-pool variant (per-layer vectors + query-conditioned layer weights).
python contrastive/train_mlp_pool.py \
  --pooled-kv retrieval_results/pooled_kv_with_layers.pt \
  --qa-json qa/chunk_based_qa_VLO_PSX.json \
  --out-dir retrieval_results/mlp_pool
```

Table 10 reports MRR for **simple averaging** vs **query-adaptive** layer pooling — the same `eval_contrastive_recall.py` script can be pointed at the MLP-pool heads to reproduce the ablation row.

## Cache-concat inference (Sec. 3.3)

Once the retriever is trained, drive end-to-end QA with the contrastive inference script:

```bash
python contrastive/inference_with_contrastive_retrieval.py \
  --pooled_kv_path retrieval_results/pooled_kv_combined.pt \
  --heads_path retrieval_results/static_pool/projection_heads.pt \
  --cache_dir "$NUMCACHE_CACHE_DIR" \
  --qa_pairs_file qa/chunk_based_qa_VLO_PSX.json \
  --top_k 5 --batch_mode --output_file results.json
```

For top-1 single-cache inference (no concat), use `inference/baseline_with_correct_cache.py` against the gold doc ID instead.
