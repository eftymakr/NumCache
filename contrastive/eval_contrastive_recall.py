"""
Re-run contrastive retrieval fresh using OUR retrieval_results/projection_heads.pt
+ retrieval_results/pooled_kv_combined.pt across all 3 QA types (chunk_based,
tracking, comparison). Computes hit@k and recall@k for k in {1,3,5,10}.

Run on a single GPU (CUDA_VISIBLE_DEVICES=N).

Outputs:
  retrieval_results/evals_our_heads/eval_chunk_based.json
  retrieval_results/evals_our_heads/eval_tracking.json
  retrieval_results/evals_our_heads/eval_comparison.json
  retrieval_results/evals_our_heads/summary.json
"""
import json, os, sys, time
from pathlib import Path
import torch
from transformers import AutoTokenizer

sys.path.insert(0, '/home/eftychia/Financial-QA-Benchmark-with-KV-cache')

# Use our existing ContrastiveRetriever
from inference_with_contrastive_retrieval import ContrastiveRetriever
from cartridges.models import FlexQwen3ForCausalLM

OUT_DIR = Path('retrieval_results/evals_our_heads')
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADS = '/home/eftychia/Financial-QA-Benchmark-with-KV-cache/retrieval_results/projection_heads.pt'
POOLED_KV = '/home/eftychia/Financial-QA-Benchmark-with-KV-cache/retrieval_results/pooled_kv_combined.pt'
MODEL_NAME = 'Qwen/Qwen3-4b'

QA_FILES = [
    ('chunk_based', 'qa/chunk_based_qa_VLO_PSX.json', 'doc_id'),
    ('tracking',    'qa/tracking_qa_VLO_PSX.json',  'doc_ids'),
    ('comparison',  'qa/company_comparison_VLO_vs_PSX.json',  'doc_ids'),
]

K_VALUES = [1, 3, 5, 10]

print(f'Loading tokenizer + Qwen3-4b model ({MODEL_NAME})...')
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = FlexQwen3ForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)
model.eval()
print(f'  loaded in {time.time()-t0:.0f}s')

print(f'Building ContrastiveRetriever (heads={HEADS}, pooled_kv={POOLED_KV})...')
# Pass a dummy cache_dir we won't use (we only call retrieve_top_k)
retr = ContrastiveRetriever(
    tokenizer=tokenizer, model=model,
    pooled_kv_path=POOLED_KV, heads_path=HEADS,
    cache_dir='/tmp/dummy_cache_dir',
)
print(f'  ready: {len(retr.doc_ids)} docs in pool')

summary = {}
for name, qa_file, gold_key in QA_FILES:
    print(f'\n=== Evaluating: {name} ===')
    qa = json.load(open(qa_file))
    print(f'  {len(qa)} QAs')

    per_query = []
    hit_counts  = {k: 0 for k in K_VALUES}
    recall_sums = {k: 0.0 for k in K_VALUES}
    n = 0
    t0 = time.time()
    for i, x in enumerate(qa):
        q = x['question']
        gold = [x['doc_id']] if gold_key == 'doc_id' else (x.get('doc_ids') or [])
        gold = set(g for g in gold if g)
        if not gold: continue
        with torch.no_grad():
            top = retr.retrieve_top_k(q, k=max(K_VALUES))
        # top = list of (doc_id, score) tuples
        retrieved_ids = [d for d, _ in top]
        n += 1
        for k in K_VALUES:
            topk_set = set(retrieved_ids[:k])
            hits = len(topk_set & gold)
            if hits > 0: hit_counts[k] += 1
            recall_sums[k] += hits / max(1, len(gold))
        per_query.append({
            'q_id': x.get('q_id'),
            'question': q,
            'gold': sorted(gold),
            'retrieved': [{'doc_id': d, 'score': float(s)} for d, s in top],
        })
        if (i+1) % 25 == 0:
            print(f'  [{i+1}/{len(qa)}] elapsed {time.time()-t0:.0f}s')
    metrics = {'n': n}
    for k in K_VALUES:
        metrics[f'hit@{k}']    = hit_counts[k] / n if n else 0
        metrics[f'recall@{k}'] = recall_sums[k] / n if n else 0
    print(f'  metrics: {metrics}')
    summary[name] = metrics
    out_path = OUT_DIR / f'eval_{name}.json'
    json.dump({'metrics': metrics, 'per_query': per_query}, open(out_path,'w'))
    print(f'  saved → {out_path}')

print(f'\n=== SUMMARY ===')
print(f'{"set":<15}{"n":>5} | {"H@1":>6}{"H@3":>6}{"H@5":>6}{"H@10":>6} | {"R@1":>6}{"R@3":>6}{"R@5":>6}{"R@10":>6}')
for name, m in summary.items():
    print(f'  {name:<13}{m["n"]:>5} | {100*m["hit@1"]:>6.1f}{100*m["hit@3"]:>6.1f}{100*m["hit@5"]:>6.1f}{100*m["hit@10"]:>6.1f} | {100*m["recall@1"]:>6.1f}{100*m["recall@3"]:>6.1f}{100*m["recall@5"]:>6.1f}{100*m["recall@10"]:>6.1f}')

json.dump(summary, open(OUT_DIR / 'summary.json','w'), indent=2)
print(f'\nSaved summary → {OUT_DIR / "summary.json"}')
