"""
Cache inference for the 43-ticker 414 QA generalizability eval.
Searches 4 cache dirs: combined, eclt_trained, merged, 43tickers_v1, p_compression.
Uses training prompt via generate_answer from /tmp/inference_with_cache_training_prompt.py.
"""
import torch, sys, os, json, glob, argparse

sys.path.insert(0, '/home/eftychia/Financial-QA-Benchmark-with-KV-cache/cartridges')
sys.path.insert(0, '/home/eftychia/Financial-QA-Benchmark-with-KV-cache')
os.environ['CARTRIDGES_DIR'] = '/home/eftychia/Financial-QA-Benchmark-with-KV-cache/cartridges'
os.environ['CARTRIDGES_OUTPUT_DIR'] = '/home/eftychia/Financial-QA-Benchmark-with-KV-cache/outputs'

from cartridges.cache import TrainableCache, AttnConfig
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from transformers import AutoTokenizer

sys.path.insert(0, '/tmp')
from inference_with_cache_training_prompt import concatenate_caches, generate_answer as _gen

parser = argparse.ArgumentParser()
parser.add_argument('--qa_file', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

BASE = '/home/eftychia/Financial-QA-Benchmark-with-KV-cache'

print(f"Loading model on GPU {args.gpu}...")
model = FlexQwen3ForCausalLM.from_pretrained("Qwen/Qwen3-4b", torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4b")
model.eval()

CACHE_DIRS_GLOBS = [
    # combined (this repo's local caches): doc_id/train/**/cache_last.pt
    # combined dropped (subset of merged)
    # combined dropped (subset of merged)
    # eclt_trained
    # eclt_trained dropped (out of scope for chunk_based)
    # 43-ticker v1 (NEW): doc_id_<timestamp>/training_output/<date>/<uuid>/cache_last.pt
    f'/ext/eftychia/automated_runs_43tickers_v1/{{}}_*/training_output/*/*/cache_last.pt',
    # p-compression
    # p_compression dropped (first-p init, not NumCache)
    # merged (peiwen's)
    f'/ext/peiwenfiles/automated_runs_merged/{{}}_*/training_output/*/*/cache_last.pt',
]


def load_cache(doc_id):
    for pat in CACHE_DIRS_GLOBS:
        files = glob.glob(pat.format(doc_id), recursive=True)
        if files:
            try:
                c = TrainableCache.from_pretrained(files[-1], device="cuda").to("cuda")
                return c, files[-1]
            except Exception as e:
                print(f"  load fail for {doc_id} at {files[-1]}: {e}")
                continue
    return None, None


with open(args.qa_file) as f:
    qa_data = json.load(f)

print(f"Loaded {len(qa_data)} QAs from {args.qa_file}")

results = []
missing_cache = 0
errors = 0
for i, qa in enumerate(qa_data):
    qid = qa.get('q_id', qa.get('qid', ''))
    question = qa.get('question', '')
    doc_ids = qa.get('doc_ids', [])
    if not doc_ids and 'doc_id' in qa:
        doc_ids = [qa['doc_id']]

    caches, paths, miss = [], [], []
    for did in doc_ids:
        c, p = load_cache(did)
        if c is not None:
            caches.append(c); paths.append(p)
        else:
            miss.append(did)

    if not caches:
        missing_cache += 1
        print(f"[{i+1}/{len(qa_data)}] {qid}: NO CACHES for {doc_ids}, skipping")
        continue

    combined = concatenate_caches(caches) if len(caches) > 1 else caches[0]
    try:
        answer, _ = _gen(model, combined, tokenizer, question, max_new_tokens=512)
    except Exception as e:
        errors += 1
        print(f"[{i+1}/{len(qa_data)}] {qid}: ERROR {str(e)[:120]}")
        continue

    results.append({
        'qid': qid, 'q_id': qid, 'question': question,
        'generated_answer': answer,
        'ground_truth': qa.get('answer', qa.get('ground_truth', '')),
        'key_points': qa.get('key_points', []),
        'doc_ids': doc_ids,
        'n_caches': len(caches),
        'missing_caches': miss,
        'ticker': qa.get('ticker', ''),
    })
    if (i + 1) % 20 == 0 or i < 3:
        print(f"[{i+1}/{len(qa_data)}] {qid}: {len(answer)} chars ({len(caches)} caches, missing={miss})")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    torch.cuda.empty_cache()

with open(args.output, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(results)} results, {missing_cache} skipped (no cache), {errors} errors -> {args.output}")
