"""
Cache inference for EC-QA (19) and LT-QA (97) with training prompt.
Concatenates caches for all gold doc_ids per question.
"""
import torch, sys, os, json, glob, argparse

sys.path.insert(0, '/home/eftychia/Financial-QA-Benchmark-with-KV-cache/cartridges')
os.environ['CARTRIDGES_DIR'] = '/home/eftychia/Financial-QA-Benchmark-with-KV-cache/cartridges'
os.environ['CARTRIDGES_OUTPUT_DIR'] = '/home/eftychia/Financial-QA-Benchmark-with-KV-cache/outputs'

from cartridges.cache import TrainableCache, AttnConfig
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from transformers import AutoTokenizer

sys.path.insert(0, '/tmp')
sys.path.insert(0, '/home/eftychia/Financial-QA-Benchmark-with-KV-cache')
from inference_with_cache_training_prompt import concatenate_caches, generate_answer as _gen

parser = argparse.ArgumentParser()
parser.add_argument('--qa_file', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

BASE = '/home/eftychia/Financial-QA-Benchmark-with-KV-cache'

print("Loading model...")
model = FlexQwen3ForCausalLM.from_pretrained("Qwen/Qwen3-4b", torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4b")
model.eval()

attn_config = AttnConfig(
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_key_value_heads,
    head_dim=model.config.head_dim,
)

with open(args.qa_file) as f:
    qa_data = json.load(f)

def load_cache(doc_id):
    """Load cache: combined first, then EC/LT trained, then merged."""
    for d in [f'{BASE}/automated_runs_combined/{doc_id}/train/**/cache_last.pt',
              f'/ext/eftychia/automated_runs_combined_eclt_trained/{doc_id}/train/**/cache_last.pt']:
        files = glob.glob(d, recursive=True)
        if files:
            return TrainableCache.from_pretrained(files[-1], device="cuda").to("cuda")
    files = glob.glob(f'/ext/peiwenfiles/automated_runs_merged/{doc_id}_*/training_output/*/*/cache_last.pt')
    if files:
        return TrainableCache.from_pretrained(files[-1], device="cuda").to("cuda")
    return None

results = []
for i, qa in enumerate(qa_data):
    qid = qa.get('q_id', qa.get('qid',''))
    question = qa.get('question','')
    doc_ids = qa.get('doc_ids', [])
    if not doc_ids and 'doc_id' in qa:
        doc_ids = [qa['doc_id']]

    caches = []
    missing = []
    for did in doc_ids:
        c = load_cache(did)
        if c is not None:
            caches.append(c)
        else:
            missing.append(did)

    if not caches:
        print(f"[{i+1}/{len(qa_data)}] {qid}: NO CACHES, skipping")
        continue

    combined = concatenate_caches(caches) if len(caches) > 1 else caches[0]

    answer, _ = _gen(model, combined, tokenizer, question, max_new_tokens=512)

    results.append({
        'qid': qid, 'q_id': qid, 'question': question,
        'generated_answer': answer,
        'ground_truth': qa.get('answer', qa.get('ground_truth','')),
        'key_points': qa.get('key_points', []),
        'doc_ids': doc_ids,
        'caches_loaded': len(caches),
        'missing_caches': missing,
        'cache_tokens': combined.num_tokens(),
    })
    print(f"[{i+1}/{len(qa_data)}] {qid}: {len(answer)} chars (caches={len(caches)}/{len(doc_ids)}, tokens={combined.num_tokens()})")

    for c in caches: del c
    if len(caches) > 1: del combined
    torch.cuda.empty_cache()

with open(args.output, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} results to {args.output}")
