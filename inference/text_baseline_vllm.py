"""
Text retrieval baseline using vLLM (OpenAI-compatible API).
Matches gen_training_prompt_topk1.py settings: training prompt, k=1, 80K-char context.
"""
import os, json, argparse, requests, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

_REPO_ROOT = Path(os.environ.get("NUMCACHE_REPO_ROOT", str(Path(__file__).resolve().parent.parent)))

parser = argparse.ArgumentParser()
parser.add_argument('--retrieval_file', required=True)
parser.add_argument('--qa_file', required=True)
parser.add_argument('--corpus_file', default=str(_REPO_ROOT / 'corpus.jsonl'))
parser.add_argument('--output', required=True)
parser.add_argument('--api_url', default='http://localhost:8200/v1/chat/completions')
parser.add_argument('--model', default='Qwen/Qwen3-4b')
parser.add_argument('--top_k', type=int, default=1)
parser.add_argument('--max_context_chars', type=int, default=80000)
parser.add_argument('--max_workers', type=int, default=4)
parser.add_argument('--max_new_tokens', type=int, default=512)
parser.add_argument('--golden_kp', default=str(_REPO_ROOT / 'qa' / 'chunk_vlo_psx_145_for_golden.json'),
                    help='Optional key-points file used to enrich the answer prompt.')
args = parser.parse_args()

SYSTEM_PROMPT = "Please answer the user's question based on your knowledge."

print(f"Loading corpus from {args.corpus_file}...")
corpus = {}
with open(args.corpus_file) as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc['_id']] = doc.get('text', '')

print(f"Loading QAs from {args.qa_file}...")
qa_data = json.load(open(args.qa_file))
qa_map = {qa.get('q_id') or qa.get('qid'): qa for qa in qa_data}

print(f"Loading retrieval from {args.retrieval_file}...")
retrieval = json.load(open(args.retrieval_file))
if retrieval and isinstance(next(iter(retrieval.values())), dict):
    flat = {}
    for v in retrieval.values():
        if isinstance(v, dict): flat.update(v)
    retrieval = flat

golden_kp = {}
try:
    for qa in json.load(open(args.golden_kp)):
        qid = qa.get('qid') or qa.get('q_id')
        golden_kp[qid] = qa.get('key_points', [])
except Exception as e:
    print(f"  (warn) golden_kp load: {e}")


def truncate(context, n=args.max_context_chars):
    if len(context) <= n: return context
    s = context[:n]
    last = s.rfind("\n\n---\n\n")
    if last > n * 0.7: s = s[:last]
    return s + "\n\n[Context truncated]"


def call_vllm(question, doc_texts):
    context = "\n\n---\n\n".join(doc_texts)
    context = truncate(context)
    user = f"{question}\n\nContext:\n{context}"
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": args.max_new_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(args.api_url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def worker(qid, doc_ids):
    qa = qa_map.get(qid)
    if not qa: return None
    doc_texts = [corpus.get(d, '') for d in doc_ids[:args.top_k] if corpus.get(d)]
    if not doc_texts: return None
    try:
        answer = call_vllm(qa['question'], doc_texts)
    except Exception as e:
        return {'qid': qid, 'error': str(e)[:200]}
    return {
        'qid': qid, 'q_id': qid, 'question': qa['question'],
        'generated_answer': answer,
        'ground_truth': qa.get('answer', ''),
        'key_points': golden_kp.get(qid, qa.get('key_points', [])),
        'doc_ids': doc_ids[:args.top_k],
        'n_docs': len(doc_texts),
    }


items = list(retrieval.items())
print(f"Running {len(items)} QAs with {args.max_workers} workers (vLLM at {args.api_url})...")
results = []
errors = 0
t0 = time.time()
with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
    futs = {ex.submit(worker, q, d): q for q, d in items}
    done = 0
    for fut in as_completed(futs):
        r = fut.result()
        done += 1
        if r is None:
            continue
        if 'error' in r:
            errors += 1
            print(f"[{done}/{len(items)}] {r['qid']}: ERROR {r['error']}")
            continue
        results.append(r)
        if done % 20 == 0:
            print(f"[{done}/{len(items)}] {r['qid']}: {len(r['generated_answer'])} chars  ({time.time()-t0:.0f}s elapsed)")

with open(args.output, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} / {len(items)} (errors: {errors}) → {args.output}")
