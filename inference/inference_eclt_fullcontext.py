"""
Full-context baseline for EC-QA (19) and LT-QA (97) with training prompt.
Uses gold doc texts concatenated as context, no cache.
"""
import torch, sys, os, json, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument('--qa_file', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--max_chars', type=int, default=80000)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

BASE = '/home/eftychia/Financial-QA-Benchmark-with-KV-cache'

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4b", torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4b")
model.eval()

print("Loading corpus...")
corpus = {}
with open(f'{BASE}/corpus.jsonl') as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc['_id']] = doc.get('text','')

with open(args.qa_file) as f:
    qa_data = json.load(f)

SYSTEM_PROMPT = "Please answer the user's question based on your knowledge."

def generate(question, doc_texts):
    context = '\n\n---\n\n'.join(doc_texts)
    if len(context) > args.max_chars:
        context = context[:args.max_chars]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Document:\n{context}\n\nQuestion: {question}"}
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
        enable_thinking=False
    ).to("cuda")

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

results = []
for i, qa in enumerate(qa_data):
    qid = qa.get('q_id', qa.get('qid',''))
    question = qa.get('question','')
    doc_ids = qa.get('doc_ids', [])
    if not doc_ids and 'doc_id' in qa:
        doc_ids = [qa['doc_id']]

    doc_texts = [corpus.get(d,'') for d in doc_ids if corpus.get(d)]
    if not doc_texts:
        print(f"[{i+1}/{len(qa_data)}] {qid}: NO DOC TEXTS, skipping")
        continue

    try:
        answer = generate(question, doc_texts)
    except Exception as e:
        print(f"[{i+1}/{len(qa_data)}] {qid}: ERROR - {e}")
        continue

    results.append({
        'qid': qid, 'q_id': qid, 'question': question,
        'generated_answer': answer,
        'ground_truth': qa.get('answer', qa.get('ground_truth','')),
        'key_points': qa.get('key_points', []),
        'doc_ids': doc_ids,
        'n_docs': len(doc_texts),
    })
    print(f"[{i+1}/{len(qa_data)}] {qid}: {len(answer)} chars ({len(doc_texts)} docs)")
    torch.cuda.empty_cache()

with open(args.output, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {len(results)} results to {args.output}")
