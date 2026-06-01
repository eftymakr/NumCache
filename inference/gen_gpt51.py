"""
GPT-5.1 with the TRAINING PROMPT for direct comparison vs Qwen3-4B + training prompt.
Three modes:
  --mode fullctx : single ground-truth doc as context (matches fullctx_training_prompt_results.json)
  --mode topk1   : top-1 retrieved doc (replays retrieval from retrieval_training_prompt_topk1_results.json)
  --mode topk5   : top-5 retrieved docs concatenated (replays from retrieval_training_prompt_topk5_results.json)

Calls OpenAI gpt-5.1 via the Responses API (preferred for GPT-5*).
Concurrent (ThreadPoolExecutor) with periodic save. Resumes from existing output.
"""
import argparse, json, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "minimal").strip().lower()

BASE_DIR = Path("/home/eftychia/Financial-QA-Benchmark-with-KV-cache")
QA_FILE = BASE_DIR / "qa" / "chunk_based_qa_VLO_PSX.json"
CORPUS_FILE = BASE_DIR / "corpus.jsonl"

SYSTEM_PROMPT = "Please answer the user's question based on your knowledge."

MAX_CONTEXT_CHARS = 200000   # ~50K tokens, gpt-5.1 supports 400K
MAX_OUT = 512


def load_json(p): return json.load(open(p))


def load_corpus():
    d = {}
    with open(CORPUS_FILE) as f:
        for line in f:
            o = json.loads(line)
            d[o["_id"]] = o["text"]
    return d


def truncate(context, n=MAX_CONTEXT_CHARS):
    if len(context) <= n: return context
    s = context[:n]
    last = s.rfind("\n\n---\n\n")
    if last > n * 0.7: s = s[:last]
    return s + "\n\n[Context truncated]"


_client = None
_client_lock = threading.Lock()
def get_client():
    global _client
    with _client_lock:
        if _client is None:
            from openai import OpenAI
            _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client


def call_llm(question, context):
    client = get_client()
    user_prompt = f"{question}\n\nContext:\n{context}"
    prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    # Try Responses API first
    try:
        params = dict(model=OPENAI_MODEL, input=prompt, max_output_tokens=MAX_OUT)
        if OPENAI_REASONING_EFFORT:
            params["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}
        resp = client.responses.create(**params)
        text = (getattr(resp, "output_text", "") or "").strip()
        if text:
            return text
    except Exception as e:
        msg = str(e)
        if "reasoning.effort" in msg and "none" in msg.lower():
            try:
                resp = client.responses.create(model=OPENAI_MODEL, input=prompt, max_output_tokens=MAX_OUT, reasoning={"effort":"minimal"})
                text = (getattr(resp, "output_text", "") or "").strip()
                if text:
                    return text
            except Exception:
                pass
    # Fallback: chat completions
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_prompt}],
            max_completion_tokens=MAX_OUT,
        )
    except TypeError:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_prompt}],
            max_tokens=MAX_OUT,
        )
    if getattr(completion, "choices", None):
        return (completion.choices[0].message.content or "").strip() or None
    return None


def build_context(qa, mode, corpus, retrieval_lookup):
    if mode == "fullctx":
        # chunk_based has doc_id (str); LT/EC have doc_ids (list)
        gold = qa.get("doc_id") or qa.get("doc_ids") or []
        if isinstance(gold, str): gold = [gold]
        texts = [corpus.get(d, "") for d in gold if corpus.get(d)]
        return truncate("\n\n---\n\n".join(texts))
    # topk1 / topk5: pull doc IDs from retrieval_lookup[qid]
    # Accepts entry as: dict with retrieved_chunks (rich format) OR plain list (flat format)
    qid = qa.get("q_id") or qa.get("qid")
    entry = retrieval_lookup.get(qid)
    if entry is None:
        return None
    if isinstance(entry, list):
        docs = entry
    elif isinstance(entry, dict):
        docs = entry.get("retrieved_chunks") or entry.get("doc_ids") or []
    else:
        return None
    k = 1 if mode == "topk1" else 5
    docs = docs[:k]
    texts = []
    for c in docs:
        did = c["doc_id"] if isinstance(c, dict) else c
        t = corpus.get(did, "")
        if t: texts.append(t)
    return truncate("\n\n---\n\n".join(texts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fullctx","topk1","topk5"], required=True)
    ap.add_argument("--qa_file", default=str(QA_FILE),
                    help="QA JSON path. Defaults to chunk_based_qa_VLO_PSX.json")
    ap.add_argument("--retrieval_file", default=None,
                    help="Override retrieval JSON. Default per mode for chunk_based.")
    ap.add_argument("--output", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out_path = args.output or f"gpt51_training_prompt_{args.mode}_results.json"

    print(f"=== gpt-5.1 + training prompt | mode={args.mode} | qa={args.qa_file} ===")
    qa_data = load_json(args.qa_file)
    corpus = load_corpus()
    print(f"  qa={len(qa_data)} corpus={len(corpus)}")

    retrieval_lookup = {}
    if args.mode in ("topk1","topk5"):
        retr_file = args.retrieval_file or (
            "retrieval_training_prompt_topk1_results.json" if args.mode=="topk1"
            else "retrieval_training_prompt_topk5_results.json")
        rdata = load_json(retr_file)
        # Two formats:
        #  (a) flat dict {qid: [doc_id, ...]}  ← /tmp/{tracking,comparison}_*_flat.json
        #  (b) rich list/dict with retrieved_chunks per QA ← Apr 10 contrastive output
        if isinstance(rdata, dict) and rdata and isinstance(next(iter(rdata.values())), list):
            retrieval_lookup = rdata
        else:
            items = rdata.get("results", rdata) if isinstance(rdata, dict) else rdata
            for x in items:
                qid = x.get("q_id") or x.get("qid")
                if qid: retrieval_lookup[qid] = x
        print(f"  loaded retrieval lookup: {len(retrieval_lookup)} from {retr_file}")

    # resume
    completed = {}
    if os.path.exists(out_path):
        try:
            ex = load_json(out_path)
            for x in ex:
                qid = x.get("qid") or x.get("q_id")
                if qid: completed[qid] = x
            print(f"  resume: {len(completed)} already done")
        except Exception:
            pass

    def worker(qa):
        qid = qa.get("q_id") or qa.get("qid")
        if qid in completed: return None
        ctx = build_context(qa, args.mode, corpus, retrieval_lookup)
        if not ctx:
            return {"qid": qid, "error": "no context"}
        t0 = time.time()
        try:
            ans = call_llm(qa["question"], ctx)
        except Exception as e:
            return {"qid": qid, "error": str(e)[:200]}
        if not ans:
            return {"qid": qid, "error": "empty response"}
        return {
            "qid": qid, "q_id": qid,
            "question": qa["question"],
            "doc_id": qa.get("doc_id",""),
            "generated_answer": ans,
            "ground_truth": qa.get("answer", qa.get("ground_truth","")),
            "key_points": qa.get("key_points", []),
            "elapsed_s": round(time.time()-t0, 2),
        }

    save_lock = threading.Lock()
    def save():
        with save_lock:
            json.dump(list(completed.values()), open(out_path,'w'), indent=2, ensure_ascii=False)

    print(f"  running with {args.workers} workers, output={out_path}")
    t_all = time.time()
    err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(worker, qa): qa for qa in qa_data}
        done = 0; total = len(futs)
        for f in as_completed(futs):
            done += 1
            r = f.result()
            if r is None: continue
            if "error" in r:
                err += 1
                print(f"  [{done}/{total}] {r['qid']}: ERROR {r['error']}")
                continue
            completed[r["qid"]] = r
            if done % 5 == 0:
                save()
                print(f"  [{done}/{total}] {r['qid']}: {len(r['generated_answer'])} chars ({r['elapsed_s']}s) | err={err}")
    save()
    print(f"\nSaved {len(completed)} (errors: {err}) -> {out_path} in {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
