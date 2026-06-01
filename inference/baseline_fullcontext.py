#!/usr/bin/env python3
"""
Full-context baseline: Feed the entire document in the prompt (no KV cache).
Uses the same model (Qwen3-4B) and financial prompt as the cache experiments.
"""

import os
import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM


FINANCIAL_PROMPT = """You are a financial analyst expert. Based on the company filings and financial documents in the knowledge base, provide detailed and accurate analysis.
When comparing companies, highlight key differences in metrics, strategies, and performance.
When tracking a company over time, identify trends, changes, and significant events.
Be specific with numbers, dates, and facts from the documents."""

TRAINING_PROMPT = "Please answer the user's question based on your knowledge."
GENERIC_PROMPT = "You are a helpful assistant. Based on the knowledge base, answer questions accurately. Output the answer text only — do not repeat or paraphrase the question, do not summarize the context."


def load_corpus(corpus_file):
    """Load corpus and return dict mapping doc_id -> text."""
    doc_map = {}
    with open(corpus_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc.get('doc_id', doc.get('_id', doc.get('id', '')))
            text = doc.get('text', doc.get('contents', ''))
            doc_map[doc_id] = text
    return doc_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-file", required=True)
    parser.add_argument("--corpus", default="vlo_psx_benchmark_corpus.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--prompt-type", choices=["financial", "generic", "none", "training"], default="financial")
    parser.add_argument("--max-context-tokens", type=int, default=28000,
                        help="Max tokens of document context to include")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    # Load QA data
    with open(args.qa_file) as f:
        qa_data = json.load(f)
    print(f"Loaded {len(qa_data)} QAs")

    # Load corpus
    doc_map = load_corpus(args.corpus)
    print(f"Loaded {len(doc_map)} documents from corpus")

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Set system prompt
    if args.prompt_type == "financial":
        system_prompt = FINANCIAL_PROMPT
    elif args.prompt_type == "generic":
        system_prompt = GENERIC_PROMPT
    elif args.prompt_type == "training":
        system_prompt = TRAINING_PROMPT
    else:
        system_prompt = None

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"baseline_fullcontext_{args.prompt_type}_{timestamp}.json"

    results = []
    for i, qa in enumerate(tqdm(qa_data, desc="Processing QAs")):
        question = qa.get("question", "")
        # Get doc_id - try multiple field names
        doc_ids = qa.get("doc_ids", [])
        doc_id = qa.get("doc_id", doc_ids[0] if doc_ids else "")

        # Get document text
        doc_text = doc_map.get(doc_id, "")
        if not doc_text:
            print(f"  [{i}] No document found for {doc_id}, skipping")
            results.append({
                "qid": qa.get("q_id", qa.get("qid", f"q_{i}")),
                "question": question,
                "gold_answer": qa.get("answer", qa.get("ground_truth", "")),
                "generated_answer": "",
                "key_points": qa.get("key_points", []),
                "doc_id": doc_id,
                "error": "no_document"
            })
            continue

        # Truncate document if needed
        doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)
        if len(doc_tokens) > args.max_context_tokens:
            doc_tokens = doc_tokens[:args.max_context_tokens]
            doc_text = tokenizer.decode(doc_tokens, skip_special_tokens=True)

        # Build chat messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_content = f"Document:\n{doc_text}\n\nQuestion: {question}"
        messages.append({"role": "user", "content": user_content})

        # Tokenize
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # Generate
        import time
        t0 = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        elapsed = time.time() - t0

        # Decode only the new tokens
        new_tokens = output_ids[0][input_ids.shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

        results.append({
            "qid": qa.get("q_id", qa.get("qid", f"q_{i}")),
            "question": question,
            "gold_answer": qa.get("answer", qa.get("ground_truth", "")),
            "generated_answer": answer,
            "key_points": qa.get("key_points", []),
            "doc_id": doc_id,
            "input_tokens": input_ids.shape[1],
            "output_tokens": len(new_tokens),
            "latency_ms": elapsed * 1000,
        })

        if (i + 1) % 5 == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

    # Save final results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {output_file}")


if __name__ == "__main__":
    main()
