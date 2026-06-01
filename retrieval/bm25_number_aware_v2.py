"""
Number-aware BM25 retrieval using rank_bm25 library for fair comparison with kdd_rag's Elasticsearch BM25.

Two approaches:
1. Standard BM25 (rank_bm25) — should closely match Elasticsearch BM25
2. Number-aware BM25 — expand queries and docs with extracted financial numbers to boost their weight

Runs on 826-doc pool and full 15k corpus.
"""
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Set
import numpy as np
from rank_bm25 import BM25Okapi
import torch

BASE_DIR = Path(__file__).parent
CORPUS_FILE = BASE_DIR / "corpus.jsonl"
QA_FILE = BASE_DIR / "qa" / "chunk_based_qa_VLO_PSX.json"
GT_FILE = BASE_DIR / "chunk_vlo_psx_145_for_golden.json"


# Financial number regex patterns
FINANCIAL_PATTERNS = [
    r'\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|B|M|K))?',  # $1,234.56 million
    r'\d+\.?\d*\s*%',                                                  # 12.5%
    r'(?:Q[1-4]\s+)?20[12]\d',                                        # Q4 2024, 2023
    r'\d{1,3}(?:,\d{3})+(?:\.\d+)?',                                  # 1,234,567.89
    r'\d+\.\d{2,}',                                                    # 127.96
]


def extract_financial_numbers(text: str) -> List[str]:
    """Extract financial numbers/patterns from text."""
    numbers = []
    for pattern in FINANCIAL_PATTERNS:
        matches = re.findall(pattern, text)
        numbers.extend(matches)
    return numbers


def tokenize(text: str) -> List[str]:
    """Tokenize text similarly to Elasticsearch's standard analyzer."""
    text = text.lower()
    # Split on non-alphanumeric, keep financial symbols
    tokens = re.findall(r'[a-z0-9$%.,]+', text)
    # Remove very short tokens (like ES stopwords removal)
    tokens = [t for t in tokens if len(t) > 1 or t in ('$', '%')]
    return tokens


def number_augment(tokens: List[str], text: str, repeat: int = 3) -> List[str]:
    """Augment token list by repeating financial numbers to boost their BM25 weight."""
    numbers = extract_financial_numbers(text)
    # Tokenize each extracted number and repeat
    augmented = list(tokens)
    for num in numbers:
        num_tokens = tokenize(num)
        for _ in range(repeat):
            augmented.extend(num_tokens)
    return augmented


def load_corpus(corpus_file: Path, doc_filter: Set[str] = None) -> Dict[str, str]:
    corpus = {}
    with open(corpus_file) as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc.get("_id", "")
            if doc_filter is not None and doc_id not in doc_filter:
                continue
            if doc_id:
                corpus[doc_id] = doc.get("text", "")
    return corpus


def run_bm25(corpus: Dict[str, str], qa_data: list, gt_map: dict,
             number_aware: bool = False, number_repeat: int = 3):
    """Run BM25 retrieval and compute metrics."""
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]

    # Tokenize corpus
    if number_aware:
        tokenized_corpus = [number_augment(tokenize(text), text, repeat=number_repeat)
                           for text in doc_texts]
    else:
        tokenized_corpus = [tokenize(text) for text in doc_texts]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Retrieve for each query
    r1 = r3 = r5 = r10 = 0
    total = 0
    retrieved_doc_ids = {}

    for qa in qa_data:
        qid = qa.get("q_id", qa.get("qid", ""))
        question = qa.get("question", "")

        # Tokenize query
        if number_aware:
            query_tokens = number_augment(tokenize(question), question, repeat=number_repeat)
        else:
            query_tokens = tokenize(question)

        # Get scores
        scores = bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:10]
        ret_docs = [doc_ids[i] for i in top_indices]
        retrieved_doc_ids[qid] = ret_docs

        # Compute recall vs ground truth — supports str (chunk_based) or list (LT/EC)
        gt = gt_map.get(qid, qa.get("doc_id") or qa.get("doc_ids") or "")
        if isinstance(gt, str):
            gt_set = {gt} if gt else set()
        else:
            gt_set = set(g for g in gt if g)
        if not gt_set:
            continue
        total += 1
        if gt_set & set(ret_docs[:1]): r1 += 1
        if gt_set & set(ret_docs[:3]): r3 += 1
        if gt_set & set(ret_docs[:5]): r5 += 1
        if gt_set & set(ret_docs[:10]): r10 += 1

    metrics = {
        "total": total,
        "R@1": r1, "R@1_pct": r1/total*100 if total else 0,
        "R@3": r3, "R@3_pct": r3/total*100 if total else 0,
        "R@5": r5, "R@5_pct": r5/total*100 if total else 0,
        "R@10": r10, "R@10_pct": r10/total*100 if total else 0,
    }
    return metrics, retrieved_doc_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["pool_826", "full_15k", "both"], default="both")
    parser.add_argument("--output_dir", default=str(BASE_DIR / "bm25_number_aware_v2_results"))
    parser.add_argument("--qa_file", default=str(QA_FILE),
                        help="Path to QA JSON (supports chunk_based, tracking, comparison)")
    parser.add_argument("--gt_file", default=str(GT_FILE),
                        help="Optional separate GT file. If missing, gt is read from QA's doc_id/doc_ids field.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load QA data
    with open(args.qa_file) as f:
        qa_data = json.load(f)
    print(f"Loaded {len(qa_data)} QAs from {args.qa_file}")

    # Load ground truth (optional — falls back to QA's own doc_id/doc_ids field per-QA)
    gt_map = {}
    if args.gt_file and Path(args.gt_file).exists():
        try:
            with open(args.gt_file) as f:
                gt_data = json.load(f)
            gt_map = {qa.get("qid", qa.get("q_id", "")): qa.get("doc_id", qa.get("doc_ids", "")) for qa in gt_data}
            print(f"Loaded GT for {len(gt_map)} QAs from {args.gt_file}")
        except Exception as e:
            print(f"  (warn) GT load failed: {e} — falling back to QA's own doc_id field")

    # Get 826-doc pool
    pool_826 = set()
    try:
        data = torch.load(BASE_DIR / "retrieval_results/pooled_kv_combined.pt", map_location="cpu")
        pool_826 = set(data['doc_ids'])
        print(f"826-doc pool: {len(pool_826)} docs")
    except:
        pass

    corpora = []
    if args.corpus in ["pool_826", "both"]:
        corpora.append(("pool_826", pool_826))
    if args.corpus in ["full_15k", "both"]:
        corpora.append(("full_15k", None))

    for corpus_name, doc_filter in corpora:
        print(f"\n{'='*60}")
        print(f"{corpus_name}")
        print(f"{'='*60}")

        corpus = load_corpus(CORPUS_FILE, doc_filter)
        print(f"Loaded {len(corpus)} documents")

        # Standard BM25
        print("\n--- Standard BM25 (rank_bm25) ---")
        metrics_std, retrieved_std = run_bm25(corpus, qa_data, gt_map, number_aware=False)
        print(f"  R@1={metrics_std['R@1']}/{metrics_std['total']} ({metrics_std['R@1_pct']:.1f}%)")
        print(f"  R@3={metrics_std['R@3']}/{metrics_std['total']} ({metrics_std['R@3_pct']:.1f}%)")
        print(f"  R@5={metrics_std['R@5']}/{metrics_std['total']} ({metrics_std['R@5_pct']:.1f}%)")
        print(f"  R@10={metrics_std['R@10']}/{metrics_std['total']} ({metrics_std['R@10_pct']:.1f}%)")

        # Number-aware BM25
        print("\n--- Number-Aware BM25 (3x repeat) ---")
        metrics_num, retrieved_num = run_bm25(corpus, qa_data, gt_map, number_aware=True, number_repeat=3)
        print(f"  R@1={metrics_num['R@1']}/{metrics_num['total']} ({metrics_num['R@1_pct']:.1f}%)")
        print(f"  R@3={metrics_num['R@3']}/{metrics_num['total']} ({metrics_num['R@3_pct']:.1f}%)")
        print(f"  R@5={metrics_num['R@5']}/{metrics_num['total']} ({metrics_num['R@5_pct']:.1f}%)")
        print(f"  R@10={metrics_num['R@10']}/{metrics_num['total']} ({metrics_num['R@10_pct']:.1f}%)")

        # Save
        results = {
            "corpus": corpus_name,
            "corpus_size": len(corpus),
            "standard_bm25": metrics_std,
            "number_aware_bm25": metrics_num,
        }
        with open(output_dir / f"{corpus_name}_results.json", "w") as f:
            json.dump(results, f, indent=2)

        with open(output_dir / f"{corpus_name}_standard_retrieved.json", "w") as f:
            json.dump(retrieved_std, f, indent=2)
        with open(output_dir / f"{corpus_name}_number_aware_retrieved.json", "w") as f:
            json.dump(retrieved_num, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
