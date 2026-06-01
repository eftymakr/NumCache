#!/usr/bin/env python3
"""
Baseline Evaluation: Test QA datasets with the correct KV cache for each question.

This follows the same approach as inference_with_cache.py but:
1. Loads QA datasets from /home/eftychia/Financial-QA-Benchmark-with-KV-cache/qa&corpus/qa/
2. For each question, looks up the correct cache based on doc_id
3. Generates answers and compares to ground truth

Datasets:
- chunk_based_qa_VLO_PSX.json - Single document questions  
- company_comparison_VLO_vs_PSX.json - Multi-document comparisons
- tracking_qa_VLO_PSX.json - Temporal tracking questions
"""

import argparse
import os
import sys
import json
import time
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch

# Disable torch.compile for RTX A6000 (shared memory limit issue)
os.environ["CARTRIDGES_DISABLE_COMPILE"] = "1"
torch._dynamo.config.suppress_errors = True

# Add cartridges to path
sys.path.append('/home/eftychia/Financial-QA-Benchmark-with-KV-cache')

from transformers import AutoTokenizer
from cartridges.cache import TrainableCache
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.generation import flex_generate
from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING

# Import concatenate_caches from the existing script
from inference_with_cache import concatenate_caches


def find_cache_path(doc_id: str, cache_base_dir: str) -> Optional[str]:
    """Find the cache directory for a given doc_id.
    
    Cache structure is:
    {cache_base_dir}/{doc_id}_*/training_output/{date}-train_config/{uuid}/cache_last.pt
    """
    pattern = os.path.join(cache_base_dir, f"{doc_id}_*")
    matches = glob.glob(pattern)
    if matches:
        cache_dir = matches[0]
        
        # First try direct cache files in the directory
        possible_names = ["cache-step100.pt", "cache-step50.pt", "cache.pt", "final_cache.pt", "cache_last.pt"]
        for name in possible_names:
            cache_file = os.path.join(cache_dir, name)
            if os.path.exists(cache_file):
                return cache_file
        
        # Look in training_output subdirectory (nested structure)
        training_output = os.path.join(cache_dir, "training_output")
        if os.path.exists(training_output):
            # Find the train_config directory
            config_dirs = glob.glob(os.path.join(training_output, "*-train_config"))
            if config_dirs:
                config_dir = config_dirs[0]
                # Find UUID subdirectory
                uuid_dirs = glob.glob(os.path.join(config_dir, "*"))
                for uuid_dir in uuid_dirs:
                    if os.path.isdir(uuid_dir):
                        # Look for cache files
                        for name in ["cache_last.pt", "cache-step100.pt", "cache-step50.pt", "cache-step2.pt"]:
                            cache_file = os.path.join(uuid_dir, name)
                            if os.path.exists(cache_file):
                                return cache_file
        
        # Fallback: recursive search for any cache*.pt file
        cache_files = glob.glob(os.path.join(cache_dir, "**", "cache*.pt"), recursive=True)
        if cache_files:
            # Prefer cache_last.pt
            for cf in cache_files:
                if "cache_last.pt" in cf:
                    return cf
            return cache_files[0]
    
    return None


def load_model_and_tokenizer(model_name: str = "Qwen/Qwen3-4B"):
    """Load the model and tokenizer once."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = FlexQwen3ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model, tokenizer


def load_cache(cache_path: str) -> TrainableCache:
    """Load a single cache file."""
    cache = TrainableCache.from_pretrained(cache_path, device="cuda")
    cache = cache.to("cuda")
    return cache


def generate_answer(
    model, 
    cache: TrainableCache, 
    tokenizer, 
    question: str, 
    max_new_tokens: int = 512
) -> Tuple[str, float]:
    """Generate answer using the KV cache."""
    
    # Use the same system prompt as training
    system_prompt = "You are a helpful assistant. Based on the knowledge base, answer questions accurately. Output the answer text only — do not repeat or paraphrase the question, do not summarize the context."
    
    # Create conversation format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    # Apply chat template
    kwargs = {"enable_thinking": False}  # Always disable thinking mode
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        chat_template=MODEL_TO_CHAT_TEMPLATE.get(tokenizer.name_or_path, None),
        **kwargs,
    )
    
    input_ids = input_ids.to("cuda")
    input_ids_concat = input_ids[0]
    
    seq_ids = torch.full((input_ids_concat.shape[0],), 0, dtype=torch.long, device="cuda")
    position_ids = torch.arange(input_ids_concat.shape[0], device="cuda")
    
    # Generate with timing
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    pred_ids: Dict[int, List[int]] = flex_generate(
        input_ids=input_ids_concat,
        seq_ids=seq_ids,
        position_ids=position_ids,
        cache=cache,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        show_progress=False
    )
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    # Decode
    for seq_id, curr_pred_ids in pred_ids.items():
        pred_text = tokenizer.decode(curr_pred_ids, skip_special_tokens=True)
        return pred_text, latency_ms
    
    return "", latency_ms


def evaluate_dataset(
    model,
    tokenizer,
    qa_file: str,
    cache_base_dir: str,
    dataset_name: str,
    max_questions: int = None,
    output_file: str = None,
    max_new_tokens: int = 512,
):
    """Evaluate a QA dataset using correct caches."""
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {dataset_name}")
    print(f"{'='*70}")
    
    # Load QA data
    with open(qa_file, 'r') as f:
        qa_data = json.load(f)
    
    print(f"Loaded {len(qa_data)} questions")
    
    if max_questions:
        qa_data = qa_data[:max_questions]
        print(f"Limited to {max_questions} questions")
    
    results = []
    cache_hits = 0
    cache_misses = 0
    latencies = []
    missing_caches = set()
    
    # Cache storage to avoid reloading same caches
    loaded_caches = {}
    
    for i, qa in enumerate(qa_data):
        question = qa.get("question", "")
        ground_truth = qa.get("answer", "")
        key_points = qa.get("key_points", [])
        q_id = qa.get("q_id", f"q_{i}")
        
        # Get doc_id(s) - can be single or list
        doc_ids = qa.get("doc_ids", [qa.get("doc_id")])
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        doc_ids = [d for d in doc_ids if d]
        
        print(f"\n[{i+1}/{len(qa_data)}] {q_id}")
        print(f"  Q: {question[:80]}...")
        print(f"  Doc IDs: {doc_ids}")
        
        # Find caches for all doc_ids
        cache_paths = []
        for doc_id in doc_ids:
            cache_path = find_cache_path(doc_id, cache_base_dir)
            if cache_path:
                cache_paths.append((doc_id, cache_path))
            else:
                missing_caches.add(doc_id)
        
        if not cache_paths:
            print(f"  ❌ No cache found for any doc_id")
            cache_misses += 1
            results.append({
                "q_id": q_id,
                "question": question,
                "doc_ids": doc_ids,
                "ground_truth": ground_truth,
                "prediction": None,
                "error": "cache_not_found",
                "latency_ms": None,
            })
            continue
        
        try:
            # Load and possibly concatenate caches
            caches_to_use = []
            for doc_id, cache_path in cache_paths:
                if cache_path not in loaded_caches:
                    print(f"  Loading cache: {doc_id}")
                    loaded_caches[cache_path] = load_cache(cache_path)
                caches_to_use.append(loaded_caches[cache_path])
            
            # Concatenate if multiple caches
            if len(caches_to_use) == 1:
                combined_cache = caches_to_use[0]
            else:
                print(f"  Concatenating {len(caches_to_use)} caches...")
                combined_cache = concatenate_caches(caches_to_use)
            
            cache_tokens = combined_cache.num_tokens()
            print(f"  Cache tokens: {cache_tokens}")
            
            # Generate answer
            prediction, latency = generate_answer(
                model, combined_cache, tokenizer, question, max_new_tokens
            )
            
            latencies.append(latency)
            cache_hits += 1
            
            print(f"  ✓ Generated ({latency:.1f}ms)")
            print(f"  Prediction: {prediction[:100]}...")
            
            results.append({
                "q_id": q_id,
                "question": question,
                "doc_ids": doc_ids,
                "used_doc_ids": [d for d, _ in cache_paths],
                "ground_truth": ground_truth,
                "prediction": prediction,
                "key_points": key_points,
                "cache_tokens": cache_tokens,
                "latency_ms": latency,
            })
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            cache_misses += 1
            results.append({
                "q_id": q_id,
                "question": question,
                "doc_ids": doc_ids,
                "ground_truth": ground_truth,
                "prediction": None,
                "error": str(e),
                "latency_ms": None,
            })
    
    # Calculate summary
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        # Skip first for warmup
        avg_latency_no_warmup = sum(latencies[1:]) / len(latencies[1:]) if len(latencies) > 1 else avg_latency
    else:
        avg_latency = 0
        avg_latency_no_warmup = 0
    
    summary = {
        "dataset": dataset_name,
        "total_questions": len(qa_data),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "missing_caches": list(missing_caches),
        "avg_latency_ms": avg_latency,
        "avg_latency_no_warmup_ms": avg_latency_no_warmup,
        "min_latency_ms": min(latencies) if latencies else 0,
        "max_latency_ms": max(latencies) if latencies else 0,
    }
    
    print(f"\n--- {dataset_name} Summary ---")
    print(f"Total questions: {summary['total_questions']}")
    print(f"Cache hits: {summary['cache_hits']}")
    print(f"Cache misses: {summary['cache_misses']}")
    print(f"Avg latency: {summary['avg_latency_ms']:.1f}ms")
    print(f"Avg latency (no warmup): {summary['avg_latency_no_warmup_ms']:.1f}ms")
    if missing_caches:
        print(f"Missing caches ({len(missing_caches)}): {list(missing_caches)[:10]}...")
    
    # Save results
    if output_file:
        output_data = {
            "summary": summary,
            "results": results,
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")
    
    return summary, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate QA datasets with correct KV caches")
    parser.add_argument("--qa-dir", type=str, 
                        default="/home/eftychia/Financial-QA-Benchmark-with-KV-cache/qa&corpus/qa",
                        help="Directory containing QA JSON files")
    parser.add_argument("--cache-dir", type=str,
                        default="/ext/peiwenfiles/automated_runs_merged",
                        help="Base directory for KV caches")
    parser.add_argument("--output-dir", type=str,
                        default="/home/eftychia/Financial-QA-Benchmark-with-KV-cache/baseline_results",
                        help="Directory to save results")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Max questions per dataset (for testing)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens to generate")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["chunk_based", "company_comparison", "tracking"],
                        help="Which datasets to evaluate")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B",
                        help="Model name")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model once
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    # Dataset paths
    dataset_files = {
        "chunk_based": os.path.join(args.qa_dir, "chunk_based_qa_VLO_PSX.json"),
        "company_comparison": os.path.join(args.qa_dir, "company_comparison_VLO_vs_PSX.json"),
        "tracking": os.path.join(args.qa_dir, "tracking_qa_VLO_PSX.json"),
    }
    
    all_summaries = []
    
    for dataset_name in args.datasets:
        if dataset_name not in dataset_files:
            print(f"Unknown dataset: {dataset_name}")
            continue
            
        qa_file = dataset_files[dataset_name]
        if not os.path.exists(qa_file):
            print(f"File not found: {qa_file}")
            continue
        
        # Generate output filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            args.output_dir, 
            f"{dataset_name}_baseline_{timestamp}.json"
        )
        
        summary, results = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            qa_file=qa_file,
            cache_base_dir=args.cache_dir,
            dataset_name=dataset_name,
            max_questions=args.max_questions,
            output_file=output_file,
            max_new_tokens=args.max_new_tokens,
        )
        
        all_summaries.append(summary)
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for summary in all_summaries:
        print(f"\n{summary['dataset']}:")
        print(f"  Questions: {summary['cache_hits']}/{summary['total_questions']} with cache")
        print(f"  Avg latency: {summary['avg_latency_ms']:.1f}ms")
        print(f"  Avg latency (no warmup): {summary['avg_latency_no_warmup_ms']:.1f}ms")


if __name__ == "__main__":
    main()
