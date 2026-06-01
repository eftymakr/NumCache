#!/usr/bin/env python3
"""
Simple inference script that mimics the training generation evaluation.
Supports loading and concatenating multiple cache files.

Usage examples:
1. Single cache:
   python inference_with_cache.py --cache_paths /path/to/cache1.pt --question "What is the answer?"

2. Multiple caches (concatenated):
   python inference_with_cache.py --cache_paths /path/to/cache1.pt /path/to/cache2.pt /path/to/cache3.pt --question "What is the answer?"

3. Directory with pattern matching:
   python inference_with_cache.py --cache_dir /path/to/cache/dir --cache_pattern "cache-step*.pt" --question "What is the answer?"

4. Batch processing with multiple caches:
   python inference_with_cache.py --cache_paths /path/to/cache1.pt /path/to/cache2.pt --batch_mode --qa_pairs_file /path/to/qa.json
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import Dict, List

# Enable torch.compile for optimized flex_attention
# Note: First run will be slower due to compilation, subsequent runs will be faster
torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.disable = True  # DISABLED to allow flex_attention optimization
# os.environ["TORCH_COMPILE"] = "0"
# os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Add cartridges to path
sys.path.append('/home/eftychia/Financial-QA-Benchmark-with-KV-cache')

from cartridges.cache import TrainableCache, AttnConfig
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.generation import flex_generate
from cartridges.initialization import KVFromText
from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING

def concatenate_caches(caches: List[TrainableCache]) -> TrainableCache:
    """
    Concatenate multiple TrainableCache objects into a single cache.
    
    Args:
        caches: List of TrainableCache objects to concatenate
        
    Returns:
        Combined TrainableCache object
    """
    if len(caches) == 1:
        return caches[0]
    
    print(f"Concatenating {len(caches)} caches...")
    
    # Use the first cache as the base
    combined_cache = caches[0]
    
    # Concatenate each additional cache
    for i in range(1, len(caches)):
        cache_to_add = caches[i]
        print(f"Adding cache {i+1} with {cache_to_add.num_tokens()} tokens...")
        
        # Concatenate trainable keys and values for each layer
        for layer_idx in range(combined_cache.config.n_layers):
            if (combined_cache.trainable_keys[layer_idx] is not None and 
                cache_to_add.trainable_keys[layer_idx] is not None):
                
                # Concatenate trainable keys
                combined_keys = torch.cat([
                    combined_cache.trainable_keys[layer_idx],
                    cache_to_add.trainable_keys[layer_idx]
                ], dim=2)
                
                # Concatenate trainable values
                combined_values = torch.cat([
                    combined_cache.trainable_values[layer_idx],
                    cache_to_add.trainable_values[layer_idx]
                ], dim=2)
                
                # Update the combined cache
                combined_cache.trainable_keys[layer_idx] = nn.Parameter(combined_keys)
                combined_cache.trainable_values[layer_idx] = nn.Parameter(combined_values)
        
        # Concatenate frozen keys and values if they exist
        if (len(combined_cache.frozen_keys) > 0 and len(cache_to_add.frozen_keys) > 0):
            for layer_idx in range(len(combined_cache.frozen_keys)):
                if (combined_cache.frozen_keys[layer_idx] is not None and 
                    cache_to_add.frozen_keys[layer_idx] is not None):
                    
                    # Concatenate frozen keys
                    combined_frozen_keys = torch.cat([
                        combined_cache.frozen_keys[layer_idx],
                        cache_to_add.frozen_keys[layer_idx]
                    ], dim=2)
                    
                    # Concatenate frozen values
                    combined_frozen_values = torch.cat([
                        combined_cache.frozen_values[layer_idx],
                        cache_to_add.frozen_values[layer_idx]
                    ], dim=2)
                    
                    # Update the combined cache
                    combined_cache.frozen_keys[layer_idx] = nn.Parameter(combined_frozen_keys)
                    combined_cache.frozen_values[layer_idx] = nn.Parameter(combined_frozen_values)
        
        # Concatenate sequence IDs
        if (combined_cache._seq_ids is not None and cache_to_add._seq_ids is not None):
            combined_cache._seq_ids = torch.cat([
                combined_cache._seq_ids,
                cache_to_add._seq_ids
            ], dim=0)
        
        # Update token counts
        combined_cache._num_trainable_tokens += cache_to_add._num_trainable_tokens
        combined_cache._num_frozen_tokens += cache_to_add._num_frozen_tokens
        combined_cache._num_tokens += cache_to_add._num_tokens
    
    print(f"Combined cache now has {combined_cache.num_tokens()} total tokens")
    return combined_cache

def load_model_and_cache(cache_paths: List[str], model_name: str = "Qwen/Qwen3-4b"):
    """Load the model and trained cache(s) using the same method as inference_with_kv_cache_v2.py."""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer using the same method as inference_with_kv_cache_v2.py
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model using the same method as inference_with_kv_cache_v2.py
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
    
    model = FlexQwen3ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 to match training
        device_map="cuda",
        trust_remote_code=True
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Load and concatenate multiple caches
    print(f"Loading {len(cache_paths)} cache(s)...")
    caches = []
    
    for i, cache_path in enumerate(cache_paths):
        print(f"Loading cache {i+1}/{len(cache_paths)} from: {cache_path}")
        cache = TrainableCache.from_pretrained(cache_path, device="cuda")
        cache = cache.to("cuda")
        caches.append(cache)
        
        print(f"  - Cache {i+1} config: n_layers={cache.config.n_layers}, n_heads={cache.config.n_heads}, head_dim={cache.config.head_dim}")
        print(f"  - Cache {i+1} tokens: {cache.num_tokens()}, Cartridge tokens: {cache.num_cartridge_tokens()}")
    
    # Concatenate all caches
    if len(caches) == 1:
        combined_cache = caches[0]
    else:
        print("Concatenating caches...")
        combined_cache = concatenate_caches(caches)
    
    print(f"Combined cache loaded successfully!")
    print(f"  - Model: {model_name}")
    print(f"  - Combined cache config: n_layers={combined_cache.config.n_layers}, n_heads={combined_cache.config.n_heads}, head_dim={combined_cache.config.head_dim}")
    print(f"  - Combined cache tokens: {combined_cache.num_tokens()}, Cartridge tokens: {combined_cache.num_cartridge_tokens()}")
    
    return model, combined_cache, tokenizer

def generate_answer(model, cache, tokenizer, question: str, max_new_tokens: int = 128):
    """Generate answer using the same method as train.py evaluate_generations."""
    print(f"\nQuestion: {question}")
    print("Generating answer...")
    
    # Option A: Generic prompt (original)
    # system_prompt = "You are a helpful assistant. Based on the knowledge base, answer questions accurately. Output the answer text only — do not repeat or paraphrase the question, do not summarize the context."

    # Training prompt (matches what the cache was trained with)
    system_prompt = """Please answer the user's question based on your knowledge."""
    
    # Create conversation format exactly like GenerateEvalDataset.__getitem__
    # This matches datasets.py lines 564-573
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    # Apply chat template exactly like GenerateEvalDataset
    kwargs = {}
    if tokenizer.name_or_path in MODELS_WITH_THINKING:
        kwargs["enable_thinking"] = False
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        chat_template=MODEL_TO_CHAT_TEMPLATE.get(tokenizer.name_or_path, None),
        **kwargs,
    )
    
    # Move input_ids to CUDA
    input_ids = input_ids.to("cuda")
    
    # Create input data exactly like train.py evaluate_generations
    # This matches train.py lines 708-717
    input_ids_concat = input_ids[0]  # Remove batch dimension for concatenation
    
    # Create seq_ids exactly like train.py lines 709-714
    seq_ids = torch.full((input_ids_concat.shape[0],), 0, dtype=torch.long, device="cuda")
    
    # Create position_ids exactly like train.py lines 715-717
    position_ids = torch.arange(input_ids_concat.shape[0], device="cuda")
    
    print(f"Input IDs shape: {input_ids_concat.shape}")
    print(f"Seq IDs shape: {seq_ids.shape}")
    print(f"Position IDs shape: {position_ids.shape}")
    
    # Use flex_generate exactly like train.py lines 718-732
    pred_ids: Dict[int, List[int]] = flex_generate(
        input_ids=input_ids_concat,
        seq_ids=seq_ids,
        position_ids=position_ids,
        cache=cache,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        show_progress=True
    )
    
    # Decode the generated tokens exactly like train.py lines 799-805
    pred = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    
    print(f"Generated tokens dict: {pred_ids}")
    print(f"Batch decoded: {pred}")
    
    # Process results exactly like train.py lines 803-805
    for seq_id, curr_pred_ids in pred_ids.items():
        pred_text = tokenizer.decode(curr_pred_ids, skip_special_tokens=True)
        print(f"✅ Generated {len(curr_pred_ids)} tokens for sequence {seq_id}")
        print(f"Generated text: {pred_text}")
        
        # Return the generated text directly (no prompt removal)
        # This matches train.py behavior where pred is used directly
        return pred_text, pred_text
    
    print("❌ No tokens generated")
    return "", ""

def process_qa_pairs(model, cache, tokenizer, qa_pairs_file: str, output_file: str, max_new_tokens: int = 128):
    """Process all QA pairs from JSON file and generate answers."""
    print(f"Loading QA pairs from: {qa_pairs_file}")
    
    # Load QA pairs
    with open(qa_pairs_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    print(f"Found {len(qa_pairs)} QA pairs to process")
    
    # Process each QA pair
    results = []
    latencies = []
    for i, qa_pair in enumerate(qa_pairs):
        print(f"\n{'='*60}")
        print(f"Processing QA pair {i+1}/{len(qa_pairs)}")
        print(f"{'='*60}")
        
        question = qa_pair['question']
        original_answer = (
        qa_pair.get('answer') or 
        qa_pair.get('ground_truth_answer') or 
        qa_pair.get('ground_truth') or
        qa_pair.get('gold_answer') or
        qa_pair.get('reference_answer') or
        qa_pair.get('expected_answer') or
        ''
    )
        
        # Generate answer with timing
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        generated_answer, full_text = generate_answer(
            model, cache, tokenizer, 
            question, 
            max_new_tokens
        )
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        # Create result entry
        result_entry = {
            'question': question,
            'answer': original_answer,
            'generated_answer': generated_answer,
            'latency_ms': latency_ms
        }
        
        results.append(result_entry)
        
        print(f"✅ Completed QA pair {i+1}/{len(qa_pairs)} | Latency: {latency_ms:.2f} ms")
    
    # Compute summary stats
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        # Skip first one (warmup) for better average
        if len(latencies) > 1:
            avg_latency_no_warmup = sum(latencies[1:]) / len(latencies[1:])
        else:
            avg_latency_no_warmup = avg_latency
        
        print(f"\n{'='*60}")
        print("LATENCY SUMMARY")
        print(f"{'='*60}")
        print(f"  Total questions:     {len(latencies)}")
        print(f"  Average latency:     {avg_latency:.2f} ms")
        print(f"  Avg (excl. warmup):  {avg_latency_no_warmup:.2f} ms")
        print(f"  Min latency:         {min_latency:.2f} ms")
        print(f"  Max latency:         {max_latency:.2f} ms")
        print(f"  Throughput:          {1000/avg_latency_no_warmup:.2f} questions/sec")
    
    # Save results with summary
    output_data = {
        'results': results,
        'latency_summary': {
            'num_questions': len(latencies),
            'avg_latency_ms': avg_latency if latencies else 0,
            'avg_latency_no_warmup_ms': avg_latency_no_warmup if latencies else 0,
            'min_latency_ms': min_latency if latencies else 0,
            'max_latency_ms': max_latency if latencies else 0,
            'throughput_qps': 1000/avg_latency_no_warmup if latencies else 0
        }
    }
    
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Successfully processed {len(results)} QA pairs")
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate answers using trained cache(s)")
    parser.add_argument("--cache_paths", type=str, nargs="+",
                       default=[
                       "/home/eftychia/Financial-QA-Benchmark-with-KV-cache/outputs_RAG_chunks/2026-01-16-18-06-13-train_config/eb54053c-d546-4179-ade8-38b91ed248b9/cache_last.pt"],
                       help="Path(s) to the trained cache file(s). Multiple paths will be concatenated.")
    parser.add_argument("--cache_dir", type=str, default="",
                       help="Directory containing cache files (alternative to --cache_paths)")
    parser.add_argument("--cache_pattern", type=str, default="*.pt",
                       help="Pattern to match cache files in directory (e.g., 'cache-step*.pt')")
    parser.add_argument("--question", type=str, default="",
                       help="Single question to ask the model (if not using batch mode)")
    parser.add_argument("--qa_pairs_file", type=str, 
                       default="/home/eftychia/Financial-QA-Benchmark-with-KV-cache/qa_output/chunks/train_eval_split_chunk_0001/chunk_0001_train.json",
                       help="Path to JSON file containing QA pairs for batch processing")
    parser.add_argument("--output_file", type=str, 
                       default="/home/eftychia/Financial-QA-Benchmark-with-KV-cache/test_ret/financial_performance_answer/chunk_001__train_379.json",
                       help="Path to save the generated answers")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4b",
                       help="Model name to use")
    parser.add_argument("--batch_mode", action="store_true",
                       help="Enable batch processing mode")
    
    args = parser.parse_args()
    
    # Determine cache paths
    cache_paths = args.cache_paths
    
    # If cache_dir is specified, find cache files matching the pattern
    if args.cache_dir:
        import glob
        cache_pattern = os.path.join(args.cache_dir, args.cache_pattern)
        found_caches = glob.glob(cache_pattern)
        if found_caches:
            cache_paths = sorted(found_caches)  # Sort for consistent ordering
            print(f"Found {len(cache_paths)} cache files in {args.cache_dir}:")
            for i, path in enumerate(cache_paths):
                print(f"  {i+1}. {path}")
        else:
            print(f"Error: No cache files found matching pattern {cache_pattern}")
            sys.exit(1)
    
    # Check if cache files exist
    for cache_path in cache_paths:
        if not os.path.exists(cache_path):
            print(f"Error: Cache file not found at {cache_path}")
            sys.exit(1)
    
    # Set environment variables
    #os.environ["CARTRIDGES_DIR"] = "/home/yidong/cartridges-new"
    #os.environ["CARTRIDGES_OUTPUT_DIR"] = "/home/yidong/cartridges-new/outputs"
    
    try:
        # Load model and cache(s)
        model, cache, tokenizer = load_model_and_cache(cache_paths, args.model)
        
        if args.batch_mode:
            # Batch processing mode
            if not os.path.exists(args.qa_pairs_file):
                print(f"Error: QA pairs file not found at {args.qa_pairs_file}")
                sys.exit(1)
            
            print("Running in batch processing mode...")
            results = process_qa_pairs(
                model, cache, tokenizer,
                args.qa_pairs_file,
                args.output_file,
                args.max_tokens
            )
            
            print(f"\n✅ Batch processing completed!")
            print(f"Results saved to: {args.output_file}")
            
        else:
            # Single question mode
            if not args.question:
                print("Error: Please provide either --question for single mode or --batch_mode for batch processing")
                sys.exit(1)
            
            print("Running in single question mode...")
            answer, full_text = generate_answer(
                model, cache, tokenizer, 
                args.question, 
                args.max_tokens
            )
            
            # Display results
            print("\n" + "="*80)
            print("RESULTS:")
            print("="*80)
            print(f"Question: {args.question}")
            print(f"Answer: {answer}")
            print("\nFull generated text:")
            print(full_text)
            print("="*80)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
