#!/usr/bin/env python3
"""
inference_with_contrastive_retrieval.py

Integrates contrastive retrieval with KV cache inference.
Uses pre-trained projection heads to retrieve relevant chunks,
then generates answers using their KV caches.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# Enable torch.compile for optimized flex_attention
# Note: First run will be slower due to compilation, subsequent runs will be faster
torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.disable = True  # DISABLED to allow flex_attention optimization
# os.environ["TORCH_COMPILE"] = "0"
# os.environ["TORCHDYNAMO_DISABLE"] = "1"

sys.path.append('/home/eftychia/Financial-QA-Benchmark-with-KV-cache/cartridges')

from cartridges.cache import TrainableCache
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.generation import flex_generate

# Import with training prompt
import sys
sys.path.insert(0, '/tmp')
from inference_with_cache_training_prompt import concatenate_caches, generate_answer


class ProjectionHead(nn.Module):
    """Same architecture as training."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(p=0.1),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class PoolMLP(nn.Module):
    """Question-conditioned layer weighting (if using MLP pool)."""
    def __init__(self, in_dim: int, num_layers: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_layers),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.mlp(x), dim=-1)


    # Documents that perform better with First-p (qualitative) initialization.
# Narrative-heavy docs (8-K, DEF14A, 10-K risk factors/legal) where First-p outperforms NumCache.
QUALITATIVE_DOCS = {
    "doc_000020", "doc_000023", "doc_000031", "doc_000053", "doc_000058",
    "doc_000067", "doc_000070", "doc_000104", "doc_000111", "doc_000135",
    "doc_000146", "doc_000162", "doc_000181", "doc_000183",
    "doc_000199", "doc_000202", "doc_000256", "doc_000298", "doc_001164",
    "doc_001284", "doc_001311", "doc_001328", "doc_001382", "doc_001393",
    "doc_001399", "doc_001416", "doc_001419", "doc_001421", "doc_001426",
    "doc_001433", "doc_001436", "doc_001440", "doc_001455", "doc_001457",
    "doc_001486", "doc_001490", "doc_001531",
    "doc_001539", "doc_001628",
    "doc_001647", "doc_001654", "doc_001660", "doc_001695", "doc_001703",
    "doc_001706",
}


class ContrastiveRetriever:
    def __init__(
        self,
        model,
        tokenizer,
        pooled_kv_path: str,
        heads_path: str,
        cache_dir: str,
        pinit_cache_dir: str = None,
        device: str = "cuda",
        use_mlp_pool: bool = True,
    ):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        if isinstance(cache_dir, (list, tuple)):
            self.cache_dirs = [Path(d) for d in cache_dir]
        else:
            self.cache_dirs = [Path(cache_dir)]
        self.cache_dir = self.cache_dirs[0]  # backwards compat
        # Separate p-init cache dir for qualitative docs
        self.pinit_cache_dir = Path(pinit_cache_dir) if pinit_cache_dir else None
        self.use_mlp_pool = use_mlp_pool
        
        # Load pooled KV data
        print(f"Loading pooled KV from: {pooled_kv_path}")
        pooled_data = torch.load(pooled_kv_path, map_location="cpu", weights_only=False)
        self.doc_ids = pooled_data["doc_ids"]
        
        # Check if we have per-layer vectors (for MLP pool)
        if use_mlp_pool and "layer_vectors" in pooled_data:
            # layer_vectors: List[List[Tensor]] - [num_chunks][num_layers]
            self.layer_vectors = pooled_data["layer_vectors"]
            self.num_layers = len(self.layer_vectors[0])
            layer_dim = self.layer_vectors[0][0].shape[0]
            print(f"Loaded {len(self.doc_ids)} chunks with {self.num_layers} layers each (dim={layer_dim})")
            
            # Build layer matrix: (num_layers, num_chunks, layer_dim)
            self.layer_matrix = torch.stack([
                torch.stack([self.layer_vectors[c][l] for c in range(len(self.doc_ids))])
                for l in range(self.num_layers)
            ]).to(device)  # (L, N, dim)
        else:
            # Fallback to simple pooled vectors
            self.chunk_vectors = pooled_data["pooled_kv"].to(device)
            self.use_mlp_pool = False
            print(f"Loaded {len(self.doc_ids)} chunks with pooled vectors")
        
        # Load projection heads
        print(f"Loading projection heads from: {heads_path}")
        heads_data = torch.load(heads_path, map_location="cpu", weights_only=False)
        
        hidden_size = model.config.hidden_size
        proj_dim = heads_data.get("meta", {}).get("proj_dim", 1024)
        
        # Determine chunk dimension
        if self.use_mlp_pool:
            chunk_dim = layer_dim
        else:
            chunk_dim = self.chunk_vectors.shape[1]
        
        # Initialize heads - use float32 for retrieval precision
        model_dtype = torch.float32
        self.question_head = ProjectionHead(hidden_size, proj_dim).to(device)
        self.chunk_head = ProjectionHead(chunk_dim, proj_dim).to(device)

        self.question_head.load_state_dict(heads_data["question_head"])
        self.chunk_head.load_state_dict(heads_data["chunk_head"])
        self.question_head.eval()
        self.chunk_head.eval()

        # Cast layer matrix to float32
        if self.use_mlp_pool:
            self.layer_matrix = self.layer_matrix.float()

        # Load PoolMLP if using MLP pool
        if self.use_mlp_pool and "pool_mlp" in heads_data:
            self.pool_mlp = PoolMLP(hidden_size, self.num_layers).to(device)
            self.pool_mlp.load_state_dict(heads_data["pool_mlp"])
            self.pool_mlp.eval()
            print(f"Loaded PoolMLP for dynamic layer weighting (dtype={model_dtype})")
        else:
            self.pool_mlp = None
        
        # Pre-compute chunk projections if not using dynamic pooling
        if not self.use_mlp_pool:
            with torch.no_grad():
                self.projected_chunks = self.chunk_head(self.chunk_vectors)
        
        print(f"Retriever initialized with {len(self.doc_ids)} chunks")

    def encode_question(self, question: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode question, return (projected, raw_embedding)."""
        tokens = self.tokenizer(
            question, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=False
        )
        input_ids = tokens["input_ids"].to(self.device)
        
        with torch.no_grad():
            # Get embeddings from embed_tokens (non-contextual, matches training)
            embeddings = self.model.model.embed_tokens(input_ids)
            # Mean pool over tokens
            question_repr = embeddings.mean(dim=1).float()  # (1, hidden_size)
            # Project
            projected = self.question_head(question_repr.to(next(self.question_head.parameters()).dtype))
        
        return projected, question_repr

    def retrieve_top_k(self, question: str, k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve top-k chunks for a question."""
        q_proj, q_repr = self.encode_question(question)  # (1, proj_dim), (1, hidden)
        
        with torch.no_grad():
            if self.use_mlp_pool and self.pool_mlp is not None:
                # Dynamic layer weighting
                layer_weights = self.pool_mlp(q_repr)  # (1, num_layers)
                
                # Weighted sum of layer vectors for each chunk
                # layer_matrix: (L, N, dim), layer_weights: (1, L)
                weighted = torch.einsum('lnd,bl->bnd', self.layer_matrix, layer_weights)  # (1, N, dim)
                chunk_repr = weighted.squeeze(0)  # (N, dim)
                
                # Project chunks
                projected_chunks = self.chunk_head(chunk_repr)  # (N, proj_dim)
            else:
                projected_chunks = self.projected_chunks
            
            # Compute similarities
            similarities = torch.matmul(q_proj, projected_chunks.T).squeeze(0)  # (N,)
            
            # Get top-k
            top_k_scores, top_k_indices = torch.topk(similarities, k=min(k, len(self.doc_ids)))
        
        results = []
        for idx, score in zip(top_k_indices.tolist(), top_k_scores.tolist()):
            results.append((self.doc_ids[idx], score))
        
        return results

    def _search_cache_in_dir(self, doc_id: str, cache_dir: Path) -> Path:
        """Search for a cache file in a single directory."""
        # Check combined format: doc_id/train/**/cache_last.pt
        combined_dir = cache_dir / doc_id
        if combined_dir.exists():
            cache_files = list(combined_dir.glob("train/**/cache_last.pt"))
            if cache_files:
                return cache_files[-1]
            cache_files = list(combined_dir.glob("train/**/cache-step*.pt"))
            if cache_files:
                cache_files.sort()
                return cache_files[-1]

        # Check merged format: doc_id_timestamp/training_output/**/cache_last.pt
        doc_dirs = list(cache_dir.glob(f"{doc_id}_*"))
        for doc_dir in doc_dirs:
            cache_files = list(doc_dir.glob("training_output/*/*/cache_last.pt"))
            if cache_files:
                return cache_files[0]
            cache_files = list(doc_dir.glob("training_output/*/*/cache-step*.pt"))
            if cache_files:
                cache_files.sort()
                return cache_files[-1]

        patterns = [
            cache_dir / doc_id / "cache_last.pt",
            cache_dir / doc_id / "cache-step100.pt",
            cache_dir / f"{doc_id}.pt",
        ]
        for pattern in patterns:
            if pattern.exists():
                return pattern

        chunk_dir = cache_dir / doc_id
        if chunk_dir.exists():
            cache_files = list(chunk_dir.glob("cache-step*.pt"))
            if cache_files:
                cache_files.sort()
                return cache_files[-1]
        return None

    def get_cache_path(self, doc_id: str) -> Path:
        """Get cache file path for a document ID.

        Routes qualitative docs to p-init cache dir, others to numcache dirs.
        """
        # If doc is qualitative and we have a p-init dir, search there first
        if self.pinit_cache_dir and doc_id in QUALITATIVE_DOCS:
            result = self._search_cache_in_dir(doc_id, self.pinit_cache_dir)
            if result:
                return result

        # Search numcache dirs
        for cache_dir in self.cache_dirs:
            result = self._search_cache_in_dir(doc_id, cache_dir)
            if result:
                return result

        # Fallback: if qualitative doc not found in p-init, try numcache
        if self.pinit_cache_dir and doc_id in QUALITATIVE_DOCS:
            pass  # already tried above, fall through to error
        elif self.pinit_cache_dir:
            # Non-qualitative doc not found in numcache, try p-init as last resort
            result = self._search_cache_in_dir(doc_id, self.pinit_cache_dir)
            if result:
                return result

        raise FileNotFoundError(f"No cache found for {doc_id} in {self.cache_dirs}")


def inference_with_retrieval(
    question: str,
    retriever: ContrastiveRetriever,
    model,
    tokenizer,
    top_k: int = 3,
    max_new_tokens: int = 512,
) -> Tuple[str, List[Tuple[str, float]], Dict[str, float]]:
    """
    1. Use contrastive retrieval to find relevant chunks
    2. Load and concatenate their KV caches
    3. Generate answer
    
    Returns:
        answer: Generated answer string
        retrieved: List of (doc_id, score) tuples
        latency: Dict with latency breakdown and throughput
    """
    latency = {}
    total_start = time.perf_counter()
    
    # Step 1: Retrieve
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    print(f"Retrieving top-{top_k} chunks...")
    
    retrieval_start = time.perf_counter()
    retrieved = retriever.retrieve_top_k(question, k=top_k)
    retrieval_end = time.perf_counter()
    latency['retrieval_latency_ms'] = (retrieval_end - retrieval_start) * 1000
    
    print("Retrieved chunks:")
    for doc_id, score in retrieved:
        print(f"  - {doc_id}: {score:.4f}")
    print(f"  Retrieval latency: {latency['retrieval_latency_ms']:.2f} ms")
    
    # Step 2: Load caches
    cache_load_start = time.perf_counter()
    caches = []
    for doc_id, score in retrieved:
        try:
            cache_path = retriever.get_cache_path(doc_id)
            cache = TrainableCache.from_pretrained(str(cache_path), device="cuda")
            cache = cache.to("cuda")
            caches.append(cache)
            print(f"  ✓ Loaded cache: {doc_id} ({cache.num_tokens()} tokens)")
        except FileNotFoundError as e:
            print(f"  ✗ Cache not found: {doc_id}")
    
    if not caches:
        raise RuntimeError("No caches found for retrieved chunks!")
    
    # Step 3: Concatenate caches
    combined_cache = concatenate_caches(caches)
    cache_load_end = time.perf_counter()
    latency['cache_load_latency_ms'] = (cache_load_end - cache_load_start) * 1000
    print(f"Combined cache: {combined_cache.num_tokens()} tokens")
    print(f"  Cache load latency: {latency['cache_load_latency_ms']:.2f} ms")
    
    # Step 4: Generate answer
    generation_start = time.perf_counter()
    answer, full_text = generate_answer(
        model, combined_cache, tokenizer,
        question, max_new_tokens
    )
    generation_end = time.perf_counter()
    latency['generation_latency_ms'] = (generation_end - generation_start) * 1000
    
    total_end = time.perf_counter()
    latency['total_latency_ms'] = (total_end - total_start) * 1000
    
    # Calculate throughput (questions per second)
    latency['throughput_qps'] = 1000.0 / latency['total_latency_ms'] if latency['total_latency_ms'] > 0 else 0
    
    print(f"\nLatency & Throughput:")
    print(f"  Retrieval:   {latency['retrieval_latency_ms']:8.2f} ms")
    print(f"  Cache Load:  {latency['cache_load_latency_ms']:8.2f} ms")
    print(f"  Generation:  {latency['generation_latency_ms']:8.2f} ms")
    print(f"  Total:       {latency['total_latency_ms']:8.2f} ms")
    print(f"  Throughput:  {latency['throughput_qps']:8.2f} questions/sec")
    
    return answer, retrieved, latency


def main():
    parser = argparse.ArgumentParser(description="Inference with contrastive retrieval")
    parser.add_argument("--pooled_kv_path", type=str, required=True,
                       help="Path to pooled_kv_with_layers.pt or pooled_kv_from_cache.pt")
    parser.add_argument("--heads_path", type=str, required=True,
                       help="Path to projection_heads.pt")
    parser.add_argument("--cache_dir", type=str, nargs='+', required=True,
                       help="Directory(ies) containing numcache chunk folders (searched in order)")
    parser.add_argument("--pinit_cache_dir", type=str, default=None,
                       help="Directory containing p-init caches for qualitative docs")
    parser.add_argument("--question", type=str, default="",
                       help="Single question to ask")
    parser.add_argument("--qa_pairs_file", type=str, default="",
                       help="JSON file with QA pairs for batch mode")
    parser.add_argument("--output_file", type=str, default="retrieval_inference_results.json",
                       help="Output JSON file")
    parser.add_argument("--top_k", type=int, default=3,
                       help="Number of chunks to retrieve")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Max tokens to generate")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4b")
    parser.add_argument("--batch_mode", action="store_true")
    parser.add_argument("--use_mlp_pool", action="store_true", default=True,
                       help="Use MLP pooling (default: True)")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = FlexQwen3ForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,  # Must match cache KV tensor dtype
        device_map="cuda",
        trust_remote_code=True
    )
    model.eval()
    
    # Initialize retriever
    retriever = ContrastiveRetriever(
        model=model,
        tokenizer=tokenizer,
        pooled_kv_path=args.pooled_kv_path,
        heads_path=args.heads_path,
        cache_dir=args.cache_dir,
        pinit_cache_dir=args.pinit_cache_dir,
        use_mlp_pool=args.use_mlp_pool,
    )
    
    if args.batch_mode and args.qa_pairs_file:
        # Batch processing
        with open(args.qa_pairs_file, 'r') as f:
            qa_pairs = json.load(f)
        
        results = []
        all_timings = []
        for i, qa in enumerate(qa_pairs):
            question = qa['question']
            print(f"\nProcessing {i+1}/{len(qa_pairs)}")
            
            try:
                answer, retrieved, timing = inference_with_retrieval(
                    question, retriever, model, tokenizer,
                    top_k=args.top_k, max_new_tokens=args.max_tokens
                )
                results.append({
                    'question': question,
                    'ground_truth': qa.get('answer', ''),
                    'generated_answer': answer,
                    'key_points': qa.get('key_points', []),
                    'q_id': qa.get('q_id', ''),
                    'retrieved_chunks': [{'doc_id': d, 'score': s} for d, s in retrieved],
                    'latency': timing
                })
                all_timings.append(timing)
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    'question': question,
                    'ground_truth': qa.get('answer', ''),
                    'key_points': qa.get('key_points', []),
                    'q_id': qa.get('q_id', ''),
                    'error': str(e)
                })
        
        # Compute aggregate latency and throughput statistics
        if all_timings:
            # Latency keys (exclude throughput_qps for averaging)
            latency_keys = ['retrieval_latency_ms', 'cache_load_latency_ms', 'generation_latency_ms', 'total_latency_ms']
            
            avg_latency = {
                key: sum(t[key] for t in all_timings) / len(all_timings)
                for key in latency_keys
            }
            total_latency = {
                key: sum(t[key] for t in all_timings)
                for key in latency_keys
            }
            
            # Throughput calculations
            # Average throughput per question
            avg_throughput = 1000.0 / avg_latency['total_latency_ms'] if avg_latency['total_latency_ms'] > 0 else 0
            # Effective throughput (total questions / total time)
            effective_throughput = len(all_timings) * 1000.0 / total_latency['total_latency_ms'] if total_latency['total_latency_ms'] > 0 else 0
            
            print(f"\n{'='*60}")
            print("Aggregate Latency & Throughput Statistics")
            print(f"{'='*60}")
            print(f"Average latency per question:")
            for key, val in avg_latency.items():
                print(f"  {key}: {val:.2f} ms")
            print(f"\nTotal latency across {len(all_timings)} questions:")
            for key, val in total_latency.items():
                print(f"  {key}: {val:.2f} ms ({val/1000:.2f} s)")
            print(f"\nThroughput:")
            print(f"  Average throughput:   {avg_throughput:.3f} questions/sec")
            print(f"  Effective throughput: {effective_throughput:.3f} questions/sec")
            
            # Add summary to output
            summary = {
                'num_questions': len(all_timings),
                'average_latency': avg_latency,
                'total_latency': total_latency,
                'average_throughput_qps': avg_throughput,
                'effective_throughput_qps': effective_throughput
            }
        else:
            summary = {}
        
        output_data = {
            'results': results,
            'timing_summary': summary
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved {len(results)} results to {args.output_file}")
    
    elif args.question:
        # Single question
        answer, retrieved, timing = inference_with_retrieval(
            args.question, retriever, model, tokenizer,
            top_k=args.top_k, max_new_tokens=args.max_tokens
        )
        print(f"\nAnswer: {answer}")
    
    else:
        print("Please provide --question or --batch_mode with --qa_pairs_file")


if __name__ == "__main__":
    main()