"""
TrimKV-based KV Cache Initialization

Uses TrimKV's pretrained retention gate to select the most important KV pairs
from the full document, then saves them as initialization for cartridges training.

Flow:
1. Load TrimKV model (Qwen3-4B + pretrained retention gate)
2. For each document: feed full text → retention gate scores each KV pair
3. Select top-k KV pairs per layer based on retention scores
4. Save selected KV pairs as cache_init.pt for cartridges training

Usage:
    python trimkv_init_caches.py --doc-ids doc_000009 doc_000012 ...
    python trimkv_init_caches.py --doc-ids-file /tmp/doc_ids.txt
"""

import os
import sys
import json
import glob
import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

# Patch TrimKV imports to skip VL models and flash_attn
TRIMKV_SRC = os.path.join(os.path.dirname(__file__), "trimkv-dev/src")
sys.path.insert(0, TRIMKV_SRC)

def patch_trimkv_imports():
    """Patch TrimKV to skip vision model and flash_attn imports."""
    patches = {
        "trimkv/models/__init__.py": [
            ("from .qwen3_vl import *", "# from .qwen3_vl import *"),
            ("from .llava import *", "# from .llava import *"),
            ("from .llava_next import *", "# from .llava_next import *"),
            ("from .qwen2_5_vl import *", "# from .qwen2_5_vl import *"),
        ],
        "trimkv/attn/__init__.py": [
            ("from . import flash_attn", "# from . import flash_attn"),
            ('"dbtrimkv_flash": flash_attn.dynamic_kv_budget_attention_forward,', '# "dbtrimkv_flash": ...,'),
            ('"bdbtrimkv_flash": flash_attn.batched_dynamic_kv_budget_attention_forward,', '# "bdbtrimkv_flash": ...,'),
        ],
    }
    originals = {}
    for filepath, replacements in patches.items():
        full_path = os.path.join(TRIMKV_SRC, filepath)
        with open(full_path) as f:
            originals[filepath] = f.read()
        content = originals[filepath]
        for old, new in replacements:
            content = content.replace(old, new)
        with open(full_path, 'w') as f:
            f.write(content)
    return originals


def restore_trimkv_imports(originals):
    """Restore original TrimKV import files."""
    for filepath, content in originals.items():
        full_path = os.path.join(TRIMKV_SRC, filepath)
        with open(full_path, 'w') as f:
            f.write(content)


BASE_DIR = Path(__file__).parent
CORPUS_FILE = BASE_DIR / "vlo_psx_benchmark_corpus.jsonl"
TRIMKV_WEIGHTS_DIR = BASE_DIR / "trimkv_pretrained_converted"
OUTPUT_DIR = BASE_DIR / "trimkv_init_caches"


def load_corpus():
    """Load document texts from corpus."""
    docs = {}
    with open(CORPUS_FILE) as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc.get("_id", doc.get("doc_id", ""))
            text = doc.get("text", doc.get("contents", ""))
            docs[doc_id] = text
    return docs


def get_target_cache_size(doc_id, compression_ratio=4.0):
    """Get target cache size from existing compressed text or compute it."""
    # Check existing compressed files
    compressed_files = glob.glob(
        str(BASE_DIR / "automated_runs" / f"{doc_id}_*" / "compressed" / "*compressed.txt")
    )
    if not compressed_files:
        compressed_files = glob.glob(
            str(Path("/ext/peiwenfiles/automated_runs_merged") / f"{doc_id}_*" / "compressed" / "*compressed.txt")
        )

    if compressed_files:
        # Get the token count from the compressed file's companion meta
        meta_files = [f + ".meta.json" if os.path.exists(f + ".meta.json")
                     else f.replace("_compressed.txt", "_compressed.meta.json")
                     for f in compressed_files]
        # Just count tokens in the compressed file
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4b")
        with open(sorted(compressed_files)[-1]) as f:
            compressed_text = f.read()
        return len(tokenizer.encode(compressed_text))

    return None


def process_document(model, tokenizer, doc_text, target_cache_size, device):
    """
    Process a document through TrimKV and run the original compress() routine.

    Uses Ngoc's exact scoring:
        log_alpha = retention_weights * (q_idx - kv_positions)
    with per-head top-k selection (each head retains its own positions).

    Returns lists of (1, n_kv_heads, K, head_dim) keys/values per layer.
    """
    # Tokenize the full document as system prompt
    messages = [{"role": "system", "content": doc_text}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    n_tokens = input_ids.shape[1]
    print(f"    Full doc tokens: {n_tokens}, target cache: {target_cache_size}")

    # Disable auto-compression so we can drive compress() with our per-doc target.
    prev_compress_memory = model.config.compress_memory
    model.config.compress_memory = False

    try:
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True,
                )
    finally:
        model.config.compress_memory = prev_compress_memory

    past_kv = outputs.past_key_values  # TrimKVCache

    n_layers = len(past_kv.key_cache)
    n_kv_heads = past_kv.key_cache[0].shape[1]

    # Apply Ngoc's compression: per-head topk on log_alpha = rw * (q_idx - kv_pos).
    # compress() only fires per layer when memory_size + buffer_size <= cache_length,
    # so docs shorter than target_cache_size are kept in full automatically.
    past_kv.compress(
        strategy=model.config.compress_strategy,           # "alpha"
        memory_size=target_cache_size,                     # per-head budget
        buffer_size=0,                                     # we control the budget directly
        floor_budget_ratio=getattr(model.config, "floor_budget_ratio", 0.),
        alpha_threshold=getattr(model.config, "alpha_threshold", 0.0),
        num_layers=n_layers,
        num_key_value_heads=n_kv_heads,
        skip_layers=getattr(model.config, "skip_layers", 0),
    )

    selected_keys = [past_kv.key_cache[layer_idx].detach().cpu() for layer_idx in range(n_layers)]
    selected_values = [past_kv.value_cache[layer_idx].detach().cpu() for layer_idx in range(n_layers)]

    # Sanity: cartridges expects the same K across layers.
    shapes = {k.shape for k in selected_keys}
    assert len(shapes) == 1, f"Per-layer shape mismatch after compress(): {shapes}"

    return selected_keys, selected_values


def main():
    parser = argparse.ArgumentParser(description="TrimKV-based cache initialization")
    parser.add_argument("--doc-ids", nargs="+", help="Document IDs to process")
    parser.add_argument("--doc-ids-file", type=str, help="File with doc IDs (one per line)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--compression-ratio", type=float, default=4.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--base-model-override", type=str, default=None,
                        help="Load this model for the transformer body instead of config.base_model "
                             "(keeps the retention gate). Use to generate init KVs in the SAME space "
                             "as the cache-training model, e.g. Qwen/Qwen3-4b.")
    args = parser.parse_args()

    # Get doc IDs
    if args.doc_ids:
        doc_ids = args.doc_ids
    elif args.doc_ids_file:
        with open(args.doc_ids_file) as f:
            doc_ids = [l.strip() for l in f if l.strip()]
    else:
        parser.error("Specify --doc-ids or --doc-ids-file")

    device = f"cuda:{args.gpu}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load corpus
    print("Loading corpus...")
    corpus = load_corpus()

    # Patch and load TrimKV
    print("Loading TrimKV model...")
    originals = patch_trimkv_imports()

    try:
        from trimkv.models.qwen3.configuration_trimkv_qwen3 import TrimKVQwen3Config
        from trimkv.models.qwen3.modeling_trimkv_qwen3 import TrimKVQwen3ForCausalLM

        config = TrimKVQwen3Config.from_pretrained(str(TRIMKV_WEIGHTS_DIR))
        body_model = args.base_model_override or config.base_model
        print(f"  Transformer body: {body_model}  (gate from {TRIMKV_WEIGHTS_DIR})")
        model = TrimKVQwen3ForCausalLM.from_pretrained(
            body_model,
            load_trimkv_weights=False,
            config=config,
            dtype=torch.bfloat16,
        ).to(device)

        # Load retention gate weights (trained on config.base_model; reused on body_model)
        gate_weights = torch.load(
            str(TRIMKV_WEIGHTS_DIR / "trimkv_weights.pth"),
            map_location='cpu'
        )
        load_res = model.load_state_dict(gate_weights, strict=False)
        gate_missing = [k for k in load_res.missing_keys if "retention_gate" in k]
        assert not gate_missing, f"retention_gate weights failed to load: {gate_missing[:3]}"
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(body_model)

        print(f"Model loaded on {device}")
        print(f"Processing {len(doc_ids)} documents...\n")

        success = 0
        failed = []

        for i, doc_id in enumerate(doc_ids):
            print(f"[{i+1}/{len(doc_ids)}] {doc_id}")

            if doc_id not in corpus:
                print(f"  SKIP: not in corpus")
                failed.append(doc_id)
                continue

            doc_text = corpus[doc_id]

            # Get target cache size
            target = get_target_cache_size(doc_id, args.compression_ratio)
            if target is None:
                # Compute from doc length
                doc_tokens = len(tokenizer.encode(doc_text))
                target = max(64, doc_tokens // int(args.compression_ratio))
                print(f"  Computed target: {doc_tokens} / {args.compression_ratio} = {target}")

            try:
                selected_keys, selected_values = process_document(
                    model, tokenizer, doc_text, target, device
                )

                # Save as cache init
                doc_output_dir = os.path.join(args.output_dir, doc_id)
                os.makedirs(doc_output_dir, exist_ok=True)

                cache_data = {
                    'keys': selected_keys,
                    'values': selected_values,
                    'n_tokens': selected_keys[0].shape[2],
                    'n_layers': len(selected_keys),
                    'doc_id': doc_id,
                    'init_method': 'trimkv',
                }

                save_path = os.path.join(doc_output_dir, "trimkv_cache_init.pt")
                torch.save(cache_data, save_path)

                print(f"  Saved: {save_path} ({selected_keys[0].shape[2]} tokens)")
                success += 1

                # Clear GPU cache
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ERROR: {e}")
                failed.append(doc_id)
                torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print(f"Done: {success} succeeded, {len(failed)} failed")
        if failed:
            print(f"Failed: {failed}")

    finally:
        restore_trimkv_imports(originals)


if __name__ == "__main__":
    main()
