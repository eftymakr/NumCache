"""
Train original cartridges with numcache (number-preserving) initialization.
Uses the same synthesis data as the p-init training, but initializes the KV cache
from numcache-compressed text instead of raw document text.
"""
import os, sys, json, glob, subprocess, tempfile
from pathlib import Path

# Config
CARTRIDGES_DIR = "/home/eftychia/Financial-QA-Benchmark-with-KV-cache/cartridge-original/cartridges"
OUTPUT_DIR = "/home/eftychia/Financial-QA-Benchmark-with-KV-cache/original_cartridges_numcache_output"
SYNTH_DIR = "/home/eftychia/Financial-QA-Benchmark-with-KV-cache/original_cartridges_output"
MODEL_NAME = "Qwen/Qwen3-4b"
EPOCHS = 2
GLOBAL_BATCH_SIZE = 32

os.environ["CARTRIDGES_DIR"] = CARTRIDGES_DIR
os.environ["CARTRIDGES_OUTPUT_DIR"] = OUTPUT_DIR


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-ids-file", type=str, default="/tmp/needed_145_docs.txt")
    parser.add_argument("--compressed-map", type=str, default="/tmp/doc_to_compressed.json")
    args = parser.parse_args()

    # Load doc list
    with open(args.doc_ids_file) as f:
        doc_ids = [l.strip() for l in f if l.strip()]

    # Load compressed text mapping
    with open(args.compressed_map) as f:
        doc_to_compressed = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success = 0
    failed = []

    for i, doc_id in enumerate(doc_ids):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(doc_ids)}] Processing {doc_id}")
        print(f"{'='*60}")

        # Check if already done
        train_output = os.path.join(OUTPUT_DIR, doc_id, "train")
        cache_files = glob.glob(os.path.join(train_output, "**", "cache-step*.pt"), recursive=True)
        if cache_files:
            print(f"[{doc_id}] Training already done, skipping")
            success += 1
            continue

        # Find synthesis parquet (reuse from p-init run)
        synth_parquets = glob.glob(
            os.path.join(SYNTH_DIR, doc_id, "synthesize", "**", "dataset.parquet"),
            recursive=True
        )
        if not synth_parquets:
            print(f"[{doc_id}] No synthesis parquet found, skipping")
            failed.append(doc_id)
            continue
        synth_parquet = synth_parquets[0]

        # Get compressed text path
        compressed_path = doc_to_compressed.get(doc_id)
        if not compressed_path or not os.path.exists(compressed_path):
            print(f"[{doc_id}] No compressed text found, skipping")
            failed.append(doc_id)
            continue

        # Read compressed text to get token count for max_tokens
        # Use max_tokens=None since the text is already compressed to target size
        print(f"[{doc_id}] Compressed text: {compressed_path}")
        print(f"[{doc_id}] Starting training (numcache init)...")

        os.makedirs(train_output, exist_ok=True)

        # Build training script for subprocess
        train_script = f'''
import os, sys
os.environ["CARTRIDGES_DIR"] = "{CARTRIDGES_DIR}"
os.environ["CARTRIDGES_OUTPUT_DIR"] = "{OUTPUT_DIR}"
sys.path.insert(0, os.environ["CARTRIDGES_DIR"])

import pydrantic
from cartridges.train import TrainConfig
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.initialization import KVFromText
from cartridges.datasets import DataSource, TrainDataset

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="{MODEL_NAME}",
        model_cls=FlexQwen3ForCausalLM,
    ),
    kv_cache_initializer=KVFromText.Config(
        text_source="{compressed_path}",
        max_tokens=None,
    ),
    dataset=TrainDataset.Config(
        data_sources=[DataSource(path="{synth_parquet}", type="local")],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),
    output_dir="{train_output}",
    name="train_{doc_id}_numcache",
    lr=2e-2,
    epochs={EPOCHS},
    global_batch_size={GLOBAL_BATCH_SIZE},
    save_every_n_steps=512,
    save_after_training=True,
    distributed_backend="gloo",
)
pydrantic.main(config)
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(train_script)
            script_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
            )
        finally:
            os.unlink(script_path)

        if result.returncode == 0:
            print(f"[{doc_id}] Training complete")
            success += 1
        else:
            print(f"[{doc_id}] ERROR: Training subprocess failed with return code {result.returncode}")
            failed.append(doc_id)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"  Success: {success}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print(f"  Failed docs: {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
