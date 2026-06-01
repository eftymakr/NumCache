"""
Train original cartridges with p-initialization and CE loss (instead of KL).
Uses the same synthesis data as the KL training, but with standard cross-entropy loss.
"""
import os, sys, json, glob, subprocess, tempfile
from pathlib import Path
from transformers import AutoTokenizer

# Config
CARTRIDGES_DIR = "/home/eftychia/Financial-QA-Benchmark-with-KV-cache/cartridge-original/cartridges"
OUTPUT_DIR = "/home/eftychia/Financial-QA-Benchmark-with-KV-cache/original_cartridges_ce_output"
SYNTH_DIR = "/home/eftychia/Financial-QA-Benchmark-with-KV-cache/original_cartridges_output"
DOCS_DIR = "/tmp/vlo_psx_docs"
MODEL_NAME = "Qwen/Qwen3-4b"
EPOCHS = 2
GLOBAL_BATCH_SIZE = 32
COMPRESSION_RATIO = 4

os.environ["CARTRIDGES_DIR"] = CARTRIDGES_DIR
os.environ["CARTRIDGES_OUTPUT_DIR"] = OUTPUT_DIR

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def get_max_tokens(doc_path):
    with open(doc_path) as f:
        text = f.read()
    token_count = len(tokenizer.encode(text))
    compressed = token_count // COMPRESSION_RATIO
    return max(256, compressed)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-ids-file", type=str, default="/tmp/needed_145_docs.txt")
    args = parser.parse_args()

    with open(args.doc_ids_file) as f:
        doc_ids = [l.strip() for l in f if l.strip()]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success = 0
    failed = []

    for i, doc_id in enumerate(doc_ids):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(doc_ids)}] Processing {doc_id}")
        print(f"{'='*60}")

        train_output = os.path.join(OUTPUT_DIR, doc_id, "train")
        cache_files = glob.glob(os.path.join(train_output, "**", "cache-step*.pt"), recursive=True)
        if cache_files:
            print(f"[{doc_id}] Training already done, skipping")
            success += 1
            continue

        # Find synthesis parquet (reuse from KL run)
        synth_parquets = glob.glob(
            os.path.join(SYNTH_DIR, doc_id, "synthesize", "**", "dataset.parquet"),
            recursive=True
        )
        if not synth_parquets:
            print(f"[{doc_id}] No synthesis parquet found, skipping")
            failed.append(doc_id)
            continue
        synth_parquet = synth_parquets[0]

        doc_path = os.path.join(DOCS_DIR, f"{doc_id}.txt")
        if not os.path.exists(doc_path):
            print(f"[{doc_id}] Doc not found at {doc_path}, skipping")
            failed.append(doc_id)
            continue

        max_tokens = get_max_tokens(doc_path)
        print(f"[{doc_id}] Starting CE training (p-init, max_tokens={max_tokens})...")

        os.makedirs(train_output, exist_ok=True)

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
        text_source="{doc_path}",
        max_tokens={max_tokens},
    ),
    dataset=TrainDataset.Config(
        data_sources=[DataSource(path="{synth_parquet}", type="local")],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),
    output_dir="{train_output}",
    name="train_{doc_id}_pinit_ce",
    lr=2e-2,
    loss_type="ce",
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
