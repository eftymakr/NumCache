#!/usr/bin/env python3
"""
Automated Training Pipeline - By Doc ID

This script automates the entire training workflow from GENERATION_WORKFLOW.md:
1. Read document from qa_output/documents/{doc_id}.json
2. Get text from enhanced corpus using doc_id
3. Tokenize with Qwen 3-4b
4. Compress with compressor v3 (4x compression)
5. Train/val split (configurable ratio, reasoning/synthesis only in train)
6. Convert data to training format using convert_qa_data.py
7. Update train config (PKL paths, context path, cache size)
8. Run training via cartridges/train.py, then automatically start next run

Documents are stored as: qa_generation_full_corpus/qa_output/documents/doc_XXXXXX.json

Usage:
    # Single doc_id
    python automated_training_pipeline.py --doc-id doc_000000 --compression-ratio 4 --train-ratio 0.9
    
    # Multiple doc_ids (runs sequentially)
    python automated_training_pipeline.py --doc-ids doc_000000 doc_000001 doc_000002
    
    # Range of doc_ids
    python automated_training_pipeline.py --doc-range 0 100 --compression-ratio 4
"""

import os
import sys
import json
import subprocess
import time
import shutil
import argparse
import pickle
import random
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


# Unicode normalization mapping: fancy chars -> ASCII equivalents
# LLMs often generate these fancy characters which tokenize differently
UNICODE_REPLACEMENTS = {
    '\u2011': '-',    # Non-breaking hyphen -> regular hyphen
    '\u2010': '-',    # Hyphen -> regular hyphen  
    '\u2012': '-',    # Figure dash -> regular hyphen
    '\u2013': '-',    # En dash -> regular hyphen
    '\u2014': '-',    # Em dash -> regular hyphen
    '\u2015': '-',    # Horizontal bar -> regular hyphen
    '\u2212': '-',    # Minus sign -> regular hyphen
    '\u202f': ' ',    # Narrow no-break space -> regular space
    '\u00a0': ' ',    # Non-breaking space -> regular space
    '\u2009': ' ',    # Thin space -> regular space
    '\u200a': ' ',    # Hair space -> regular space
    '\u2002': ' ',    # En space -> regular space
    '\u2003': ' ',    # Em space -> regular space
    '\u201c': '"',    # Left double curly quote -> regular quote
    '\u201d': '"',    # Right double curly quote -> regular quote
    '\u201e': '"',    # Double low-9 quotation mark -> regular quote
    '\u201f': '"',    # Double high-reversed-9 quotation mark -> regular quote
    '\u2018': "'",    # Left single curly quote -> apostrophe
    '\u2019': "'",    # Right single curly quote -> apostrophe
    '\u201a': "'",    # Single low-9 quotation mark -> apostrophe
    '\u201b': "'",    # Single high-reversed-9 quotation mark -> apostrophe
    '\u2032': "'",    # Prime -> apostrophe
    '\u2033': '"',    # Double prime -> regular quote
    '\u2026': '...',  # Horizontal ellipsis -> three dots
}


def normalize_text(text: str) -> str:
    """Normalize text by replacing fancy Unicode characters with ASCII equivalents."""
    if not text:
        return text
    for unicode_char, ascii_char in UNICODE_REPLACEMENTS.items():
        text = text.replace(unicode_char, ascii_char)
    return text


def normalize_qa_data(qa_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize all text fields in QA data to fix Unicode chars."""
    normalized = []
    for qa in qa_list:
        qa_copy = qa.copy()
        if 'question' in qa_copy:
            qa_copy['question'] = normalize_text(qa_copy['question'])
        if 'answer' in qa_copy:
            qa_copy['answer'] = normalize_text(qa_copy['answer'])
        normalized.append(qa_copy)
    return normalized


# Base paths
BASE_DIR = Path("/home/eftychia/Financial-QA-Benchmark-with-KV-cache")
QA_OUTPUT_DIR = BASE_DIR / "qa_generation_full_corpus" / "qa_output"
DOCUMENTS_DIR = QA_OUTPUT_DIR / "documents"
ENHANCED_CORPUS = BASE_DIR / "qa&corpus" / "qa" / "enhanced_corpus_new.jsonl"
OUTPUT_BASE_DIR = BASE_DIR / "automated_runs"


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    doc_id: str                          # Document ID (e.g., doc_000000)
    compression_ratio: float = 4.0       # Compression ratio (e.g., 4 = 4x compression)
    train_ratio: float = 0.9             # Train/val split ratio (0.9 = 90% train, 10% eval)
    train_templates: List[str] = None    # Templates for train (default: reasoning, synthesis)
    tokenizer_name: str = "Qwen/Qwen3-4b"
    epochs: int = 5
    lr: float = 2e-2
    global_batch_size: int = 32
    max_concurrent_runs: int = 1         # Run sequentially by default
    training_mode: str = "standard_ce"   # Training mode: standard_ce, teacher_forcing, extended_ntl
    packed_seq_length: int = 2048        # Sequence length for packing
    freeze_keys: bool = False            # Freeze key vectors, only train values
    init_method: str = "numcache"        # Initialization method: numcache (compressed) or pinit (first-p tokens)
    qa_file: str = None                  # Optional: external QA file (JSON list with doc_id field) instead of per-doc qa_output
    output_dir: str = None               # Optional: custom output base directory
    
    def __post_init__(self):
        if self.train_templates is None:
            self.train_templates = ["reasoning", "synthesis"]


class AutomatedPipeline:
    """Automated training pipeline orchestrator - processes by doc_id"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Determine output directory
        if config.output_dir:
            output_base = Path(config.output_dir)
        elif config.init_method == "pinit":
            output_base = BASE_DIR / "automated_runs_p_compression"
        else:
            output_base = OUTPUT_BASE_DIR
        self.run_dir = output_base / f"{config.doc_id}_{self.run_timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.log_file = self.run_dir / "pipeline.log"
        
        # Paths for this run
        self.doc_json_file = DOCUMENTS_DIR / f"{config.doc_id}.json"
        self.compressed_dir = self.run_dir / "compressed"
        self.qa_output_dir = self.run_dir / "qa_output"
        self.train_eval_dir = self.run_dir / "train_eval_split"
        self.config_file = self.run_dir / "train_config.py"
        
        # Document data (loaded in step 1)
        self.doc_data = None
        self.doc_text = None
        
    def log(self, message: str, level: str = "INFO"):
        """Log message to console and file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
    
    def run_command(self, cmd: List[str], cwd: str = None) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr"""
        self.log(f"Running: {' '.join(cmd)}")
        # Pass environment variables including CARTRIDGES_DIR
        env = os.environ.copy()
        env['CARTRIDGES_DIR'] = str(BASE_DIR)
        env['CARTRIDGES_OUTPUT_DIR'] = str(BASE_DIR / 'outputs')
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or str(BASE_DIR),
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
                env=env
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            self.log("Command timed out!", "ERROR")
            return -1, "", "Timeout"
        except Exception as e:
            self.log(f"Command failed: {e}", "ERROR")
            return -1, "", str(e)
    
    # =========================================================================
    # Step 1: Load document from qa_output/documents/{doc_id}.json
    # =========================================================================
    def step1_load_document(self) -> Dict[str, Any]:
        """Load document JSON and get text from enhanced corpus"""
        self.log(f"Step 1: Loading document {self.config.doc_id}")
        
        # Check if doc JSON exists
        if not self.doc_json_file.exists():
            if self.config.qa_file:
                self.log(f"  Per-doc JSON not found, using external QA file: {self.config.qa_file}")
                self.doc_data = {"qa_pairs": []}  # Will be loaded in step4
            else:
                self.log(f"Document file not found: {self.doc_json_file}", "ERROR")
                return None
        else:
            # Load the document JSON (contains QA pairs and metadata)
            with open(self.doc_json_file, "r") as f:
                self.doc_data = json.load(f)
            self.log(f"  Loaded: {self.doc_data.get('qa_count', 0)} QA pairs")
            self.log(f"  Metadata: {self.doc_data.get('metadata', {})}")
        
        # Get the original text from enhanced corpus
        self.doc_text = self._get_text_from_corpus(self.config.doc_id)
        
        if self.doc_text:
            self.log(f"  Text length: {len(self.doc_text)} chars")
        else:
            self.log(f"  Warning: Could not find text in enhanced corpus", "WARN")
        
        return self.doc_data
    
    def _get_text_from_corpus(self, doc_id: str) -> Optional[str]:
        """Get document text from enhanced_corpus_new.jsonl by doc_id"""
        if not ENHANCED_CORPUS.exists():
            self.log(f"Enhanced corpus not found: {ENHANCED_CORPUS}", "WARN")
            return None
        
        with open(ENHANCED_CORPUS, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("_id") == doc_id:
                        return entry.get("text", "")
                except json.JSONDecodeError:
                    continue
        
        return None
    
    # =========================================================================
    # Step 2: Tokenize with Qwen 3-4b
    # =========================================================================
    def step2_tokenize(self) -> int:
        """Tokenize the document text and return token count"""
        self.log(f"Step 2: Tokenizing with {self.config.tokenizer_name}")
        
        if not self.doc_text:
            self.log("No document text available", "ERROR")
            return 0
        
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name, 
                trust_remote_code=True
            )
        except Exception as e:
            self.log(f"Failed to load tokenizer: {e}", "ERROR")
            return len(self.doc_text) // 4  # Rough estimate
        
        tokens = tokenizer.encode(self.doc_text, add_special_tokens=False)
        token_count = len(tokens)
        
        # Save token count
        counts_file = self.run_dir / "token_counts.json"
        with open(counts_file, "w") as f:
            json.dump({self.config.doc_id: token_count}, f, indent=2)
        
        self.log(f"  Token count: {token_count:,}")
        
        return token_count
    
    # =========================================================================
    # Step 3: Compress with compressor v3 (using compress_from_json.py)
    # =========================================================================
    def step3_compress(self, token_count: int) -> Tuple[str, int]:
        """Compress document using number_preserving_compressor_v3"""
        self.log(f"Step 3: Compressing with {self.config.compression_ratio}x ratio")
        
        if not self.doc_text:
            self.log("No document text to compress", "ERROR")
            return None, 0
        
        self.compressed_dir.mkdir(exist_ok=True)
        
        # Calculate budget based on compression ratio
        budget = int(token_count / self.config.compression_ratio)
        budget = max(budget, 50)  # Minimum budget
        
        self.log(f"  Original tokens: {token_count:,}")
        self.log(f"  Target budget: {budget:,} tokens")
        
        # Write temp file for compressor
        temp_input = self.compressed_dir / f"{self.config.doc_id}_input.txt"
        with open(temp_input, "w") as f:
            f.write(self.doc_text)
        
        compressed_file = self.compressed_dir / f"{self.config.doc_id}_budget_{budget}_compressed.txt"
        
        # Run compressor v3 (better for flat financial tables per workflow docs)
        cmd = [
            sys.executable, str(BASE_DIR / "number_preserving_compressor_v3.py"),
            "--document", str(temp_input),
            "--budget", str(budget),
            "--tokenizer", self.config.tokenizer_name,
            "--output", str(compressed_file)
        ]
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        if exit_code == 0 and compressed_file.exists():
            with open(compressed_file, "r") as f:
                compressed_text = f.read()
            
            # Count final tokens
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.tokenizer_name, 
                    trust_remote_code=True
                )
                final_tokens = len(tokenizer.encode(compressed_text, add_special_tokens=False))
            except:
                final_tokens = len(compressed_text) // 4
            
            compression_achieved = token_count / final_tokens if final_tokens > 0 else 0
            self.log(f"  Compressed: {final_tokens:,} tokens")
            self.log(f"  Ratio achieved: {compression_achieved:.2f}x")
        else:
            self.log(f"Compression failed: {stderr}", "WARN")
            # Use original text truncated as fallback
            with open(compressed_file, "w") as f:
                f.write(self.doc_text[:budget * 4])
            final_tokens = budget
        
        # Cleanup temp input
        temp_input.unlink(missing_ok=True)
        
        self.log(f"  Saved to: {compressed_file}")
        return str(compressed_file), final_tokens
    
    # =========================================================================
    # Step 3b: P-initialization (use raw text, no compression)
    # =========================================================================
    def step3_pinit(self, token_count: int) -> Tuple[str, int]:
        """Use raw document text for first-p initialization (no compression)"""
        self.log(f"Step 3: P-initialization (using raw text, no compression)")

        if not self.doc_text:
            self.log("No document text for p-init", "ERROR")
            return None, 0

        self.compressed_dir.mkdir(exist_ok=True)

        # For p-init, use first 1/compression_ratio of tokens (same as old pipeline)
        cache_size = int(token_count / self.config.compression_ratio)

        # Save raw text to file (KVFromText will handle truncation via max_tokens)
        raw_file = self.compressed_dir / f"{self.config.doc_id}_pinit_raw.txt"
        with open(raw_file, "w") as f:
            f.write(self.doc_text)

        self.log(f"  Raw tokens: {token_count:,}")
        self.log(f"  Cache size: {cache_size:,}")
        self.log(f"  Saved to: {raw_file}")
        return str(raw_file), cache_size

    # =========================================================================
    # Step 4: Train/Val Split (using split_qa_train_eval.py logic)
    # =========================================================================
    def step4_train_val_split(self) -> Tuple[str, str]:
        """Split QA data into train/val with specified ratio and template filtering
        
        - Train: ALL templates (factual, reasoning, synthesis, etc.)
        - Eval: Only some QAs from reasoning and synthesis (based on eval_ratio)
        """
        eval_ratio = 1.0 - self.config.train_ratio  # e.g., 0.1 = 10% of reasoning/synthesis go to eval
        self.log(f"Step 4: Train/val split")
        self.log(f"  Eval templates: {self.config.train_templates}")
        self.log(f"  Eval ratio from these: {eval_ratio:.0%}")
        
        self.train_eval_dir.mkdir(exist_ok=True)
        
        # Get QA pairs: from external QA file or loaded document
        if self.config.qa_file:
            with open(self.config.qa_file, "r") as f:
                all_qas = json.load(f)
            qa_data = [qa for qa in all_qas if qa.get("doc_id") == self.config.doc_id]
            self.log(f"Loaded {len(qa_data)} QA pairs for {self.config.doc_id} from {self.config.qa_file}")
        elif self.doc_data and "qa_pairs" in self.doc_data:
            qa_data = self.doc_data["qa_pairs"]
            self.log(f"Loaded {len(qa_data)} QA pairs from {self.config.doc_id}")
        else:
            self.log("No QA data loaded", "ERROR")
            return None, None
        
        # Classify by template type
        template_counts = {}
        for qa in qa_data:
            t = qa.get("template_type", "unknown")
            template_counts[t] = template_counts.get(t, 0) + 1
        self.log(f"Template distribution: {template_counts}")
        
        # ALL QAs go to train
        train_data = list(qa_data)
        
        # Eval: only some reasoning/synthesis QAs
        eval_data = []
        random.seed(42)  # Reproducibility
        
        for qa in qa_data:
            template_type = qa.get("template_type", "unknown")
            
            # Only reasoning and synthesis can go to eval
            if template_type in self.config.train_templates:
                if random.random() < eval_ratio:
                    eval_data.append(qa)
        
        self.log(f"Split result:")
        self.log(f"  Train: {len(train_data)} (all QAs)")
        self.log(f"  Eval: {len(eval_data)} (subset of reasoning/synthesis)")
        
        # Normalize Unicode characters in questions/answers before saving
        # This ensures consistent tokenization matching the original documents
        train_data = normalize_qa_data(train_data)
        eval_data = normalize_qa_data(eval_data)
        self.log(f"  Normalized Unicode characters in QA text")
        
        # Save as JSON for conversion in step 5
        train_json = self.train_eval_dir / "train.json"
        eval_json = self.train_eval_dir / "eval.json"
        
        with open(train_json, "w") as f:
            json.dump(train_data, f, indent=2)
        with open(eval_json, "w") as f:
            json.dump(eval_data, f, indent=2)
        
        return str(train_json), str(eval_json)
    
    # =========================================================================
    # Step 5: Convert Data (using convert_qa_data.py)
    # =========================================================================
    def step5_convert_data(self, train_json: str, eval_json: str) -> Tuple[str, str]:
        """Convert JSON QA data to PKL format for training using convert_qa_data.py
        
        Per workflow Step 3: python convert_qa_data.py --qa-file X --output-file Y --tokenizer-name Qwen/Qwen3-4b
        """
        self.log("Step 5: Converting data to training format (PKL)")
        
        train_pkl = self.train_eval_dir / "train.pkl"
        eval_pkl = self.train_eval_dir / "eval.pkl"
        
        # Use convert_qa_data.py for conversion (as per workflow)
        for input_file, output_file, name in [
            (train_json, train_pkl, "train"), 
            (eval_json, eval_pkl, "eval")
        ]:
            self.log(f"  Converting {name}...")
            cmd = [
                sys.executable, str(BASE_DIR / "convert_qa_data.py"),
                "--file-mode",
                "--qa-file", input_file,
                "--output-file", str(output_file),
                "--tokenizer", self.config.tokenizer_name
            ]
            
            exit_code, stdout, stderr = self.run_command(cmd)
            
            if exit_code != 0:
                self.log(f"Conversion failed for {name}: {stderr}", "WARN")
                self.log("Attempting fallback conversion...")
                self._fallback_convert(input_file, str(output_file))
            else:
                self.log(f"  {name} converted successfully")
        
        # Verify outputs exist
        if train_pkl.exists() and eval_pkl.exists():
            train_size = train_pkl.stat().st_size / 1024
            eval_size = eval_pkl.stat().st_size / 1024
            self.log(f"  train.pkl: {train_size:.1f} KB")
            self.log(f"  eval.pkl: {eval_size:.1f} KB")
        
        return str(train_pkl), str(eval_pkl)
    
    def _fallback_convert(self, json_file: str, pkl_file: str):
        """Fallback conversion if main script fails"""
        with open(json_file, "r") as f:
            data = json.load(f)
        with open(pkl_file, "wb") as f:
            pickle.dump(data, f)
        self.log(f"Used fallback conversion for {json_file}")
    
    # =========================================================================
    # Step 6: Generate Train Config (matching workflow Step 5)
    # =========================================================================
    def step6_generate_config(self, train_pkl: str, eval_pkl: str, 
                               context_path: str, cache_size: int) -> str:
        """Generate training config with updated paths and cache size
        
        Per workflow Step 5: Update train_config.py with:
        - kv_cache_initializer text_source and max_tokens
        - training_mode (teacher_forcing, extended_ntl, or standard_ce)
        - dataset paths for train and eval
        """
        self.log("Step 6: Generating train config")
        self.log(f"  Context path: {context_path}")
        self.log(f"  Cache size: {cache_size:,} tokens")
        self.log(f"  Training mode: {self.config.training_mode}")
        
        # Get metadata for naming
        metadata = self.doc_data.get("metadata", {}) if self.doc_data else {}
        ticker = metadata.get("ticker", "unknown")
        
        config_content = f'''#!/usr/bin/env python3
# Auto-generated training config for {self.config.doc_id}
# Generated at: {datetime.now().isoformat()}
# Based on GENERATION_WORKFLOW.md Step 5
# Ticker: {ticker}

import os
from pathlib import Path
import pydrantic
from peft import LoraConfig

from cartridges.initialization import KVFromText
from cartridges.train import TrainConfig, LossEvalConfig, GenerationEvalConfig 
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.datasets import DataSource, GenerateEvalDataset, TrainDataset, LossEvalDataset
from cartridges.utils.wandb import WandBConfig
from cartridges.models.config import PeftConfig

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="{self.config.tokenizer_name}",
        model_cls=FlexQwen3ForCausalLM,
    ),
    
    # Context initialization (per workflow Step 5)
    kv_cache_initializer=KVFromText.Config(
        text_source="{context_path}",
        max_tokens={cache_size}
    ),
    
    # Training mode: "teacher_forcing" (Cartridges default), "extended_ntl", "standard_ce"
    training_mode="{self.config.training_mode}",
    
    # NTL parameters (for extended_ntl mode)
    ntl_alpha=1.0,        # CE structural
    ntl_beta_ce=1.0,      # CE digits
    ntl_beta_was=0.3,     # Wasserstein digits
    ntl_gamma=1.0,        # CE units
    ntl_delta=1.0,        # CE other
    ntl_use_wasserstein=True,
    use_special_tokens=False,
    
    stochastic_probability=0.1,
    include_reconstruction_qa=True,
    reconstruction_qa_weight=1.0,
    
    # Training hyperparameters
    lr={self.config.lr},
    epochs={self.config.epochs},
    global_batch_size={self.config.global_batch_size},
    early_stopping_patience=5,
    early_stopping_min_delta=0.001,

    # Freeze key vectors (only train values) - keys act as stable routers
    freeze_keys={self.config.freeze_keys},

    # Train dataset (per workflow Step 5)
    dataset=TrainDataset.Config(
        data_sources=[
            DataSource(path="{train_pkl}", type="local"),
        ],
        top_k_logits=20,
        packed_seq_length={self.config.packed_seq_length},
        packing_mode="truncate",
    ),
    
    # Loss evaluation (per workflow Step 5)
    loss_eval_every_n_steps=16,
    loss_evals=[
        LossEvalConfig(
            dataset=LossEvalDataset.Config(
                data_source=DataSource(
                    path="{eval_pkl}",
                    type="local",
                ),
                packed_seq_length={self.config.packed_seq_length},
            ),
            name_for_wandb="{self.config.doc_id}_qa_eval",
        )
    ],

    # Generation evaluation - disabled (testing done in inference)
    # generate_eval_every_n_steps=128,
    # generate_evals=[...],
    
    distributed_backend="gloo",
    save_every_n_steps=512,
    save_to_wandb=True,
    name="{self.config.doc_id}-{ticker}-{self.config.training_mode}-{self.run_timestamp}",
    wandb=WandBConfig(), 
)

if __name__ == "__main__":
    pydrantic.main(config)
'''
        
        with open(self.config_file, "w") as f:
            f.write(config_content)
        
        self.log(f"  Config saved to: {self.config_file}")
        return str(self.config_file)
    
    # =========================================================================
    # Step 7: Run Training (per workflow Step 6)
    # =========================================================================
    def step7_run_training(self, config_path: str) -> bool:
        """Run training with the generated config
        
        Uses micromamba run -n financial-qa to run the training config,
        matching run_training.sh pattern.
        """
        self.log("Step 7: Starting training")
        self.log(f"  Config: {config_path}")
        
        # Set up environment variables like run_training.sh
        output_dir = self.run_dir / "training_output"
        output_dir.mkdir(exist_ok=True)
        
        env = os.environ.copy()
        env["CARTRIDGES_DIR"] = str(BASE_DIR)
        env["CARTRIDGES_OUTPUT_DIR"] = str(output_dir)
        # Add BASE_DIR to PYTHONPATH so cartridges module is found
        pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{BASE_DIR}:{pythonpath}" if pythonpath else str(BASE_DIR)
        
        # Preserve CUDA_VISIBLE_DEVICES if set (for GPU selection)
        # Default to GPU 0 if not specified (GPUs 2,3 often used by vLLM)
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        self.log(f"  Using GPU(s): CUDA_VISIBLE_DEVICES={cuda_devices}")
        
        # Log file for this training run
        log_file = output_dir / f"training_{self.run_timestamp}.log"
        
        # Command using micromamba (matching run_training.sh)
        # Note: env dict is passed to subprocess.Popen which sets CUDA_VISIBLE_DEVICES
        cmd = [
            "/home/eftychia/micromamba", "run", "-n", "financial-qa",
            "python", config_path
        ]
        
        self.log(f"  Output dir: {output_dir}")
        self.log(f"  Log file: {log_file}")
        self.log("=" * 50)
        self.log("Training started... This may take several hours.")
        self.log("Monitor progress via WandB or logs.")
        self.log("=" * 50)
        
        start_time = time.time()
        
        # Run with tee-like logging
        try:
            with open(log_file, "w") as lf:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(BASE_DIR),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                for line in process.stdout:
                    print(line, end="")
                    lf.write(line)
                    lf.flush()
                
                process.wait()
                exit_code = process.returncode
        except Exception as e:
            self.log(f"Training process error: {e}", "ERROR")
            exit_code = 1
        
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        if exit_code == 0:
            self.log(f"Training completed successfully in {hours}h {minutes}m!")
            self.log(f"  Log saved to: {log_file}")
            return True
        else:
            self.log(f"Training failed after {hours}h {minutes}m with exit code {exit_code}", "ERROR")
            return False
    
    # =========================================================================
    # Main Pipeline
    # =========================================================================
    def run_pipeline(self, dry_run: bool = False):
        """Run the complete pipeline for a single document
        
        Args:
            dry_run: If True, run steps 1-6 only (no GPU needed). Skip training.
        """
        self.log("=" * 60)
        self.log(f"Starting Automated Training Pipeline for {self.config.doc_id}")
        self.log(f"Run directory: {self.run_dir}")
        if dry_run:
            self.log("DRY RUN MODE - Will skip training (step 7)")
        self.log("=" * 60)
        
        try:
            # Step 1: Load document and get text from corpus
            doc_loaded = self.step1_load_document()
            if not doc_loaded:
                self.log("Document loading failed. Exiting.", "ERROR")
                return False
            
            # Step 2: Tokenize the document text
            token_count = self.step2_tokenize()
            if not token_count:
                self.log("Tokenization failed. Exiting.", "ERROR")
                return False
            
            # Step 3: Compress (or use raw text for p-init)
            if self.config.init_method == "pinit":
                context_path, cache_size = self.step3_pinit(token_count)
            else:
                context_path, cache_size = self.step3_compress(token_count)
            if not context_path:
                self.log("Context initialization failed. Exiting.", "ERROR")
                return False
            
            # Step 4: Train/val split using QA pairs from the doc
            train_json, eval_json = self.step4_train_val_split()
            if not train_json or not eval_json:
                self.log("Train/val split failed. Exiting.", "ERROR")
                return False
            
            # Step 5: Convert to PKL
            train_pkl, eval_pkl = self.step5_convert_data(train_json, eval_json)
            if not train_pkl or not eval_pkl:
                self.log("Data conversion failed. Exiting.", "ERROR")
                return False
            
            # Step 6: Generate config
            config_path = self.step6_generate_config(train_pkl, eval_pkl, context_path, cache_size)
            
            # Step 7: Run training (skip in dry-run mode)
            if dry_run:
                self.log("=" * 60)
                self.log("DRY RUN COMPLETE - Steps 1-6 successful!")
                self.log(f"Generated config: {config_path}")
                self.log(f"To run training manually:")
                self.log(f"  ./run_training.sh  # (update paths in script)")
                self.log(f"  OR: /home/eftychia/micromamba run -n financial-qa python {config_path}")
                self.log("=" * 60)
                return True
            
            success = self.step7_run_training(config_path)
            
            self.log("=" * 60)
            self.log(f"Pipeline completed. Success: {success}")
            self.log("=" * 60)
            
            return success
            
        except Exception as e:
            self.log(f"Pipeline failed with exception: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            return False


def run_multiple_doc_ids(doc_ids: List[str], dry_run: bool = False, **kwargs):
    """Run pipeline for multiple doc_ids sequentially
    
    After one training completes, automatically starts the next.
    """
    results = {}
    total_docs = len(doc_ids)
    
    print("\n" + "="*60)
    print(f"AUTOMATED MULTI-DOCUMENT PIPELINE")
    print(f"Documents: {len(doc_ids)} to process")
    if dry_run:
        print("DRY RUN MODE - Will skip training")
    print(f"Each run will complete before the next begins")
    print("="*60 + "\n")
    
    for i, doc_id in enumerate(doc_ids, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{total_docs}] Processing document: {doc_id}")
        print(f"{'='*60}\n")
        
        config = PipelineConfig(doc_id=doc_id, **kwargs)
        pipeline = AutomatedPipeline(config)
        
        success = pipeline.run_pipeline(dry_run=dry_run)
        results[doc_id] = success
        
        if not success:
            print(f"⚠️  Warning: Pipeline failed for {doc_id}")
            print("Continuing to next document...")
        else:
            print(f"✅ Pipeline completed successfully for {doc_id}")
        
        # Small delay between runs
        if i < total_docs:
            print(f"\nWaiting 10 seconds before starting next document...")
            time.sleep(10)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    successful = sum(1 for s in results.values() if s)
    failed = len(results) - successful
    
    for doc_id, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {doc_id}: {status}")
    
    print("-"*60)
    print(f"Total: {successful} succeeded, {failed} failed out of {len(results)}")
    print("="*60)
    
    return results


def get_available_doc_ids() -> List[str]:
    """Get list of available doc_ids from the documents folder"""
    docs_dir = BASE_DIR / "qa_generation_full_corpus" / "qa_output" / "documents"
    if not docs_dir.exists():
        return []
    
    doc_ids = []
    for f in sorted(docs_dir.glob("doc_*.json")):
        doc_id = f.stem  # e.g., "doc_000000"
        doc_ids.append(doc_id)
    
    return doc_ids


def main():
    parser = argparse.ArgumentParser(
        description="Automated Training Pipeline - Based on GENERATION_WORKFLOW.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single document with default settings (4x compression, 90% train, reasoning+synthesis)
  python automated_training_pipeline.py --doc-id doc_000000
  
  # Custom compression and train ratio
  python automated_training_pipeline.py --doc-id doc_000001 --compression-ratio 4 --train-ratio 0.9
  
  # Use teacher_forcing training mode (Cartridges default)
  python automated_training_pipeline.py --doc-id doc_000002 --training-mode teacher_forcing
  
  # Multiple documents (runs sequentially, starts next after previous completes)
  python automated_training_pipeline.py --doc-ids doc_000000 doc_000001 doc_000002
  
  # Range of documents
  python automated_training_pipeline.py --doc-range 0 100
  
  # Process all available documents
  python automated_training_pipeline.py --all-docs
  
  # List available documents
  python automated_training_pipeline.py --list-docs
        """
    )
    
    parser.add_argument("--doc-id", type=str,
                        help="Single document ID (e.g., doc_000000)")
    parser.add_argument("--doc-ids", nargs="+", default=None,
                        help="Multiple document IDs to process sequentially")
    parser.add_argument("--doc-range", nargs=2, type=int, metavar=("START", "END"),
                        help="Range of document numbers (e.g., --doc-range 0 100)")
    parser.add_argument("--all-docs", action="store_true",
                        help="Process all available documents")
    parser.add_argument("--list-docs", action="store_true",
                        help="List all available document IDs and exit")
    
    parser.add_argument("--compression-ratio", type=float, default=4.0,
                        help="Compression ratio for context (default: 4.0 = 4x compression)")
    parser.add_argument("--train-ratio", type=float, default=0.9,
                        help="Train/val split ratio (default: 0.9 = 90%% train, 10%% eval)")
    parser.add_argument("--train-templates", nargs="+", default=["reasoning", "synthesis"],
                        help="Templates to include in train set (default: reasoning synthesis)")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-4b",
                        help="Tokenizer model (default: Qwen/Qwen3-4b)")
    parser.add_argument("--training-mode", type=str, default="standard_ce",
                        choices=["standard_ce", "teacher_forcing", "extended_ntl"],
                        help="Training mode (default: standard_ce)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=2e-2,
                        help="Learning rate (default: 2e-2)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Global batch size (default: 32)")
    parser.add_argument("--packed-seq-length", type=int, default=2048,
                        help="Packed sequence length (default: 2048)")
    parser.add_argument("--freeze-keys", action="store_true",
                        help="Freeze key vectors during training (only train values)")
    parser.add_argument("--init-method", type=str, default="numcache",
                        choices=["numcache", "pinit"],
                        help="Cache initialization method: numcache (compressed text) or pinit (first-p raw tokens)")
    parser.add_argument("--qa-file", type=str, default=None,
                        help="External QA file (JSON list with doc_id field) instead of per-doc qa_output")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output base directory for training runs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run steps 1-6 only (no GPU needed). Skip training.")
    
    args = parser.parse_args()
    
    # Handle --list-docs
    if args.list_docs:
        doc_ids = get_available_doc_ids()
        print(f"Available documents: {len(doc_ids)}")
        for doc_id in doc_ids[:20]:
            print(f"  {doc_id}")
        if len(doc_ids) > 20:
            print(f"  ... and {len(doc_ids) - 20} more")
        return
    
    # Determine which doc_ids to process
    doc_ids_to_process = []
    
    if args.doc_id:
        doc_ids_to_process = [args.doc_id]
    elif args.doc_ids:
        doc_ids_to_process = args.doc_ids
    elif args.doc_range:
        start, end = args.doc_range
        doc_ids_to_process = [f"doc_{i:06d}" for i in range(start, end)]
    elif args.all_docs:
        doc_ids_to_process = get_available_doc_ids()
    else:
        parser.error("Specify --doc-id, --doc-ids, --doc-range, or --all-docs")
    
    if not doc_ids_to_process:
        print("No documents to process")
        return
    
    # Common kwargs for config
    common_kwargs = dict(
        compression_ratio=args.compression_ratio,
        train_ratio=args.train_ratio,
        train_templates=args.train_templates,
        tokenizer_name=args.tokenizer,
        training_mode=args.training_mode,
        epochs=args.epochs,
        lr=args.lr,
        global_batch_size=args.batch_size,
        packed_seq_length=args.packed_seq_length,
        freeze_keys=args.freeze_keys,
        init_method=args.init_method,
        qa_file=args.qa_file,
        output_dir=args.output_dir
    )
    
    if len(doc_ids_to_process) == 1:
        # Single document run
        config = PipelineConfig(
            doc_id=doc_ids_to_process[0],
            **common_kwargs
        )
        
        pipeline = AutomatedPipeline(config)
        pipeline.run_pipeline(dry_run=args.dry_run)
    else:
        # Multiple documents - run sequentially
        run_multiple_doc_ids(doc_ids_to_process, dry_run=args.dry_run, **common_kwargs)


if __name__ == "__main__":
    main()
