#!/usr/bin/env python3
"""
Train a contrastive question-to-chunk retriever using title-derived KV caches.

Example (uses all defaults):
    python train_title_cache_contrastive.py \
        --qa-json original_data/chunk_based_qa.json \
        --corpus-jsonl original_data/enhanced_corpus_new.jsonl

Workflow:
1. Load every chunk from original_data/enhanced_corpus_new.jsonl, convert its
   title or text into a KV cache (keys/values) with Qwen3, and pool KV features
   into vectors. If a pooled vector checkpoint already exists, reuse it directly.
   When >=2 GPUs are visible, the Qwen backbone is sharded across them by default.
2. Load original_data/chunk_based_qa.json to obtain (question, doc_id) pairs.
3. Train lightweight projection heads so that question vectors align with the
   pooled KV vectors of their gold doc_id (InfoNCE with in-batch negatives).
4. Evaluate each question by retrieving the best-matching chunk across the entire
   corpus and write predictions/accuracy to JSON. Persist the KV vectors for reuse.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


DEFAULT_RAG_ROOT = Path(__file__).resolve().parent / "Financial-QA-Benchmark-with-KV-cache-rag"
DEFAULT_LEGACY_ROOT = Path(__file__).resolve().parent / "Financial-QA-Benchmark-with-KV-cache"
if "CARTRIDGES_DIR" in os.environ:
    PROJECT_ROOT = Path(os.environ["CARTRIDGES_DIR"]).resolve()
elif DEFAULT_RAG_ROOT.exists():
    PROJECT_ROOT = DEFAULT_RAG_ROOT
else:
    PROJECT_ROOT = DEFAULT_LEGACY_ROOT
sys.path.append(str(PROJECT_ROOT))
os.environ.setdefault("CARTRIDGES_DIR", str(PROJECT_ROOT))
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))

from cartridges.cache import AttnConfig, CARTRIDGE_SEQ_ID, TrainableCache  # noqa: E402
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM  # noqa: E402


DEFAULT_QA_JSON = Path("original_data/generated_questions_from_chunks_10k.json")
DEFAULT_CORPUS_JSONL = Path("original_data/enhanced_corpus_new_filtered_by_generated_questions_10k.jsonl")
# DEFAULT_QA_JSON = Path("original_data/chunk_based_qa.json")
# DEFAULT_CORPUS_JSONL = Path("original_data/enhanced_corpus_new.jsonl")
# DEFAULT_OUTPUT_JSON = {
#     "title": Path("title_contrastive_results.json"),
#     "text": Path("text_contrastive_results.json"),
# }
DEFAULT_OUTPUT_JSON = {
    "title": Path("title_contrastive_results_10k.json"),
    "text": Path("text_contrastive_results_10k_dyn.json"),
}
DEFAULT_POOLED_PT = {
    "title": Path("title_pooled_kv_4k_dyn.pt"),
    "text": Path("text_pooled_kv_4k_dyn.pt"),
}
# DEFAULT_POOLED_PT = {
#     "title": Path("title_pooled_kv.pt"),
#     "text": Path("text_pooled_kv.pt"),
# }
CHUNK_FIELD_OPTIONS = ("title", "text")
DEFAULT_CHUNK_FIELD = "text"


def pool_kv_features(
    kv_keys: List[torch.Tensor],
    last_n_layers: int,
) -> torch.Tensor:
    """
    Pool the last N layers of KV keys into a flattened CPU vector.
    Strategy: take specified layers, mean over tokens -> (heads, dim),
    average layers, flatten, cast to float32.
    """
    total_layers = len(kv_keys)
    if last_n_layers <= 0 or last_n_layers > total_layers:
        selected = kv_keys
    else:
        selected = kv_keys[-last_n_layers:]

    pooled_layers = []
    for layer_k in selected:
        layer_mean = layer_k.mean(dim=2).squeeze(0)  # (heads, head_dim)
        pooled_layers.append(layer_mean)
    stacked = torch.stack(pooled_layers, dim=0)  # (layers, heads, head_dim)
    avg = stacked.mean(dim=0)  # (heads, head_dim)
    return avg.flatten().to(torch.float32)


@dataclass
class TitleChunkEntry:
    doc_id: str
    chunk_text: str
    token_count: int
    pooled_kv: torch.Tensor  # flattened KV feature vector (CPU)
    layer_vectors: List[torch.Tensor]  # flattened per-layer vectors (CPU)
    kv_keys: List[torch.Tensor]
    kv_values: List[torch.Tensor]
    cache_path: Optional[str] = None


class QuestionChunkDataset(Dataset):
    def __init__(
        self,
        qa_pairs: List[Dict],
        chunk_index: Dict[str, int],
        tokenizer,
        max_question_tokens: Optional[int],
        question_cache: Optional[Dict[str, torch.Tensor]] = None,
        cache_key_fn=None,
    ):
        self.samples: List[Tuple[torch.LongTensor, int]] = []
        self.matched_pairs: List[Dict] = []
        use_cache = question_cache is not None
        for qa in qa_pairs:
            doc_id = qa.get("doc_id") or qa.get("chunk_id")
            question = qa.get("question", "").strip()
            if not doc_id or not question:
                continue
            chunk_idx = chunk_index.get(doc_id)
            if chunk_idx is None:
                continue
            if use_cache:
                if cache_key_fn is None:
                    raise RuntimeError("cache_key_fn must be provided when using question_cache.")
                key = cache_key_fn(qa)
                embedding = question_cache.get(key) if question_cache else None
                if embedding is None:
                    raise RuntimeError(f"Missing cached embedding for key: {key}")
                self.samples.append((embedding, chunk_idx))
            else:
                tokens = tokenize_text(tokenizer, question, max_question_tokens)
                self.samples.append((tokens, chunk_idx))
            self.matched_pairs.append(qa)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fn(batch):
    questions, chunk_idx = zip(*batch)
    idx_tensor = torch.tensor(chunk_idx, dtype=torch.long)
    return list(questions), idx_tensor


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        if num_layers < 3 or num_layers > 6:
            raise ValueError("head_layers must be in [3, 6].")
        layers: List[nn.Module] = []
        # First layer maps input to proj dim.
        layers.extend(
            [
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.SiLU(),
                nn.Dropout(p=0.1),
            ]
        )
        # Hidden layers (num_layers - 2)
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    nn.Linear(out_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.SiLU(),
                    nn.Dropout(p=0.1),
                ]
            )
        # Final layer without activation/dropout.
        layers.extend(
            [
                nn.Linear(out_dim, out_dim),
                nn.LayerNorm(out_dim),
            ]
        )
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class PoolMLP(nn.Module):
    """
    Predicts layer pooling weights from question representations.
    """

    def __init__(self, in_dim: int, num_layers: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_layers),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.mlp(x), dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa-json",
        type=Path,
        default=DEFAULT_QA_JSON,
        help=f"QA dataset with question/doc_id pairs (default: {DEFAULT_QA_JSON})",
    )
    parser.add_argument(
        "--corpus-jsonl",
        type=Path,
        default=DEFAULT_CORPUS_JSONL,
        help=f"Full corpus JSONL containing chunk titles/text (default: {DEFAULT_CORPUS_JSONL})",
    )
    parser.add_argument(
        "--chunk-field",
        type=str,
        choices=CHUNK_FIELD_OPTIONS,
        default=DEFAULT_CHUNK_FIELD,
        help=f"Which field to use for building chunk KV caches (default: {DEFAULT_CHUNK_FIELD}).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to dump evaluation predictions (default depends on chunk-field).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional directory to store raw per-chunk KV cache tensors.",
    )
    parser.add_argument(
        "--pooled-kv-pt",
        type=Path,
        default=None,
        help="Checkpoint for pooled KV vectors (default depends on chunk-field). "
        "If it exists, pooled vectors will be loaded instead of recomputed.",
    )
    parser.add_argument(
        "--save-heads",
        type=Path,
        default=None,
        help="Optional path to save projection heads after training.",
    )
    parser.add_argument(
        "--load-heads",
        type=Path,
        default=None,
        help="Optional path to load projection heads before evaluation.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only evaluate using --load-heads.",
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-4b", help="Backbone HF repo (default: Qwen/Qwen3-4b).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on (default: auto).")
    parser.add_argument("--max-chunk-tokens", type=int, default=4096, help="Truncation length for chunk field text (default: 256).")
    parser.add_argument("--max-question-tokens", type=int, default=256, help="Truncation length for questions (default: 256).")
    parser.add_argument(
        "--question-emb-cache",
        type=Path,
        default=None,
        help="Optional path to cache question embeddings for reuse across runs.",
    )
    parser.add_argument(
        "--question-emb-cache-only",
        action="store_true",
        help="Require question embedding cache; do not rebuild if missing.",
    )
    parser.add_argument(
        "--question-emb-cache-dtype",
        type=str,
        choices=("fp16", "fp32"),
        default="fp16",
        help="Storage dtype for cached question embeddings (default: fp16).",
    )
    parser.add_argument("--limit-chunks", type=int, default=None, help="Process only first N chunks (debug).")
    parser.add_argument("--limit-qa", type=int, default=None, help="Use only first N QA pairs (debug).")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of QA data reserved for validation split (default: 0.1).")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Fraction of QA data reserved for test split (default: 0.1).")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for train/test split (default: 42).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for contrastive training (default: 64).")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for projection heads (default: 1e-4).")
    parser.add_argument("--proj-dim", type=int, default=1024, help="Projection head hidden size (default: 512).")
    parser.add_argument("--head-layers", type=int, default=5, help="Projection head depth in [3, 6] (default: 5).")
    parser.add_argument("--temperature", type=float, default=0.05, help="InfoNCE temperature (default: 0.05).")
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Logging frequency (steps). Set 0 to disable step logs.",
    )
    parser.add_argument(
        "--log-every-epochs",
        type=int,
        default=5,
        help="Logging frequency (epochs) for training summaries (default: 5).",
    )
    parser.add_argument(
        "--early-stop-train-patience",
        type=int,
        default=0,
        help=(
            "Early-stop patience based on train avg_loss when no validation set is used. "
            "Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--early-stop-train-min-delta",
        type=float,
        default=0.0,
        help="Minimum avg_loss improvement required to reset early-stop patience.",
    )
    parser.add_argument(
        "--early-stop-train-min-epochs",
        type=int,
        default=1,
        help="Minimum epochs before train-loss early stopping can trigger.",
    )
    parser.add_argument("--kv-pool-layers", type=int, default=4, help="Number of final KV layers to pool (default: 4).")
    parser.add_argument(
        "--pooled-flush-interval",
        type=int,
        default=500,
        help="Number of chunks to buffer before flushing pooled KV vectors to disk (default: 500, set 0 to disable).",
    )
    parser.add_argument(
        "--disable-multi-gpu",
        action="store_true",
        help="Force single-device execution even if >=2 GPUs are visible.",
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=4,
        help="Maximum number of GPUs to leverage when multi-GPU mode is enabled (default: 2).",
    )
    args = parser.parse_args()
    if args.output_json is None:
        args.output_json = DEFAULT_OUTPUT_JSON[args.chunk_field]
    if args.pooled_kv_pt is None:
        args.pooled_kv_pt = DEFAULT_POOLED_PT[args.chunk_field]
    if args.eval_only and args.load_heads is None:
        parser.error("--eval-only requires --load-heads")
    return args


def save_heads_checkpoint(
    path: Path,
    question_head: ProjectionHead,
    chunk_head: ProjectionHead,
    pool_mlp: PoolMLP,
    output: Dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "question_head": question_head.state_dict(),
        "chunk_head": chunk_head.state_dict(),
        "pool_mlp": pool_mlp.state_dict(),
        "meta": {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": output.get("model_name", ""),
            "proj_dim": output.get("proj_dim", ""),
            "head_layers": output.get("head_layers", ""),
            "kv_pool_layers": output.get("kv_pool_layers", ""),
        },
    }
    torch.save(payload, path)


def load_heads_checkpoint(
    path: Path,
    question_head: ProjectionHead,
    chunk_head: ProjectionHead,
    pool_mlp: PoolMLP,
) -> Dict:
    payload = torch.load(path, map_location="cpu")
    question_head.load_state_dict(payload["question_head"])
    chunk_head.load_state_dict(payload["chunk_head"])
    pool_mlp.load_state_dict(payload["pool_mlp"])
    return payload.get("meta") or {}


def load_model_and_tokenizer(model_name: str, device: torch.device, use_multi_gpu: bool, max_gpus: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FlexQwen3ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    primary_device = device

    if use_multi_gpu and torch.cuda.device_count() > 1:
        num_gpus = min(torch.cuda.device_count(), max_gpus)
        device_ids = list(range(num_gpus))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        print(f"[Multi-GPU] Using DataParallel on GPUs: {device_ids}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer, primary_device


def build_attn_config(model: FlexQwen3ForCausalLM) -> AttnConfig:
    head_dim = (
        model.config.head_dim
        if hasattr(model.config, "head_dim")
        else model.config.hidden_size // model.config.num_attention_heads
    )
    return AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=head_dim,
    )


def tokenize_text(tokenizer, text: str, max_tokens: Optional[int]) -> torch.LongTensor:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=max_tokens is not None,
        max_length=max_tokens,
    )
    return encoded.input_ids.squeeze(0)


def build_question_key(qa: Dict) -> str:
    qid = qa.get("qid")
    if qid:
        return f"qid:{qid}"
    question = (qa.get("question") or "").strip()
    return f"q:{question}"


def extract_gold_doc_ids(qa: Dict) -> List[str]:
    doc_id = qa.get("doc_id") or qa.get("chunk_id")
    if doc_id:
        return [doc_id]
    doc_ids = qa.get("doc_ids")
    if isinstance(doc_ids, list):
        return [d for d in doc_ids if d]
    return []


def filter_qa_pairs_for_eval(qa_pairs: List[Dict], doc_index: Dict[str, int]) -> List[Dict]:
    matched: List[Dict] = []
    for qa in qa_pairs:
        gold_ids = extract_gold_doc_ids(qa)
        if not gold_ids:
            continue
        if any(doc_id in doc_index for doc_id in gold_ids):
            matched.append(qa)
    return matched


def load_question_emb_cache(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict]:
    payload = torch.load(path, map_location="cpu")
    embeddings = payload.get("embeddings") or {}
    meta = payload.get("meta") or {}
    return embeddings, meta


def save_question_emb_cache(path: Path, embeddings: Dict[str, torch.Tensor], meta: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"embeddings": embeddings, "meta": meta}
    torch.save(payload, path)


def compute_question_embeddings(
    qa_pairs: List[Dict],
    tokenizer,
    model,
    device: torch.device,
    max_question_tokens: Optional[int],
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    key_to_question: Dict[str, str] = {}
    for qa in qa_pairs:
        question = (qa.get("question") or "").strip()
        if not question:
            continue
        key = build_question_key(qa)
        if key not in key_to_question:
            key_to_question[key] = question

    keys = list(key_to_question.keys())
    embeddings: Dict[str, torch.Tensor] = {}
    for idx in range(0, len(keys), batch_size):
        batch_keys = keys[idx : idx + batch_size]
        token_batch = [
            tokenize_text(tokenizer, key_to_question[k], max_question_tokens)
            for k in batch_keys
        ]
        batch_emb = embed_questions(token_batch, model, device)
        for key, emb in zip(batch_keys, batch_emb):
            embeddings[key] = emb.detach().cpu()
    return embeddings


def collect_missing_questions(
    qa_pairs: List[Dict],
    question_cache: Dict[str, torch.Tensor],
) -> List[Dict]:
    missing: List[Dict] = []
    for qa in qa_pairs:
        question = (qa.get("question") or "").strip()
        if not question:
            continue
        key = build_question_key(qa)
        if key not in question_cache:
            missing.append(qa)
    return missing


def batch_questions_to_embeddings(
    batch_questions: List[torch.Tensor],
    model,
    device: torch.device,
) -> torch.Tensor:
    if not batch_questions:
        return torch.empty((0,))
    first = batch_questions[0]
    if isinstance(first, torch.Tensor) and first.dtype != torch.long:
        return torch.stack([q.to(device).float() for q in batch_questions], dim=0)
    return embed_questions(batch_questions, model, device).to(device).float()


def encode_chunk_to_cache(
    doc_id: str,
    content: str,
    tokenizer,
    model,
    attn_config: AttnConfig,
    max_tokens: Optional[int],
    kv_pool_layers: int,
    device: torch.device,
) -> TitleChunkEntry:
    input_ids = tokenize_text(tokenizer, content, max_tokens=max_tokens).to(device)
    seq_len = input_ids.shape[-1]
    seq_ids = torch.full((seq_len,), CARTRIDGE_SEQ_ID, dtype=torch.long, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    cache = TrainableCache(config=attn_config).to(device)

    with torch.no_grad():
        model(
            input_ids=input_ids,
            seq_ids=seq_ids,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=cache,
            mode="generate",
        )
    kv_keys = [layer_k.detach().to("cpu") for layer_k in cache._keys]
    kv_values = [layer_v.detach().to("cpu") for layer_v in cache._values]
    pooled_kv = pool_kv_features(kv_keys, last_n_layers=kv_pool_layers)
    # Precompute per-layer flattened vectors for dynamic pooling (last kv_pool_layers).
    per_layer_vecs: List[torch.Tensor] = []
    total_layers = len(kv_keys)
    selected = kv_keys if kv_pool_layers <= 0 or kv_pool_layers > total_layers else kv_keys[-kv_pool_layers:]
    for layer_k in selected:
        layer_mean = layer_k.mean(dim=2).squeeze(0)  # (heads, head_dim)
        per_layer_vecs.append(layer_mean.flatten().to(torch.float32))

    return TitleChunkEntry(
        doc_id=doc_id,
        chunk_text=content,
        token_count=seq_len,
        pooled_kv=pooled_kv,
        layer_vectors=per_layer_vecs,
        kv_keys=kv_keys,
        kv_values=kv_values,
    )


def maybe_save_cache(entry: TitleChunkEntry, cache_dir: Optional[Path], chunk_field: str) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{entry.doc_id}_{chunk_field}_cache.pt"
    torch.save(
        {
            "doc_id": entry.doc_id,
            "chunk_field": chunk_field,
            "text": entry.chunk_text,
            "token_count": entry.token_count,
            "keys": entry.kv_keys,
            "values": entry.kv_values,
            "pooled_kv": entry.pooled_kv,
            "layer_vectors": entry.layer_vectors,
        },
        out_path,
    )
    entry.cache_path = str(out_path.resolve())


def append_pooled_vectors(path: Path, doc_ids: List[str], vectors: List[torch.Tensor]) -> None:
    if not doc_ids:
        return
    stacked = torch.stack([vec.to(torch.float32) for vec in vectors])
    if path.exists():
        data = torch.load(path, map_location="cpu")
        data["doc_ids"].extend(doc_ids)
        data["pooled_kv"] = torch.cat([data["pooled_kv"], stacked], dim=0)
    else:
        data = {"doc_ids": list(doc_ids), "pooled_kv": stacked}
    torch.save(data, path)


def load_chunk_entries_from_pt(pt_path: Path) -> List[TitleChunkEntry]:
    data = torch.load(pt_path, map_location="cpu")
    doc_ids = data["doc_ids"]
    pooled = data["pooled_kv"]
    layer_vectors_raw = data.get("layer_vectors")
    layer_vectors_per_chunk: List[List[torch.Tensor]] = []
    if layer_vectors_raw is not None:
        if isinstance(layer_vectors_raw, torch.Tensor) and layer_vectors_raw.ndim == 3:
            if layer_vectors_raw.shape[0] == len(doc_ids):
                for idx in range(len(doc_ids)):
                    layer_vectors_per_chunk.append(
                        [
                            layer_vectors_raw[idx, j].clone().to(torch.float32)
                            for j in range(layer_vectors_raw.shape[1])
                        ]
                    )
            elif layer_vectors_raw.shape[1] == len(doc_ids):
                for idx in range(len(doc_ids)):
                    layer_vectors_per_chunk.append(
                        [
                            layer_vectors_raw[j, idx].clone().to(torch.float32)
                            for j in range(layer_vectors_raw.shape[0])
                        ]
                    )
        elif isinstance(layer_vectors_raw, list) and len(layer_vectors_raw) == len(doc_ids):
            for item in layer_vectors_raw:
                if isinstance(item, list):
                    layer_vectors_per_chunk.append(
                        [vec.clone().to(torch.float32) for vec in item]
                    )
                else:
                    layer_vectors_per_chunk.append([])
    entries: List[TitleChunkEntry] = []
    for idx, (doc_id, vec) in enumerate(zip(doc_ids, pooled)):
        entries.append(
            TitleChunkEntry(
                doc_id=doc_id,
                chunk_text="",
                token_count=0,
                pooled_kv=vec.clone().to(torch.float32),
                layer_vectors=layer_vectors_per_chunk[idx]
                if idx < len(layer_vectors_per_chunk)
                else [],
                kv_keys=[],
                kv_values=[],
            )
        )
    print(f"Loaded {len(entries)} pooled KV vectors from {pt_path}.")
    return entries


def build_chunk_layer_matrix(
    entries: List[TitleChunkEntry],
    pool_layers: int,
) -> torch.Tensor:
    if pool_layers <= 0:
        raise ValueError("pool_layers must be > 0 for dynamic pooling.")
    num_chunks = len(entries)
    if num_chunks == 0:
        raise ValueError("No chunk entries available.")

    def get_vec(entry: TitleChunkEntry, layer_idx: int) -> torch.Tensor:
        if entry.layer_vectors and layer_idx < len(entry.layer_vectors):
            return entry.layer_vectors[layer_idx]
        return entry.pooled_kv

    dim = get_vec(entries[0], 0).numel()
    layers: List[torch.Tensor] = []
    for l in range(pool_layers):
        layer_vecs = []
        for entry in entries:
            vec = get_vec(entry, l)
            if vec.numel() != dim:
                raise RuntimeError("Inconsistent vector dims across chunks.")
            layer_vecs.append(vec.to(torch.float32))
        layers.append(torch.stack(layer_vecs, dim=0))
    return torch.stack(layers, dim=0)  # (pool_layers, num_chunks, dim)


def prepare_chunk_entries(
    args: argparse.Namespace,
    tokenizer,
    model,
    attn_config: AttnConfig,
    device: torch.device,
) -> Tuple[List[TitleChunkEntry], Dict[str, int]]:
    pooled_path = args.pooled_kv_pt
    if pooled_path and pooled_path.exists():
        entries = load_chunk_entries_from_pt(pooled_path)
        doc_index = {entry.doc_id: idx for idx, entry in enumerate(entries)}
        return entries, doc_index

    if pooled_path is None:
        raise RuntimeError("pooled_kv_pt must be provided when rebuilding pooled vectors.")

    if pooled_path.exists():
        pooled_path.unlink()
    pooled_path.parent.mkdir(parents=True, exist_ok=True)

    corpus_entries = load_corpus(args.corpus_jsonl, args.limit_chunks)
    print(f"Loaded {len(corpus_entries)} corpus entries.")

    flush_interval = max(0, args.pooled_flush_interval)
    buffered_ids: List[str] = []
    buffered_vecs: List[torch.Tensor] = []

    for idx, entry in enumerate(corpus_entries):
        doc_id = entry.get("_id")
        if not doc_id:
            continue
        raw_text = entry.get(args.chunk_field, "")
        if args.chunk_field == "title" and not raw_text:
            raw_text = entry.get("text", "")
        if not raw_text:
            continue
        chunk_text = raw_text.strip()
        if not chunk_text:
            continue
        chunk_entry = encode_chunk_to_cache(
            doc_id=doc_id,
            content=chunk_text,
            tokenizer=tokenizer,
            model=model,
            attn_config=attn_config,
            max_tokens=args.max_chunk_tokens,
            kv_pool_layers=args.kv_pool_layers,
            device=device,
        )
        maybe_save_cache(chunk_entry, args.cache_dir, args.chunk_field)
        buffered_ids.append(doc_id)
        buffered_vecs.append(chunk_entry.pooled_kv.detach().to("cpu"))
        if flush_interval > 0 and len(buffered_ids) >= flush_interval:
            append_pooled_vectors(pooled_path, buffered_ids, buffered_vecs)
            buffered_ids.clear()
            buffered_vecs.clear()
        if args.log_every and (idx + 1) % args.log_every == 0:
            print(f"[Cache-{args.chunk_field}] processed {idx+1} corpus entries.")

    if buffered_ids:
        append_pooled_vectors(pooled_path, buffered_ids, buffered_vecs)

    if not pooled_path.exists():
        raise RuntimeError("Failed to create pooled KV checkpoint.")

    entries = load_chunk_entries_from_pt(pooled_path)
    doc_index = {entry.doc_id: idx for idx, entry in enumerate(entries)}
    return entries, doc_index


def load_corpus(corpus_path: Path, limit: Optional[int]) -> List[Dict]:
    entries: List[Dict] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            entries.append(data)
            if limit is not None and len(entries) >= limit:
                break
    return entries


def load_qa_pairs(path: Path, limit: Optional[int]) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        data = data[:limit]
    return data


def embed_questions(question_ids: List[torch.LongTensor], model, device: torch.device) -> torch.Tensor:
    reps = []
    for ids in question_ids:
        ids = ids.to(device)
        with torch.no_grad():
            embeds = model.model.embed_tokens(ids.unsqueeze(0))
            reps.append(embeds.mean(dim=1).squeeze(0))
    return torch.stack([rep.float() for rep in reps], dim=0)


def train_contrastive(
    dataset: QuestionChunkDataset,
    chunk_entries: List[TitleChunkEntry],
    base_model,
    tokenizer,
    device: torch.device,
    val_pairs: List[Dict],
    args: argparse.Namespace,
    question_cache: Optional[Dict[str, torch.Tensor]] = None,
    cache_key_fn=None,
) -> Tuple[ProjectionHead, ProjectionHead, PoolMLP, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    chunk_layer_matrix = build_chunk_layer_matrix(chunk_entries, args.kv_pool_layers).to(device)
    _, num_chunks, chunk_dim = chunk_layer_matrix.shape
    question_head = ProjectionHead(base_model.config.hidden_size, args.proj_dim, args.head_layers).to(device)
    chunk_head = ProjectionHead(chunk_dim, args.proj_dim, args.head_layers).to(device)
    pool_mlp = PoolMLP(base_model.config.hidden_size, chunk_layer_matrix.shape[0]).to(device)
    if args.load_heads is not None:
        load_heads_checkpoint(args.load_heads, question_head, chunk_head, pool_mlp)
        print(f"[Train] Loaded initial heads from {args.load_heads}")
    opt = torch.optim.Adam(
        list(question_head.parameters()) + list(chunk_head.parameters()) + list(pool_mlp.parameters()),
        lr=args.lr,
    )

    global_step = 0
    best_val = -1.0
    best_q_state, best_c_state, best_p_state = None, None, None
    best_train_loss = float("inf")
    best_train_q_state, best_train_c_state, best_train_p_state = None, None, None
    no_improve_epochs = 0

    epoch_log_every = args.log_every_epochs if args.log_every_epochs is not None else 0
    for epoch in range(args.epochs):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for batch_questions, chunk_idx in loader:
            if len(batch_questions) == 0:
                continue

            opt.zero_grad()
            q_emb = batch_questions_to_embeddings(batch_questions, base_model, device)
            q_proj = question_head(q_emb)

            # Dynamic pooling per question.
            pool_w = pool_mlp(q_emb)  # (B, layers)
            chunk_vec = torch.einsum("bl,lnd->bnd", pool_w, chunk_layer_matrix)
            chunk_proj = chunk_head(chunk_vec.view(-1, chunk_dim)).view(len(batch_questions), num_chunks, -1)
            logits = torch.einsum("bd,bnd->bn", q_proj, chunk_proj) / args.temperature
            labels = chunk_idx.to(device)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            epoch_loss_sum += float(loss.item())
            epoch_steps += 1

            global_step += 1
            if args.log_every and global_step % args.log_every == 0:
                print(f"[Train] epoch={epoch+1} step={global_step} loss={loss.item():.4f}")

        if epoch_steps > 0 and epoch_log_every and (epoch + 1) % epoch_log_every == 0:
            avg_loss = epoch_loss_sum / epoch_steps
            print(f"[Train] epoch={epoch+1} avg_loss={avg_loss:.4f} steps={epoch_steps}")
        elif epoch_steps > 0:
            avg_loss = epoch_loss_sum / epoch_steps
        else:
            avg_loss = None

        # validate each epoch
        if val_pairs:
            _, val_acc, _ = evaluate(
                val_pairs,
                chunk_entries,
                question_head,
                chunk_head,
                pool_mlp,
                chunk_layer_matrix,
                tokenizer,
                base_model,
                device,
                args,
                question_cache=question_cache,
                cache_key_fn=cache_key_fn,
            )
            if val_acc > best_val:
                best_val = val_acc
                best_q_state = copy.deepcopy(question_head.state_dict())
                best_c_state = copy.deepcopy(chunk_head.state_dict())
                best_p_state = copy.deepcopy(pool_mlp.state_dict())
            print(f"[Val] epoch={epoch+1} acc={val_acc:.4f} best={best_val:.4f}")
        elif (
            args.early_stop_train_patience > 0
            and avg_loss is not None
        ):
            improved = avg_loss < (best_train_loss - args.early_stop_train_min_delta)
            if improved:
                best_train_loss = avg_loss
                no_improve_epochs = 0
                best_train_q_state = copy.deepcopy(question_head.state_dict())
                best_train_c_state = copy.deepcopy(chunk_head.state_dict())
                best_train_p_state = copy.deepcopy(pool_mlp.state_dict())
            elif (epoch + 1) >= args.early_stop_train_min_epochs:
                no_improve_epochs += 1
                print(
                    f"[EarlyStop-Train] epoch={epoch+1} avg_loss={avg_loss:.4f} "
                    f"best={best_train_loss:.4f} no_improve={no_improve_epochs}/"
                    f"{args.early_stop_train_patience}"
                )
                if no_improve_epochs >= args.early_stop_train_patience:
                    print(
                        f"[EarlyStop-Train] Triggered at epoch={epoch+1}. "
                        f"Restoring best train-loss checkpoint (avg_loss={best_train_loss:.4f})."
                    )
                    break

    # restore best on validation if available
    if best_q_state is not None and best_c_state is not None and best_p_state is not None:
        question_head.load_state_dict(best_q_state)
        chunk_head.load_state_dict(best_c_state)
        pool_mlp.load_state_dict(best_p_state)
    elif (
        best_train_q_state is not None
        and best_train_c_state is not None
        and best_train_p_state is not None
    ):
        question_head.load_state_dict(best_train_q_state)
        chunk_head.load_state_dict(best_train_c_state)
        pool_mlp.load_state_dict(best_train_p_state)

    return question_head, chunk_head, pool_mlp, chunk_layer_matrix


def evaluate(
    qa_pairs: List[Dict],
    chunk_entries: List[TitleChunkEntry],
    question_head: ProjectionHead,
    chunk_head: ProjectionHead,
    pool_mlp: PoolMLP,
    chunk_layer_matrix: torch.Tensor,
    tokenizer,
    base_model,
    device: torch.device,
    args: argparse.Namespace,
    question_cache: Optional[Dict[str, torch.Tensor]] = None,
    cache_key_fn=None,
) -> Tuple[List[Dict], float, Dict[str, float]]:
    if not qa_pairs:
        return [], 0.0, {}
    chunk_layer_matrix = chunk_layer_matrix.to(device)
    predictions: List[Dict] = []
    correct = 0
    processed = 0
    skipped = 0
    doc_ids = [entry.doc_id for entry in chunk_entries]
    doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    k_list = [1, 3, 5, 10]
    hit_counts = {k: 0 for k in k_list}
    recall_sums = {k: 0.0 for k in k_list}
    precision_sums = {k: 0.0 for k in k_list}
    mrr_sum = 0.0

    for idx, qa in enumerate(qa_pairs):
        gold_ids = extract_gold_doc_ids(qa)
        question = qa.get("question", "")
        if not gold_ids:
            skipped += 1
            continue
        gold_ids = [doc_id for doc_id in gold_ids if doc_id in doc_to_idx]
        if not gold_ids:
            skipped += 1
            continue
        gold_set = set(gold_ids)
        gold_idx_set = {doc_to_idx[doc_id] for doc_id in gold_ids}
        if question_cache is not None:
            if cache_key_fn is None:
                raise RuntimeError("cache_key_fn must be provided when using question_cache.")
            key = cache_key_fn(qa)
            emb = question_cache.get(key)
            if emb is None:
                raise RuntimeError(f"Missing cached embedding for key: {key}")
            q_vec = emb.unsqueeze(0).to(device).float()
        else:
            tokens = tokenize_text(tokenizer, question, args.max_question_tokens)
            q_vec = embed_questions([tokens], base_model, device).to(device).float()
        processed += 1
        with torch.no_grad():
            pool_w = pool_mlp(q_vec)  # (1, layers)
            chunk_vec = torch.einsum("bl,lnd->bnd", pool_w, chunk_layer_matrix)
            chunk_proj = chunk_head(chunk_vec.view(-1, chunk_layer_matrix.shape[2])).view(1, chunk_layer_matrix.shape[1], -1)
            q_proj = question_head(q_vec)
            scores = torch.einsum("bd,bnd->bn", q_proj, chunk_proj).squeeze(0) / args.temperature
        top_idx = int(torch.argmax(scores).item())
        pred_doc = doc_ids[top_idx]
        is_correct = pred_doc in gold_set
        correct += int(is_correct)

        for k in k_list:
            topk_idx = torch.topk(scores, k=k).indices.tolist()
            topk_docs = {doc_ids[i] for i in topk_idx}
            hits = len(topk_docs & gold_set)
            if hits > 0:
                hit_counts[k] += 1
            recall_sums[k] += hits / max(1, len(gold_set))
            precision_sums[k] += hits / k
        sorted_idx = torch.argsort(scores, descending=True).tolist()
        for rank, idx_val in enumerate(sorted_idx, start=1):
            if idx_val in gold_idx_set:
                mrr_sum += 1.0 / rank
                break

        predictions.append(
            {
                "qid": qa.get("qid") or qa.get("q_id"),
                "question": question,
                "gold_doc_id": gold_ids[0] if len(gold_ids) == 1 else None,
                "gold_doc_ids": gold_ids,
                "pred_doc_id": pred_doc,
                "score": float(scores[top_idx].item()),
                "correct": is_correct,
            }
        )

        if args.log_every and processed % args.log_every == 0:
            running_acc = correct / processed
            print(f"[Eval] processed {processed} samples, acc={running_acc:.4f}")

    accuracy = correct / max(1, processed)
    print(f"[Eval] Final accuracy: {accuracy:.4f} ({correct}/{processed})")
    if skipped:
        print(f"[Eval] Skipped {skipped} samples without matching doc_ids.")
    total = max(1, processed)
    metrics: Dict[str, float] = {}
    metrics["hit_rate"] = hit_counts.get(10, 0) / total
    metrics["mrr"] = mrr_sum / total
    for k in k_list:
        metrics[f"hit@{k}"] = hit_counts[k] / total
        metrics[f"recall@{k}"] = recall_sums[k] / total
        metrics[f"precision@{k}"] = precision_sums[k] / total
    print(
        "  Hit Rate       : {0:.3f}\n"
        "  MRR            : {1:.3f}\n"
        "  Hit@1          : {2:.3f}\n"
        "  Hit@3          : {3:.3f}\n"
        "  Hit@5          : {4:.3f}\n"
        "  Hit@10         : {5:.3f}\n"
        "  Recall@1       : {6:.3f}    Precision@1 : {7:.3f}\n"
        "  Recall@3       : {8:.3f}    Precision@3 : {9:.3f}\n"
        "  Recall@5       : {10:.3f}    Precision@5 : {11:.3f}\n"
        "  Recall@10      : {12:.3f}    Precision@10: {13:.3f}".format(
            metrics["hit_rate"],
            metrics["mrr"],
            metrics["hit@1"],
            metrics["hit@3"],
            metrics["hit@5"],
            metrics["hit@10"],
            metrics["recall@1"],
            metrics["precision@1"],
            metrics["recall@3"],
            metrics["precision@3"],
            metrics["recall@5"],
            metrics["precision@5"],
            metrics["recall@10"],
            metrics["precision@10"],
        )
    )
    return predictions, accuracy, metrics


def main():
    args = parse_args()
    device = torch.device(args.device)
    pooled_preexisting = args.pooled_kv_pt.exists() if args.pooled_kv_pt else False
    use_multi_gpu = (
        not args.disable_multi_gpu
        and device.type == "cuda"
        and torch.cuda.device_count() >= 2
    )
    if use_multi_gpu:
        print(
            f"[Multi-GPU] Enabled with up to {min(torch.cuda.device_count(), args.max_gpus)} GPUs."
        )
    model, tokenizer, primary_device = load_model_and_tokenizer(
        args.model_name, device, use_multi_gpu, args.max_gpus
    )
    base_model = model.module if hasattr(model, "module") else model
    attn_config = build_attn_config(base_model)

    chunk_entries, doc_index = prepare_chunk_entries(
        args,
        tokenizer,
        base_model,
        attn_config,
        primary_device,
    )

    qa_pairs = load_qa_pairs(args.qa_json, args.limit_qa)
    if len(qa_pairs) == 0:
        raise RuntimeError("QA dataset is empty after filtering.")

    question_cache = None
    cache_key_fn = None
    if args.question_emb_cache is not None:
        cache_key_fn = build_question_key
        cache_path = args.question_emb_cache
        cache_dtype = torch.float16 if args.question_emb_cache_dtype == "fp16" else torch.float32
        cache_valid = False
        cache_dirty = False
        cache_exists = cache_path.exists()
        if cache_exists:
            cached_embeddings, meta = load_question_emb_cache(cache_path)
            cache_valid = (
                meta.get("model_name") == args.model_name
                and meta.get("max_question_tokens") == args.max_question_tokens
            )
            if cache_valid:
                question_cache = {k: v.to(cache_dtype) for k, v in cached_embeddings.items()}
            elif args.question_emb_cache_only:
                raise RuntimeError(
                    "Question embedding cache metadata mismatch and --question-emb-cache-only was set."
                )
            else:
                question_cache = {}
                cache_dirty = True
        else:
            if args.question_emb_cache_only:
                raise RuntimeError("Question embedding cache missing and --question-emb-cache-only was set.")
            question_cache = {}
            cache_dirty = True

        if question_cache is not None:
            missing_qas = collect_missing_questions(qa_pairs, question_cache)
            if missing_qas:
                if args.question_emb_cache_only:
                    raise RuntimeError(
                        "Question embedding cache is missing entries and --question-emb-cache-only was set."
                    )
                new_embeddings = compute_question_embeddings(
                    missing_qas,
                    tokenizer,
                    base_model,
                    primary_device,
                    args.max_question_tokens,
                    batch_size=max(1, args.batch_size),
                )
                if new_embeddings:
                    for key, value in new_embeddings.items():
                        question_cache[key] = value.to(cache_dtype)
                    cache_dirty = True

            if question_cache and (cache_dirty or not cache_exists or not cache_valid):
                meta = {
                    "model_name": args.model_name,
                    "max_question_tokens": args.max_question_tokens,
                    "embedding_dim": int(next(iter(question_cache.values())).numel()),
                    "dtype": args.question_emb_cache_dtype,
                    "num_questions": len(question_cache),
                }
                save_question_emb_cache(cache_path, question_cache, meta)

    if args.eval_only:
        chunk_layer_matrix = build_chunk_layer_matrix(chunk_entries, args.kv_pool_layers).to(primary_device)
        question_head = ProjectionHead(base_model.config.hidden_size, args.proj_dim, args.head_layers).to(primary_device)
        chunk_head = ProjectionHead(chunk_layer_matrix.shape[2], args.proj_dim, args.head_layers).to(primary_device)
        pool_mlp = PoolMLP(base_model.config.hidden_size, chunk_layer_matrix.shape[0]).to(primary_device)
        meta = load_heads_checkpoint(args.load_heads, question_head, chunk_head, pool_mlp)

        eval_pairs = filter_qa_pairs_for_eval(qa_pairs, doc_index)
        if len(eval_pairs) == 0:
            raise RuntimeError("No QA samples matched corpus doc_ids for evaluation.")

        eval_predictions, eval_accuracy, eval_metrics = evaluate(
            eval_pairs,
            chunk_entries,
            question_head,
            chunk_head,
            pool_mlp,
            chunk_layer_matrix,
            tokenizer,
            base_model,
            primary_device,
            args,
            question_cache=question_cache,
            cache_key_fn=cache_key_fn,
        )

        output = {
            "run_mode": "eval_only",
            "heads_checkpoint": str(args.load_heads),
            "heads_checkpoint_resolved": str(args.load_heads.resolve()),
            "heads_meta": meta,
            "qa_json": str(args.qa_json),
            "qa_json_resolved": str(args.qa_json.resolve()),
            "model_name": args.model_name,
            "proj_dim": args.proj_dim,
            "kv_pool_layers": args.kv_pool_layers,
            "num_chunks": len(chunk_entries),
            "num_qa": len(qa_pairs),
            "test_accuracy": eval_accuracy,
            "test_metrics": eval_metrics,
            "test_predictions": eval_predictions,
            "test_samples": len(eval_pairs),
        }
        args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Wrote predictions to {args.output_json}")
        return

    rng = random.Random(args.split_seed)
    indices = list(range(len(qa_pairs)))
    rng.shuffle(indices)

    val_size = max(1, int(len(qa_pairs) * args.val_ratio)) if args.val_ratio > 0 else 0
    test_size = max(1, int(len(qa_pairs) * args.test_ratio)) if args.test_ratio > 0 else 0
    train_size = len(qa_pairs) - val_size - test_size
    if train_size <= 0:
        raise RuntimeError("Train split is empty; reduce val/test ratios.")

    train_indices = indices[:train_size]
    val_indices = indices[train_size: train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_pairs = [qa_pairs[i] for i in train_indices]
    val_pairs = [qa_pairs[i] for i in val_indices]
    test_pairs = [qa_pairs[i] for i in test_indices]

    train_dataset = QuestionChunkDataset(
        train_pairs,
        doc_index,
        tokenizer,
        args.max_question_tokens,
        question_cache=question_cache,
        cache_key_fn=cache_key_fn,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No QA samples matched corpus doc_ids for the train split.")

    val_dataset = QuestionChunkDataset(
        val_pairs,
        doc_index,
        tokenizer,
        args.max_question_tokens,
        question_cache=question_cache,
        cache_key_fn=cache_key_fn,
    ) if val_pairs else None

    test_dataset = QuestionChunkDataset(
        test_pairs,
        doc_index,
        tokenizer,
        args.max_question_tokens,
        question_cache=question_cache,
        cache_key_fn=cache_key_fn,
    ) if test_pairs else None

    print(f"Train dataset size: {len(train_dataset)} samples.")
    if val_dataset:
        print(f"Val dataset size: {len(val_dataset)} samples (from {len(val_pairs)} QA).")
    if test_dataset:
        print(f"Test dataset size: {len(test_dataset)} samples (from {len(test_pairs)} QA).")
    elif test_pairs:
        print("No test samples matched corpus doc_ids; skipping test evaluation.")

    question_head, chunk_head, pool_mlp, chunk_layer_matrix = train_contrastive(
        train_dataset,
        chunk_entries,
        base_model,
        tokenizer,
        primary_device,
        val_pairs,
        args,
        question_cache=question_cache,
        cache_key_fn=cache_key_fn,
    )

    train_predictions, train_accuracy, train_metrics = evaluate(
        train_dataset.matched_pairs,
        chunk_entries,
        question_head,
        chunk_head,
        pool_mlp,
        chunk_layer_matrix,
        tokenizer,
        base_model,
        primary_device,
        args,
        question_cache=question_cache,
        cache_key_fn=cache_key_fn,
    )

    val_predictions, val_accuracy, val_metrics = ([], None, {})
    if val_dataset and len(val_dataset) > 0:
        val_predictions, val_accuracy, val_metrics = evaluate(
            val_dataset.matched_pairs,
            chunk_entries,
            question_head,
            chunk_head,
            pool_mlp,
            chunk_layer_matrix,
            tokenizer,
            base_model,
            primary_device,
            args,
            question_cache=question_cache,
            cache_key_fn=cache_key_fn,
        )

    test_predictions, test_accuracy, test_metrics = ([], None, {})
    if test_dataset and len(test_dataset) > 0:
        test_predictions, test_accuracy, test_metrics = evaluate(
            test_dataset.matched_pairs,
            chunk_entries,
            question_head,
            chunk_head,
            pool_mlp,
            chunk_layer_matrix,
            tokenizer,
            base_model,
            primary_device,
            args,
            question_cache=question_cache,
            cache_key_fn=cache_key_fn,
        )

    output = {
        "run_mode": "train",
        "qa_json": str(args.qa_json),
        "qa_json_resolved": str(args.qa_json.resolve()),
        "init_heads_checkpoint": str(args.load_heads) if args.load_heads else None,
        "init_heads_checkpoint_resolved": str(args.load_heads.resolve()) if args.load_heads else None,
        "model_name": args.model_name,
        "proj_dim": args.proj_dim,
        "head_layers": args.head_layers,
        "kv_pool_layers": args.kv_pool_layers,
        "num_chunks": len(chunk_entries),
        "num_qa": len(qa_pairs),
        "train_accuracy": train_accuracy,
        "train_metrics": train_metrics,
        "val_accuracy": val_accuracy,
        "val_metrics": val_metrics,
        "test_accuracy": test_accuracy,
        "test_metrics": test_metrics,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
        "test_samples": len(test_dataset) if test_dataset else 0,
        "train_predictions": train_predictions,
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
    }
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote predictions to {args.output_json}")
    if args.save_heads:
        save_heads_checkpoint(args.save_heads, question_head, chunk_head, pool_mlp, output)
        print(f"Saved projection heads to {args.save_heads}")
    if args.pooled_kv_pt:
        if pooled_preexisting:
            print(f"Pooled KV checkpoint already existed at {args.pooled_kv_pt}; reused as-is.")
        else:
            print(f"Pooled KV vectors saved to {args.pooled_kv_pt}")


if __name__ == "__main__":
    main()
