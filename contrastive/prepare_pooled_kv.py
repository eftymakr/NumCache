#!/usr/bin/env python3
"""
Prepare pooled KV vectors from doc cache roots (primary + fallback) and
run contrastive training using train_all.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch


CACHE_STEP_RE = re.compile(r"cache-step(\d+)\.pt$")
# Override the default cache root with --runs-root or the NUMCACHE_CACHE_DIR env var.
# Points to a directory of trained per-doc KV caches: one subdir per doc_id,
# each containing cache-step*.pt files produced by the training pipeline.
_DEFAULT_RUNS_ROOT = Path(
    os.environ.get("NUMCACHE_CACHE_DIR", str(Path(__file__).resolve().parent.parent / "trained_caches"))
)


@dataclass
class DocCache:
    doc_id: str
    cache_paths: List[Path]
    doc_dir: Path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_runs_root = _DEFAULT_RUNS_ROOT
    default_runs_roots = [_DEFAULT_RUNS_ROOT]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        nargs="*",
        default=default_runs_roots,
        help=(
            "Root directories containing doc_XXXX_* folders (trained per-doc KV caches). "
            "Provide multiple paths in priority order. "
            "Override via the NUMCACHE_CACHE_DIR env var."
        ),
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=default_runs_root / "train_all.json",
        help="Training QA JSON (default: train_all.json under runs-root).",
    )
    parser.add_argument(
        "--pooled-kv-pt",
        type=Path,
        default=default_runs_root / "pooled_kv_from_cache.pt",
        help="Output pooled KV checkpoint path (no per-layer vectors).",
    )
    parser.add_argument(
        "--pooled-kv-pt-layered",
        type=Path,
        default=default_runs_root / "pooled_kv_with_layers.pt",
        help="Output pooled KV checkpoint path with per-layer vectors.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=script_dir / "train_chunk_cache_contrastive_single_mlp_pool.py",
        help="Path to train_chunk_cache_contrastive_single_mlp_pool.py",
    )
    parser.add_argument("--kv-pool-layers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--proj-dim", type=int, default=None)
    parser.add_argument("--head-layers", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--log-every-epochs", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument(
        "--question-emb-cache",
        type=Path,
        default=default_runs_root / "question_emb_cache.pt",
    )
    parser.add_argument("--question-emb-cache-only", action="store_true")
    parser.add_argument(
        "--question-emb-cache-dtype",
        type=str,
        choices=("fp16", "fp32"),
        default="fp16",
    )
    parser.add_argument(
        "--only-docs-in-train",
        action="store_true",
        help="Only include docs whose doc_id appears in --train-json.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit number of doc caches processed (debug).",
    )
    parser.add_argument(
        "--no-mlp-pool",
        action="store_true",
        help="Disable per-layer vectors; use pooled KV only.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild pooled KV checkpoint even if it exists.",
    )
    parser.add_argument(
        "--run-train",
        action="store_true",
        help="Run training after preparing pooled KV vectors.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip pooled KV preparation step.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results" / "automated_runs_merged",
        help="Directory for output JSON and heads checkpoints.",
    )
    parser.add_argument(
        "--cartridges-dir",
        type=Path,
        default=None,
        help="Optional CARTRIDGES_DIR to set when running training.",
    )
    parser.add_argument(
        "--no-auto-eval",
        action="store_true",
        help="Disable automatic VLO/PSX evals after training.",
    )
    return parser.parse_args()


def parse_doc_id(dir_name: str) -> str:
    parts = dir_name.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return dir_name


def list_doc_dirs(runs_root: Path) -> List[Path]:
    return sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("doc_")])


def load_train_doc_ids(train_json: Path) -> List[str]:
    data = json.loads(train_json.read_text(encoding="utf-8"))
    doc_ids = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        doc_id = entry.get("doc_id") or entry.get("chunk_id")
        if doc_id:
            doc_ids.append(doc_id)
    return doc_ids


def find_run_tag(cache_path: Path, training_output: Path) -> str:
    for parent in cache_path.parents:
        if parent.parent == training_output:
            return parent.name
    return ""


def cache_sort_key(cache_path: Path, training_output: Path) -> Tuple[str, int, int, float]:
    run_tag = find_run_tag(cache_path, training_output)
    is_last = 1 if cache_path.name == "cache_last.pt" else 0
    step = -1
    if not is_last:
        match = CACHE_STEP_RE.search(cache_path.name)
        if match:
            step = int(match.group(1))
    try:
        mtime = cache_path.stat().st_mtime
    except FileNotFoundError:
        mtime = 0.0
    return (run_tag, is_last, step, mtime)


def list_cache_candidates(doc_dir: Path) -> List[Path]:
    training_output = doc_dir / "training_output"
    if not training_output.exists():
        return []
    candidates = list(training_output.rglob("cache_last.pt"))
    candidates.extend(training_output.rglob("cache-step*.pt"))
    if not candidates:
        return []
    return sorted(candidates, key=lambda p: cache_sort_key(p, training_output), reverse=True)


def collect_doc_caches_from_roots(
    runs_roots: Iterable[Path],
    only_doc_ids: Optional[Iterable[str]] = None,
    max_docs: Optional[int] = None,
) -> Tuple[List[DocCache], List[str], Dict[str, Path]]:
    doc_ids_filter = set(only_doc_ids) if only_doc_ids is not None else None

    doc_by_id: Dict[str, Path] = {}
    doc_source: Dict[str, Path] = {}
    for runs_root in runs_roots:
        doc_dirs = list_doc_dirs(runs_root)
        root_doc_by_id: Dict[str, Path] = {}
        duplicates: List[str] = []
        for doc_dir in doc_dirs:
            doc_id = parse_doc_id(doc_dir.name)
            if doc_ids_filter is not None and doc_id not in doc_ids_filter:
                continue
            if doc_id in root_doc_by_id:
                duplicates.append(doc_id)
                if doc_dir.name > root_doc_by_id[doc_id].name:
                    root_doc_by_id[doc_id] = doc_dir
            else:
                root_doc_by_id[doc_id] = doc_dir
        if duplicates:
            print(
                f"[warn] Duplicate doc_ids in {runs_root}; using latest by name: "
                f"{', '.join(sorted(set(duplicates)))}"
            )
        for doc_id, doc_dir in root_doc_by_id.items():
            if doc_id in doc_by_id:
                continue
            doc_by_id[doc_id] = doc_dir
            doc_source[doc_id] = runs_root

    items: List[DocCache] = []
    missing: List[str] = []
    for doc_id in sorted(doc_by_id):
        doc_dir = doc_by_id[doc_id]
        cache_paths = list_cache_candidates(doc_dir)
        if not cache_paths:
            missing.append(doc_id)
            continue
        items.append(DocCache(doc_id=doc_id, cache_paths=cache_paths, doc_dir=doc_dir))
        if max_docs is not None and len(items) >= max_docs:
            break
    return items, missing, doc_source


def extract_cache_keys(cache_path: Path) -> List[torch.Tensor]:
    state = torch.load(cache_path, map_location="cpu", weights_only=False)
    trainable = list(state.get("trainable_keys") or [])
    frozen = list(state.get("frozen_keys") or [])
    keys = []
    for idx, train in enumerate(trainable):
        train = train.detach().to("cpu")
        if frozen:
            fixed = frozen[idx].detach().to("cpu")
            keys.append(torch.cat([fixed, train], dim=2))
        else:
            keys.append(train)
    if not keys:
        raise RuntimeError(f"No keys found in cache: {cache_path}")
    return keys


def pool_kv_features(kv_keys: List[torch.Tensor], last_n_layers: int) -> torch.Tensor:
    total_layers = len(kv_keys)
    if last_n_layers <= 0 or last_n_layers > total_layers:
        selected = kv_keys
    else:
        selected = kv_keys[-last_n_layers:]

    pooled_layers = []
    for layer_k in selected:
        layer_mean = layer_k.mean(dim=2).squeeze(0)
        pooled_layers.append(layer_mean)
    stacked = torch.stack(pooled_layers, dim=0)
    avg = stacked.mean(dim=0)
    return avg.flatten().to(torch.float32)


def build_pooled_kv(
    items: Iterable[DocCache],
    pooled_kv_pt: Path,
    kv_pool_layers: int,
    save_layer_vectors: bool = False,
) -> List[str]:
    doc_ids = []
    vectors = []
    layer_vectors = [] if save_layer_vectors else None
    for item in items:
        kv_keys = None
        chosen_path: Optional[Path] = None
        for candidate in item.cache_paths:
            try:
                kv_keys = extract_cache_keys(candidate)
                chosen_path = candidate
                break
            except Exception as exc:
                print(f"[warn] Failed to load {candidate}: {type(exc).__name__} {exc}")
        if kv_keys is None:
            print(f"[warn] Skipping {item.doc_id} (no valid cache found).")
            continue
        pooled = pool_kv_features(kv_keys, kv_pool_layers)
        if save_layer_vectors:
            total_layers = len(kv_keys)
            selected = kv_keys if kv_pool_layers <= 0 or kv_pool_layers > total_layers else kv_keys[-kv_pool_layers:]
            per_layer_vecs = []
            for layer_k in selected:
                layer_mean = layer_k.mean(dim=2).squeeze(0)
                per_layer_vecs.append(layer_mean.flatten().to(torch.float32))
            layer_vectors.append(per_layer_vecs)
        doc_ids.append(item.doc_id)
        vectors.append(pooled)
        if chosen_path is not None:
            print(f"[cache] {item.doc_id} -> {chosen_path}")

    if not doc_ids:
        raise RuntimeError("No cache files found under runs-root.")

    pooled_kv_pt.parent.mkdir(parents=True, exist_ok=True)
    payload = {"doc_ids": doc_ids, "pooled_kv": torch.stack(vectors)}
    if save_layer_vectors and layer_vectors is not None:
        payload["layer_vectors"] = layer_vectors
    torch.save(payload, pooled_kv_pt)
    print(f"Wrote pooled KV checkpoint: {pooled_kv_pt}")
    return doc_ids


def build_train_command(
    train_script: Path,
    train_json: Path,
    pooled_kv_pt: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    heads_path = output_dir / "projection_heads.pt"
    train_output = output_dir / "train_results.json"
    placeholder_corpus = output_dir / "placeholder_corpus.jsonl"
    placeholder_corpus.write_text("", encoding="utf-8")

    def add_opt(cmd: List[str], flag: str, value):
        if value is not None:
            cmd.extend([flag, str(value)])

    train_cmd = [
        sys.executable,
        str(train_script),
        "--qa-json",
        str(train_json),
        "--corpus-jsonl",
        str(placeholder_corpus),
        "--pooled-kv-pt",
        str(pooled_kv_pt),
        "--output-json",
        str(train_output),
        "--save-heads",
        str(heads_path),
        "--val-ratio",
        str(args.val_ratio),
        "--test-ratio",
        str(args.test_ratio),
        "--kv-pool-layers",
        str(args.kv_pool_layers),
    ]
    add_opt(train_cmd, "--device", args.device)
    add_opt(train_cmd, "--model-name", args.model_name)
    add_opt(train_cmd, "--batch-size", args.batch_size)
    add_opt(train_cmd, "--epochs", args.epochs)
    add_opt(train_cmd, "--lr", args.lr)
    add_opt(train_cmd, "--proj-dim", args.proj_dim)
    add_opt(train_cmd, "--head-layers", args.head_layers)
    add_opt(train_cmd, "--temperature", args.temperature)
    add_opt(train_cmd, "--log-every", args.log_every)
    add_opt(train_cmd, "--log-every-epochs", args.log_every_epochs)
    add_opt(train_cmd, "--question-emb-cache", args.question_emb_cache)
    add_opt(train_cmd, "--question-emb-cache-dtype", args.question_emb_cache_dtype)
    if args.question_emb_cache_only:
        train_cmd.append("--question-emb-cache-only")

    return train_cmd


def with_p_compression_suffix(path: Path) -> Path:
    if path.suffix:
        stem = path.stem
        if stem.endswith("_p_compression"):
            return path
        return path.with_name(f"{stem}_p_compression{path.suffix}")
    if path.name.endswith("_p_compression"):
        return path
    return path.with_name(f"{path.name}_p_compression")


def build_eval_command(
    train_script: Path,
    qa_json: Path,
    pooled_kv_pt: Path,
    heads_path: Path,
    output_json: Path,
    placeholder_corpus: Path,
    args: argparse.Namespace,
) -> List[str]:
    def add_opt(cmd: List[str], flag: str, value):
        if value is not None:
            cmd.extend([flag, str(value)])

    eval_cmd = [
        sys.executable,
        str(train_script),
        "--qa-json",
        str(qa_json),
        "--corpus-jsonl",
        str(placeholder_corpus),
        "--pooled-kv-pt",
        str(pooled_kv_pt),
        "--output-json",
        str(output_json),
        "--load-heads",
        str(heads_path),
        "--eval-only",
        "--kv-pool-layers",
        str(args.kv_pool_layers),
    ]
    add_opt(eval_cmd, "--device", args.device)
    add_opt(eval_cmd, "--model-name", args.model_name)
    add_opt(eval_cmd, "--proj-dim", args.proj_dim)
    add_opt(eval_cmd, "--head-layers", args.head_layers)
    add_opt(eval_cmd, "--question-emb-cache", args.question_emb_cache)
    add_opt(eval_cmd, "--question-emb-cache-dtype", args.question_emb_cache_dtype)
    add_opt(eval_cmd, "--log-every", args.log_every)
    add_opt(eval_cmd, "--log-every-epochs", args.log_every_epochs)
    if args.question_emb_cache_only:
        eval_cmd.append("--question-emb-cache-only")
    return eval_cmd


def main() -> None:
    args = parse_args()
    runs_roots = list(args.runs_root) if isinstance(args.runs_root, list) else [args.runs_root]
    if not runs_roots:
        runs_roots = [DEFAULT_PRIMARY_RUNS_ROOT, DEFAULT_FALLBACK_RUNS_ROOT]
    missing_roots = [root for root in runs_roots if not root.exists()]
    if missing_roots:
        print(f"[warn] runs-root not found (skipping): {', '.join(str(r) for r in missing_roots)}")
    runs_roots = [root for root in runs_roots if root.exists()]
    if not runs_roots:
        raise FileNotFoundError("No valid runs-root directories found.")
    if not args.train_json.exists():
        raise FileNotFoundError(f"train-json not found: {args.train_json}")
    if not args.train_script.exists():
        raise FileNotFoundError(f"train-script not found: {args.train_script}")

    use_mlp_pool = not args.no_mlp_pool
    pooled_kv_pt = args.pooled_kv_pt_layered if use_mlp_pool else args.pooled_kv_pt
    rebuild_needed = not args.skip_prepare and (args.force_rebuild or not pooled_kv_pt.exists())
    if rebuild_needed:
        pooled_kv_pt = with_p_compression_suffix(pooled_kv_pt)

    train_doc_ids = load_train_doc_ids(args.train_json)
    train_doc_set = set(train_doc_ids)
    print(f"Loaded train JSON: {args.train_json} ({len(train_doc_ids)} items, {len(train_doc_set)} docs)")

    only_doc_ids = train_doc_set if args.only_docs_in_train else None
    items, missing, doc_source = collect_doc_caches_from_roots(
        runs_roots, only_doc_ids=only_doc_ids, max_docs=args.max_docs
    )
    print(f"Collected {len(items)} doc caches for pooling.")
    if doc_source:
        counts: Dict[str, int] = {}
        for item in items:
            source = doc_source.get(item.doc_id)
            if source is None:
                continue
            key = str(source)
            counts[key] = counts.get(key, 0) + 1
        print("Doc cache roots used:")
        for root in runs_roots:
            key = str(root)
            if key in counts:
                print(f"  {key}: {counts[key]}")
    if missing:
        print(f"[warn] Missing cache for {len(missing)} doc_ids.")
        print(f"[warn] Example missing: {', '.join(missing[:10])}")

    if args.only_docs_in_train:
        covered = {item.doc_id for item in items}
        missing_from_train = sorted(train_doc_set - covered)
        if missing_from_train:
            print(f"[warn] train_json has {len(missing_from_train)} doc_ids with no cache.")

    if not args.skip_prepare:
        if pooled_kv_pt.exists() and not args.force_rebuild:
            print(f"Pooled KV checkpoint already exists: {pooled_kv_pt} (skipping rebuild)")
        else:
            build_pooled_kv(items, pooled_kv_pt, args.kv_pool_layers, save_layer_vectors=use_mlp_pool)
    elif not pooled_kv_pt.exists():
        raise FileNotFoundError(
            f"--skip-prepare was set but pooled KV checkpoint is missing: {pooled_kv_pt}"
        )

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    train_cmd = build_train_command(
        args.train_script,
        args.train_json,
        pooled_kv_pt,
        run_dir,
        args,
    )

    print("\n[train cmd]")
    print(" ".join(train_cmd))

    if args.run_train:
        env = os.environ.copy()
        if args.cartridges_dir is not None:
            env["CARTRIDGES_DIR"] = str(args.cartridges_dir)
            env["CARTRIDGES_OUTPUT_DIR"] = str(args.cartridges_dir / "outputs")
            existing_pythonpath = env.get("PYTHONPATH", "")
            if str(args.cartridges_dir) not in existing_pythonpath.split(os.pathsep):
                env["PYTHONPATH"] = str(args.cartridges_dir) + (
                    os.pathsep + existing_pythonpath if existing_pythonpath else ""
                )
        log_path = run_dir / "train.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            subprocess.run(train_cmd, check=True, env=env, stdout=log_file, stderr=log_file)
        print(f"Wrote log: {log_path}")

        if not args.no_auto_eval:
            eval_dir = run_dir / f"evals_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            placeholder = eval_dir / "placeholder_corpus.jsonl"
            placeholder.write_text("", encoding="utf-8")
            eval_log = eval_dir / "eval.log"
            qa_dir = Path(
                os.environ.get("NUMCACHE_QA_DIR", str(Path(__file__).resolve().parent.parent / "qa"))
            )
            eval_sets = [
                ("chunk_based", qa_dir / "chunk_based_qa_VLO_PSX.json"),
                ("company_comparison", qa_dir / "company_comparison_VLO_vs_PSX.json"),
                ("tracking", qa_dir / "tracking_qa_VLO_PSX.json"),
            ]
            heads_path = run_dir / "projection_heads.pt"
            with eval_log.open("w", encoding="utf-8") as log_file:
                for suffix, qa_path in eval_sets:
                    if not qa_path.exists():
                        print(f"[warn] Eval QA not found: {qa_path}", file=log_file)
                        continue
                    output_json = eval_dir / f"eval_{suffix}.json"
                    eval_cmd = build_eval_command(
                        args.train_script,
                        qa_path,
                        pooled_kv_pt,
                        heads_path,
                        output_json,
                        placeholder,
                        args,
                    )
                    print("\n[eval cmd]", file=log_file)
                    print(" ".join(eval_cmd), file=log_file)
                    subprocess.run(eval_cmd, check=True, env=env, stdout=log_file, stderr=log_file)
            print(f"Wrote evals to {eval_dir}")


if __name__ == "__main__":
    main()
