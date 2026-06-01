"""
Attention-Score-Based KV Cache Initialization

Instead of using compressed text (numcache) or raw text (p-init),
this initializer:
1. Feeds the FULL document through the model
2. Computes attention scores to identify the most important positions
3. Selects the top-k KV pairs based on cumulative attention received
4. Uses those as the starting cache for cartridge training

This is inspired by TrimKV's retention gate approach, but uses the model's
own attention patterns instead of a trained gate.
"""

import os
from pathlib import Path
from typing import Optional
import torch

from cartridges.cache import AttnConfig, KVCacheFactory, TrainableCache
from cartridges.initialization.tokenization_utils import MODEL_TO_SYSTEM_PROMPT_TOKENIZER


class KVFromAttentionSelection(KVCacheFactory):
    """Select most-attended KV pairs from full document as cache init."""

    class Config(KVCacheFactory.Config):
        max_cache_tokens: int = 512  # target cache size
        text_source: str = ""  # full document text path
        system_prompt_template: Optional[str] = "{text}"
        selection_method: str = "mean_attention"  # mean_attention or last_attention

    def initialize_kv_cache(
        self,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> TrainableCache:
        content = Path(self.config.text_source).read_text()
        if self.config.system_prompt_template is not None:
            content = self.config.system_prompt_template.format(text=content)

        model_key = tokenizer.name_or_path.lower()
        if model_key not in MODEL_TO_SYSTEM_PROMPT_TOKENIZER:
            if 'qwen' in model_key or 'qwen' in str(type(tokenizer)).lower():
                model_key = "qwen/qwen3-4b"
            else:
                model_key = "qwen/qwen3-4b"
        tokenize_data_into_system_prompt = MODEL_TO_SYSTEM_PROMPT_TOKENIZER[model_key]

        input_ids = tokenize_data_into_system_prompt(
            tokenizer=tokenizer,
            content=content,
            max_tokens=None,  # Use full document, no truncation
        ).squeeze(0)

        n_tokens = input_ids.shape[-1]
        target_size = min(self.config.max_cache_tokens, n_tokens)

        print(f"  [AttentionSelect] Full doc tokens: {n_tokens}, target cache: {target_size}")

        # If document is already smaller than target, just use all tokens (like p-init)
        if n_tokens <= target_size:
            print(f"  [AttentionSelect] Doc fits in cache, using all tokens (p-init mode)")
            init_cache = TrainableCache(config=attn_config)
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    input_ids = input_ids.to(model.device)
                    seq_ids = torch.full_like(input_ids, 0, dtype=torch.long)
                    position_ids = torch.arange(n_tokens, dtype=torch.long).to(model.device)
                    model(
                        input_ids=input_ids,
                        seq_ids=seq_ids,
                        position_ids=position_ids,
                        use_cache=True,
                        past_key_values=init_cache,
                        mode="generate",
                    )
                return TrainableCache(
                    config=attn_config,
                    init_keys=init_cache._keys,
                    init_values=init_cache._values,
                    num_frozen_tokens=self.config.num_frozen_tokens,
                )

        # Feed full document through model and collect KV pairs
        full_cache = TrainableCache(config=attn_config)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                input_ids = input_ids.to(model.device)
                seq_ids = torch.full_like(input_ids, 0, dtype=torch.long)
                position_ids = torch.arange(n_tokens, dtype=torch.long).to(model.device)

                # Get model output with attention weights
                outputs = model(
                    input_ids=input_ids,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    use_cache=True,
                    past_key_values=full_cache,
                    mode="generate",
                    output_attentions=True,
                )

        # Compute importance scores per position
        # Use attention weights if available, otherwise use KV norm as proxy
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            # Sum attention received by each position across all layers and heads
            # attention shape per layer: (batch, heads, seq_len, seq_len)
            importance = torch.zeros(n_tokens, device=model.device)
            for attn in outputs.attentions:
                # Sum over query positions and heads -> how much each key position is attended to
                importance += attn.squeeze(0).sum(dim=0).sum(dim=0)  # (seq_len,)
            print(f"  [AttentionSelect] Using attention weights for selection")
        else:
            # Fallback: use L2 norm of value vectors as importance proxy
            # Positions with larger value norms carry more information
            print(f"  [AttentionSelect] No attention weights, using value norm as proxy")
            importance = torch.zeros(n_tokens, device=model.device)
            for layer_idx in range(len(full_cache._values)):
                # values shape: (1, n_heads, seq_len, head_dim)
                v = full_cache._values[layer_idx]
                importance += v.squeeze(0).norm(dim=-1).sum(dim=0)  # sum over heads

        # Always keep first token (BOS) and last few tokens
        importance[0] = float('inf')  # always keep BOS

        # Select top-k positions
        _, selected_indices = importance.topk(target_size)
        selected_indices = selected_indices.sort().values  # maintain order

        print(f"  [AttentionSelect] Selected {len(selected_indices)} positions out of {n_tokens}")

        # Extract selected KV pairs
        selected_keys = []
        selected_values = []
        for layer_idx in range(len(full_cache._keys)):
            k = full_cache._keys[layer_idx]  # (1, n_heads, seq_len, head_dim)
            v = full_cache._values[layer_idx]
            selected_keys.append(k[:, :, selected_indices, :])
            selected_values.append(v[:, :, selected_indices, :])

        return TrainableCache(
            config=attn_config,
            init_keys=selected_keys,
            init_values=selected_values,
            num_frozen_tokens=self.config.num_frozen_tokens,
        )
