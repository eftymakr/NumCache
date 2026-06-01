"""
KV Cache Initializer that loads pre-computed TrimKV-selected KV pairs.

Loads cache_init.pt files created by trimkv_init_caches.py, which contain
KV pairs selected by TrimKV's trained retention gate from the full document.
"""

import torch
from typing import Optional
from cartridges.cache import AttnConfig, KVCacheFactory, TrainableCache


class KVFromTrimKV(KVCacheFactory):
    """Initialize cache from pre-computed TrimKV selection."""

    class Config(KVCacheFactory.Config):
        trimkv_cache_path: str = ""  # Path to trimkv_cache_init.pt

    def initialize_kv_cache(
        self,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> TrainableCache:
        cache_data = torch.load(self.config.trimkv_cache_path, map_location=model.device)

        init_keys = [k.to(model.device) for k in cache_data['keys']]
        init_values = [v.to(model.device) for v in cache_data['values']]

        n_tokens = cache_data['n_tokens']
        print(f"  [KVFromTrimKV] Loaded {n_tokens} tokens from {self.config.trimkv_cache_path}")

        return TrainableCache(
            config=attn_config,
            init_keys=init_keys,
            init_values=init_values,
            num_frozen_tokens=self.config.num_frozen_tokens,
        )
