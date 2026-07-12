# Copyright (c) 2026 Tohoku University. All rights reserved.

"""Layer-wise FFN redistribution and cross-layer FFN sharing for Megatron-LM GPT.

This package ports the lingua ``apps/layerwise_ffn_redistribution`` model to Megatron-LM:
  - per-layer FFN hidden size (a value of 0 removes the FFN from that layer)
  - cross-layer FFN parameter sharing groups

It is designed for the Transformer Engine backend and touches only the project-level
builder (``gpt_builders.py``); no ``megatron/core`` files are modified.
"""

from layerwise_ffn.arguments import add_layerwise_ffn_args
from layerwise_ffn.config import LayerwiseFFNTransformerConfig
from layerwise_ffn.layer_specs import build_layerwise_ffn_block_spec
from layerwise_ffn.sharing import tie_ffn_across_layers, validate_ffn_sharing

__all__ = [
    "add_layerwise_ffn_args",
    "LayerwiseFFNTransformerConfig",
    "build_layerwise_ffn_block_spec",
    "tie_ffn_across_layers",
    "validate_ffn_sharing",
]
