# Copyright (c) 2026 Tohoku University. All rights reserved.

"""Cross-layer FFN parameter sharing for layer-wise FFN redistribution.

After the GPTModel is built, each slave layer in a sharing group has its ``mlp`` submodule
replaced by a reference to the group's master (smallest-index) layer MLP. With the Transformer
Engine backend the pre-MLP LayerNorm is fused into the MLP's ``linear_fc1``, so sharing the MLP
module also shares that norm (equivalent to lingua's ``share_pre_ffn_norm=True``).

Sharing is applied inside the model builder, i.e. before the optimizer is created and before
the DDP / Float16 wrappers, so downstream components observe the already-tied module tree.
``torch.nn.Module.parameters()`` de-duplicates shared parameters, so a tied FFN occupies a
single DDP bucket entry and accumulates gradients from every layer that reuses it.
"""

from typing import Dict

import torch


def _layers_by_global_index(model) -> Dict[int, torch.nn.Module]:
    # TransformerLayer.layer_number is a global 1-based index; return a 0-based map.
    return {layer.layer_number - 1: layer for layer in model.decoder.layers}


def tie_ffn_across_layers(model, config) -> None:
    """Tie every slave-layer MLP to its group master MLP, in place."""
    if not getattr(config, "ffn_sharing_groups", None):
        return

    layers = _layers_by_global_index(model)

    for slave_idx, master_idx in config.ffn_idx_to_master.items():
        if slave_idx == master_idx:
            continue
        # Both layers must live on this pipeline stage to share a Python module object.
        if slave_idx not in layers or master_idx not in layers:
            raise RuntimeError(
                f"FFN sharing needs layers {master_idx} and {slave_idx} on the same pipeline "
                f"stage, but pipeline parallelism placed them on different ranks."
            )
        layers[slave_idx].mlp = layers[master_idx].mlp

    validate_ffn_sharing(model, config)


def validate_ffn_sharing(model, config) -> None:
    """Assert that every slave layer shares the exact MLP object of its master."""
    if not getattr(config, "ffn_sharing_groups", None):
        return

    layers = _layers_by_global_index(model)
    for slave_idx, master_idx in config.ffn_idx_to_master.items():
        if slave_idx == master_idx:
            continue
        if slave_idx not in layers or master_idx not in layers:
            # Split across pipeline stages; tie_ffn_across_layers already rejects this.
            continue
        assert layers[slave_idx].mlp is layers[master_idx].mlp, (
            f"layer {slave_idx} MLP is not shared with master layer {master_idx}"
        )
