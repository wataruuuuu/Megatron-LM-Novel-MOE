# Copyright (c) 2026 Tohoku University. All rights reserved.

"""Per-layer transformer block spec for layer-wise FFN redistribution (TE backend).

Layers whose FFN hidden dim is 0 get a no-op MLP (attention-only layer), mirroring the lingua
AttenOnly block. All other layers reuse the standard TE TransformerLayer spec; their FFN width
is supplied per layer by ``LayerwiseFFNTransformerConfig.get_config_for_layer``.
"""

import copy

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset


def build_layerwise_ffn_block_spec(config, base_layer_spec, vp_stage=None):
    """Build a TransformerBlockSubmodules with one layer spec per decoder layer.

    Args:
        config: LayerwiseFFNTransformerConfig holding ``ffn_hidden_dims``.
        base_layer_spec: ModuleSpec for a standard TE TransformerLayer (built by the caller
            via the usual ``_get_transformer_layer_spec``).
        vp_stage: Virtual pipeline stage, forwarded to the pipeline offset helpers.

    Returns:
        TransformerBlockSubmodules whose ``layer_specs`` cover the layers built on this
        pipeline stage.
    """
    layer_specs = []
    for hidden_dim in config.ffn_hidden_dims:
        if hidden_dim > 0:
            # Reuse the shared template; build_layer treats specs as read-only.
            layer_specs.append(base_layer_spec)
        else:
            # Attention-only layer: drop the FFN and its (TE-fused) pre-MLP norm.
            spec = copy.deepcopy(base_layer_spec)
            spec.submodules.pre_mlp_layernorm = IdentityOp
            spec.submodules.mlp = ModuleSpec(module=IdentityOp)
            spec.submodules.mlp_bda = IdentityFuncOp
            layer_specs.append(spec)

    # Keep only the layers built on this pipeline stage (identity slice when PP=1).
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    num_layers = get_num_layers_to_build(config, vp_stage=vp_stage)
    layer_specs = layer_specs[offset : offset + num_layers]

    return TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=TENorm)
