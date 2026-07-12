# Copyright (c) 2026 Tohoku University. All rights reserved.

"""TransformerConfig subclass for layer-wise FFN redistribution and FFN sharing.

Per-layer FFN size is applied through Megatron's existing heterogeneous hook: setting
``heterogeneous_block_specs = True`` makes ``TransformerBlock._build_layers`` call
``get_config_for_layer`` for each layer, which here returns a plain ``TransformerConfig``
with ``ffn_hidden_size`` overridden for that layer. No ``megatron/core`` change is needed.

Cross-layer FFN sharing is not expressed here; it is applied after model construction by
``layerwise_ffn.sharing.tie_ffn_across_layers`` using ``ffn_idx_to_master`` computed below.
"""

from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional

from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class LayerwiseFFNTransformerConfig(TransformerConfig):
    """TransformerConfig with per-layer FFN width and cross-layer FFN sharing groups."""

    # Per-layer FFN hidden size, one value per decoder layer. 0 disables the FFN for that
    # layer (attention-only). When None, the model stays homogeneous at ``ffn_hidden_size``.
    ffn_hidden_dims: Optional[List[int]] = None

    # Groups of layer indices (0-based) that share one FFN. The smallest index in a group is
    # the master; the others reuse its FFN parameters.
    ffn_sharing_groups: Optional[List[List[int]]] = None

    # Share the pre-MLP norm together with the FFN. With Transformer Engine the norm is fused
    # into linear_fc1, so this is always effectively True (enforced in _validate).
    share_pre_ffn_norm: bool = True

    def __post_init__(self):
        super().__post_init__()

        # Route per-layer configs through Megatron's heterogeneous build hook.
        self.heterogeneous_block_specs = True

        if self.ffn_hidden_dims is None:
            self.ffn_hidden_dims = [self.ffn_hidden_size] * self.num_layers
        if self.ffn_sharing_groups is None:
            self.ffn_sharing_groups = []

        self._validate()

        # Map each shared (slave) layer index to the master (smallest) index of its group.
        self.ffn_idx_to_master: Dict[int, int] = {}
        for group in self.ffn_sharing_groups:
            if not group:
                continue
            master = min(group)
            for idx in group:
                self.ffn_idx_to_master[idx] = master

    def _validate(self) -> None:
        assert len(self.ffn_hidden_dims) == self.num_layers, (
            f"ffn_hidden_dims length ({len(self.ffn_hidden_dims)}) must equal "
            f"num_layers ({self.num_layers})"
        )

        # TE fuses the pre-MLP norm into linear_fc1, so an unshared norm cannot coexist with a
        # shared FFN. Keep the flag but forbid the unsupported combination up front.
        assert self.share_pre_ffn_norm, (
            "share_pre_ffn_norm=False is not supported with the Transformer Engine backend: "
            "the pre-MLP LayerNorm is fused into linear_fc1 and is shared with the FFN."
        )

        seen = set()
        for group in self.ffn_sharing_groups:
            for idx in group:
                assert 0 <= idx < self.num_layers, (
                    f"sharing group layer index {idx} out of range [0, {self.num_layers})"
                )
                assert idx not in seen, f"layer {idx} appears in more than one sharing group"
                seen.add(idx)
            dims = {self.ffn_hidden_dims[idx] for idx in group}
            assert len(dims) == 1, (
                f"all layers in sharing group {group} must have the same ffn_hidden_dim, got {dims}"
            )
            assert 0 not in dims, (
                f"sharing group {group} cannot contain an FFN-less (dim 0) layer"
            )

    def get_config_for_layer(self, layer_number: int) -> TransformerConfig:
        """Return a plain TransformerConfig for a 1-based layer number.

        Only ``ffn_hidden_size`` is overridden per layer. A plain TransformerConfig (not this
        subclass) is returned so that building the layer does not recurse into heterogeneous
        handling.
        """
        layer_idx = layer_number - 1
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(
                f"invalid layer number {layer_number}; expected 1..{self.num_layers}"
            )
        hidden_dim = self.ffn_hidden_dims[layer_idx]

        config_dict = asdict(self)
        base_field_names = {f.name for f in fields(TransformerConfig)}
        config_dict = {k: v for k, v in config_dict.items() if k in base_field_names}

        # The per-layer config builds a single layer, not another block; keep it homogeneous.
        config_dict["heterogeneous_block_specs"] = False
        if hidden_dim > 0:
            config_dict["ffn_hidden_size"] = hidden_dim
        # For hidden_dim == 0 the layer spec uses a no-op MLP, so ffn_hidden_size is unused.

        return TransformerConfig(**config_dict)
