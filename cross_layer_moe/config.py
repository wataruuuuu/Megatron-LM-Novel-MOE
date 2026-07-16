# Copyright (c) 2026 Tohoku University. All rights reserved.

"""TransformerConfig subclass for Cross-Layer Expert Sharing (CLES) in MoE.

Cross-Layer Expert Sharing ties one MoE expert pool across several decoder layers. It is a
distinct concept from a MoE *shared expert* (an always-on expert inside a single layer): here a
group of layers reuses the *same routed-expert pool*, sized ``len(group) * num_moe_experts``.

Per-layer expert count is applied through Megatron's existing heterogeneous hook: setting
``heterogeneous_block_specs = True`` makes ``TransformerBlock._build_layers`` call
``get_config_for_layer`` for each layer, which here returns a plain ``TransformerConfig`` with
``num_moe_experts`` overridden to the group's pool size. The MoE module spec is expert-count
agnostic (the count is read from config at build time), so no ``megatron/core`` change is needed.

The sharing itself is not expressed here; it is applied after model construction by
``cross_layer_moe.sharing.tie_cross_layer_experts`` using ``cross_layer_idx_to_master`` below.
"""

from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional

from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class CrossLayerExpertSharingConfig(TransformerConfig):
    """TransformerConfig with cross-layer MoE expert-pool sharing groups."""

    # Groups of layer indices (0-based) that share one MoE expert pool. The smallest index in a
    # group is the master; the others reuse its experts (and, in the base, its router). Each
    # group's pool size is ``len(group) * num_moe_experts``.
    cross_layer_expert_sharing_groups: Optional[List[List[int]]] = None

    # Router-sharing knob. When False (default), only the experts are shared across a group and
    # each layer keeps an independent router. When True, the router (gate) is shared across the
    # group as well, so every layer routes over the shared pool with the same gate.
    cross_layer_expert_sharing_shared_router: bool = False

    # Cross-layer (pooled) aux-loss knob. When False (default), each grouped layer computes its
    # normal per-layer load-balancing loss over the shared pool. When True, the group's routed-load
    # is pooled across all L layers and a single aux loss is attached once at the group's tail
    # layer (exact, per-microbatch), so each layer's router is pushed by the whole group's
    # congestion. Orthogonal to ``cross_layer_expert_sharing_shared_router``.
    cross_layer_expert_sharing_cross_layer_aux_loss: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.cross_layer_expert_sharing_groups is None:
            self.cross_layer_expert_sharing_groups = []

        # Route per-layer configs through Megatron's heterogeneous build hook only when needed.
        if self.cross_layer_expert_sharing_groups:
            self.heterogeneous_block_specs = True

        self._validate()

        # Map each shared (slave) layer index to its group master, and each grouped layer index to
        # the pool size that layer's router/experts should use.
        self.cross_layer_idx_to_master: Dict[int, int] = {}
        self.layer_to_pool_size: Dict[int, int] = {}
        for grp in self.cross_layer_expert_sharing_groups:
            if not grp:
                continue
            master = min(grp)
            pool_size = len(grp) * self.num_moe_experts
            for idx in grp:
                self.cross_layer_idx_to_master[idx] = master
                self.layer_to_pool_size[idx] = pool_size

    def _validate(self) -> None:
        if not self.cross_layer_expert_sharing_groups:
            return

        assert self.num_moe_experts is not None and self.num_moe_experts > 0, (
            "Cross-layer expert sharing requires an MoE model; set --num-experts > 0"
        )

        seen = set()
        for grp in self.cross_layer_expert_sharing_groups:
            for idx in grp:
                assert 0 <= idx < self.num_layers, (
                    f"sharing group layer index {idx} out of range [0, {self.num_layers})"
                )
                assert idx not in seen, f"layer {idx} appears in more than one sharing group"
                seen.add(idx)

    def get_config_for_layer(self, layer_number: int) -> TransformerConfig:
        """Return a plain TransformerConfig for a 1-based layer number.

        Only ``num_moe_experts`` is overridden per layer (to the group's pool size for grouped
        layers). A plain TransformerConfig (not this subclass) is returned so that building the
        layer does not recurse into heterogeneous handling.
        """
        layer_idx = layer_number - 1
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(
                f"invalid layer number {layer_number}; expected 1..{self.num_layers}"
            )

        config_dict = asdict(self)
        base_field_names = {f.name for f in fields(TransformerConfig)}
        config_dict = {k: v for k, v in config_dict.items() if k in base_field_names}

        # The per-layer config builds a single layer, not another block; keep it homogeneous.
        config_dict["heterogeneous_block_specs"] = False
        if layer_idx in self.layer_to_pool_size:
            config_dict["num_moe_experts"] = self.layer_to_pool_size[layer_idx]

        return TransformerConfig(**config_dict)
