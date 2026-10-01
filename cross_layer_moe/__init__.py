# Copyright (c) 2026 Tohoku University. All rights reserved.

"""Cross-Layer Expert Sharing (CLES) for Megatron-LM MoE GPT.

Ties a single MoE routed-expert pool across a group of decoder layers: a group of ``L`` layers,
each nominally holding ``E`` experts, instead shares one pool of ``L * E`` experts, and every
layer routes top-k over that shared pool. This is distinct from a MoE *shared expert* (an
always-on expert inside one layer).

Designed for the Transformer Engine backend. The sharing itself touches only the project-level
builder (``gpt_builders.py``); the single ``megatron/core`` change is a generic opt-in hook in
``TopKRouter._apply_aux_loss`` used by the cross-layer pooled aux loss, which is a no-op unless a
router has an aggregator attached.
"""

from cross_layer_moe.arguments import add_cross_layer_expert_sharing_args
from cross_layer_moe.aux_loss import (
    get_cross_layer_aux_tracker,
    install_cross_layer_aux_loss,
    report_cross_layer_aux_loss,
)
from cross_layer_moe.config import CrossLayerExpertSharingConfig
from cross_layer_moe.debug import check_cross_layer_tying
from cross_layer_moe.meta_build import meta_init_slave_experts
from cross_layer_moe.sharing import (
    assert_cross_layer_tying,
    tie_cross_layer_experts,
    validate_cross_layer_experts,
)

__all__ = [
    "add_cross_layer_expert_sharing_args",
    "get_cross_layer_aux_tracker",
    "install_cross_layer_aux_loss",
    "report_cross_layer_aux_loss",
    "CrossLayerExpertSharingConfig",
    "check_cross_layer_tying",
    "meta_init_slave_experts",
    "assert_cross_layer_tying",
    "tie_cross_layer_experts",
    "validate_cross_layer_experts",
]
