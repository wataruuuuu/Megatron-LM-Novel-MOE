# Copyright (c) 2026 Tohoku University. All rights reserved.

"""Build slave-layer MoE experts on the meta device for Cross-Layer Expert Sharing (CLES).

``TransformerBlock._build_layers`` constructs every decoder layer before any of them can be
freed, and CLES gives each layer of a sharing group a ``num_moe_experts`` equal to the whole
group's pool size. So without this module a group of ``L`` layers materializes ``L`` full expert
pools on the GPU, and ``tie_cross_layer_experts`` throws ``L - 1`` of them away immediately
afterwards. The transient peak is ``L`` times the pool: for ``d1024 / 32 experts`` with a 24-layer
group (pool = 768 experts) that is 219 GiB against a 12 GiB model, which OOMs mid-construction.

A slave's pool is discarded, so it never needs storage. Megatron's Transformer Engine extension
already selects the parameter device from the config it is handed -- ``_get_extra_te_kwargs`` in
``megatron/core/extensions/transformer_engine.py`` maps ``init_model_with_meta_device`` to
``device="meta"`` -- and ``MoELayer`` passes its config straight to the experts builder
(``self.submodules.experts(self.num_local_experts, self.config, ...)``). Handing *that one
sub-module* a config copy with the flag set therefore costs nothing but the module tree's shape.

The flag must not be set on the layer config itself: that would put the slave's attention and
norms on meta as well, and Megatron materializes meta tensors with ``torch.empty_like``
(``to_empty_if_meta_device``), i.e. without re-initializing them. Scoping it to the experts
builder keeps every tensor that survives tying real and initialized.

No meta tensor outlives construction: ``tie_cross_layer_experts`` rebinds ``mlp.experts`` on every
slave to the master's module, dropping the meta one entirely. Nothing else holds a reference to it
-- the same reason the pools were fully freed before this change.
"""

import copy

from megatron.core.transformer.transformer_layer import get_transformer_layer_offset


def _meta_experts_builder(builder):
    """Wrap an ``ExpertsBuilder`` so that it allocates its parameters on the meta device."""

    def build(num_local_experts, config, **kwargs):
        # Shallow copy: every layer already gets a freshly built plain TransformerConfig from
        # CrossLayerExpertSharingConfig.get_config_for_layer, and copy.copy leaves __post_init__
        # alone (dataclasses.replace would re-run it). The copy is scoped to this one call so the
        # layer's attention and norms keep building on the real device.
        meta_config = copy.copy(config)
        meta_config.init_model_with_meta_device = True
        return builder(num_local_experts, meta_config, **kwargs)

    return build


def meta_init_slave_experts(block_spec, config, vp_stage=None) -> None:
    """Point every slave layer's experts builder at the meta device, in place.

    Called on the decoder block spec before ``GPTModel`` is constructed. No-op without a sharing
    group; the master of each group keeps building its pool for real, since that is the pool every
    layer of the group ends up sharing.
    """
    if not getattr(config, "cross_layer_expert_sharing_groups", None):
        return

    # block_spec.layer_specs is the slice this pipeline stage builds, so global layer indices have
    # to be shifted by the same offset get_gpt_decoder_block_spec used to cut it.
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    for layer_idx, master_idx in config.cross_layer_idx_to_master.items():
        if layer_idx == master_idx:
            continue
        local_idx = layer_idx - offset
        if not 0 <= local_idx < len(block_spec.layer_specs):
            # Built on another pipeline stage; tie_cross_layer_experts reports the split group.
            continue
        # get_gpt_decoder_layer_specs hands every MoE layer the *same* spec object, so this has to
        # copy before editing -- mutating in place would put the master on meta too.
        layer_spec = copy.deepcopy(block_spec.layer_specs[local_idx])
        moe_submodules = layer_spec.submodules.mlp.submodules
        moe_submodules.experts = _meta_experts_builder(moe_submodules.experts)
        block_spec.layer_specs[local_idx] = layer_spec
