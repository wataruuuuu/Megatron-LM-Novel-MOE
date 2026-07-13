# Copyright (c) 2026 Tohoku University. All rights reserved.

"""Cross-Layer Expert Sharing (CLES): tie one MoE expert pool across several decoder layers.

After the GPTModel is built, each slave layer in a sharing group has its MoE sub-modules replaced
by references to the group's master (smallest-index) layer. Both modes tie at the sub-module
level and keep each layer's own ``MoELayer`` object (``layer.mlp``), so per-layer state such as the
token dispatcher and, importantly, ``MoELayer.layer_number`` survives on every layer:

  - default (independent router): only ``layer.mlp.experts`` is shared; each layer keeps its own
    router over the shared pool.
  - option ``--cross-layer-expert-sharing-shared-router`` (shared gate): both ``layer.mlp.router``
    and ``layer.mlp.experts`` are shared, so every layer in the group routes over -- and is trained
    on -- the same pool with the same gate.

Sharing is applied inside the model builder, i.e. before the optimizer is created and before the
DDP / Float16 wrappers, so downstream components observe the already-tied module tree.
``torch.nn.Module.parameters()`` de-duplicates shared parameters, so a tied router/expert pool
occupies a single optimizer/DDP entry and accumulates gradients from every layer that reuses it.
Aux/z-loss need no special handling: each layer's forward invokes the (shared) router on its own
tokens and attaches its own load-balancing loss to its own activation via ``MoEAuxLossAutoScaler``,
so the per-invocation gradients sum naturally across the group.

Per-layer logging: when the router is shared, that router's ``layer_number`` is frozen to the
master's, which would collapse every group layer's aux/z-loss logging onto the master slot.
``_install_layer_number_logging_hooks`` restores correct attribution by re-stamping the shared
router with the current layer's number (read from the surviving per-layer ``MoELayer.layer_number``)
in a ``forward_pre_hook``. This is logging-only -- ``layer_number`` does not affect gradients or the
reduced total loss. The default (independent-router) mode needs no such hook: each layer already
has its own router with its own ``layer_number``.
"""

from typing import Dict

import torch


def _layers_by_global_index(model) -> Dict[int, torch.nn.Module]:
    # TransformerLayer.layer_number is a global 1-based index; return a 0-based map.
    return {layer.layer_number - 1: layer for layer in model.decoder.layers}


def tie_cross_layer_experts(model, config) -> None:
    """Tie every slave-layer MoE pool to its group master, in place."""
    if not getattr(config, "cross_layer_expert_sharing_groups", None):
        return

    shared_router = getattr(config, "cross_layer_expert_sharing_shared_router", False)
    layers = _layers_by_global_index(model)

    for slave_idx, master_idx in config.cross_layer_idx_to_master.items():
        if slave_idx == master_idx:
            continue
        # Both layers must live on this pipeline stage to share a Python module object.
        if slave_idx not in layers or master_idx not in layers:
            raise RuntimeError(
                f"Cross-layer expert sharing needs layers {master_idx} and {slave_idx} on the "
                f"same pipeline stage, but pipeline parallelism placed them on different ranks."
            )
        # Tie at the sub-module level in both modes; keep each layer's own MoELayer wrapper so its
        # per-layer layer_number (and token dispatcher) survive. Share the gate too only when asked.
        layers[slave_idx].mlp.experts = layers[master_idx].mlp.experts
        if shared_router:
            layers[slave_idx].mlp.router = layers[master_idx].mlp.router

    validate_cross_layer_experts(model, config)

    if shared_router:
        # The shared router carries the master's frozen layer_number; fix per-layer logging.
        _install_layer_number_logging_hooks(model, config)


def _install_layer_number_logging_hooks(model, config) -> None:
    """Re-stamp the shared router with the current layer's number before each forward.

    Installed only for shared-router mode. Sharing ``layer.mlp.router`` across a group freezes the
    router's ``layer_number`` to the master's, so every group layer's aux/z-loss logging would land
    in the master slot. Each layer still owns its ``MoELayer`` (``layer.mlp``) with the correct
    ``layer_number``, so a ``forward_pre_hook`` on that MoELayer pushes the current layer's number
    into the shared router right before it runs. Decoder layers execute sequentially, so the
    ``set -> record`` order holds and ``MoEMetricsTracker.record`` writes to the correct slot.

    Logging-only: ``layer_number`` does not affect gradients or the reduced total loss. The
    ``set -> forward`` ordering assumption can break under activation recompute / CUDA graph / VPP
    (a general MoE+recompute caveat); it is exact for the supported PP1, no-recompute setting.
    """
    layers = _layers_by_global_index(model)

    def _pre_hook(mlp, args):
        # ``mlp`` is this layer's own MoELayer; its layer_number is the correct per-layer value.
        if hasattr(mlp.router, "set_layer_number"):
            mlp.router.set_layer_number(mlp.layer_number)
        return None

    for idx in config.cross_layer_idx_to_master:  # keys are every grouped layer, masters included
        if idx not in layers:
            continue
        layers[idx].mlp.register_forward_pre_hook(_pre_hook)


def validate_cross_layer_experts(model, config) -> None:
    """Assert that every slave layer shares the exact expert pool of its master."""
    if not getattr(config, "cross_layer_expert_sharing_groups", None):
        return

    shared_router = getattr(config, "cross_layer_expert_sharing_shared_router", False)
    layers = _layers_by_global_index(model)
    for slave_idx, master_idx in config.cross_layer_idx_to_master.items():
        if slave_idx == master_idx:
            continue
        if slave_idx not in layers or master_idx not in layers:
            # Split across pipeline stages; tie_cross_layer_experts already rejects this.
            continue
        # Experts are shared in both modes, but each layer keeps its own MoELayer wrapper.
        assert layers[slave_idx].mlp.experts is layers[master_idx].mlp.experts, (
            f"layer {slave_idx} experts are not shared with master layer {master_idx}"
        )
        assert layers[slave_idx].mlp is not layers[master_idx].mlp, (
            f"layer {slave_idx} must keep its own MoELayer wrapper (only sub-modules are tied)"
        )
        if shared_router:
            assert layers[slave_idx].mlp.router is layers[master_idx].mlp.router, (
                f"layer {slave_idx} router is not shared with master layer {master_idx}"
            )
        else:
            assert layers[slave_idx].mlp.router is not layers[master_idx].mlp.router, (
                f"layer {slave_idx} router must stay independent from master layer {master_idx}"
            )
