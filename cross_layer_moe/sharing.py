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

import re
from collections import Counter
from typing import Dict

import torch

from megatron.core.utils import unwrap_model


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

    _checkpoint_master_experts_only(model, config)


def _checkpoint_master_experts_only(model, config) -> None:
    """Save and load a shared expert pool once, under its master layer.

    ``TransformerBlock.sharded_state_dict`` walks layers by name, so without this a tied pool is
    written once per grouped layer as byte-identical copies. Besides inflating the checkpoint, a
    resume then swiglu-merges ``len(group)`` pools on the GPU, which OOMs into Megatron's per-expert
    CPU-merge fallback (``gc.collect()`` each time; ~1.5 h for d1024 with a 24-layer group).

    Each slave layer's ``sharded_state_dict`` drops its ``mlp.experts.`` keys, so checkpoint save
    and load only touch the master's copy. Checkpoints written before this change still load
    unchanged: their slave copies are simply not requested (the default ``assume_ok_unexpected``
    strictness ignores keys present only in the checkpoint) and equal the master's. Routers are left
    as they are: shared-router copies are small and ``src/router_subspace`` reads them per layer.

    Megatron ends a load with ``module.load_state_dict(strict=True)``, which would report the slave
    experts as missing; a post-hook removes exactly those keys from ``missing_keys``.
    """
    layers = _layers_by_global_index(model)
    slaves = {idx for idx, master in config.cross_layer_idx_to_master.items() if idx != master}

    for idx in slaves:

        def sharded_state_dict(
            prefix="", sharded_offsets=(), metadata=None, _orig=layers[idx].sharded_state_dict
        ):
            sd = _orig(prefix, sharded_offsets, metadata)
            return {k: v for k, v in sd.items() if not k.startswith(f"{prefix}mlp.experts.")}

        layers[idx].sharded_state_dict = sharded_state_dict

    # load_state_dict keys use the ModuleList position, which differs from the global index under PP.
    positions = [i for i, layer in enumerate(model.decoder.layers) if layer.layer_number - 1 in slaves]
    slave_experts = re.compile(rf"(^|\.)layers\.({'|'.join(map(str, positions))})\.mlp\.experts\.")

    def _ignore_missing_slave_experts(module, incompatible_keys):
        missing = incompatible_keys.missing_keys
        missing[:] = [k for k in missing if not slave_experts.search(k)]

    model.register_load_state_dict_post_hook(_ignore_missing_slave_experts)


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


def _grad_of(param) -> torch.Tensor:
    """Return the live gradient of a parameter, preferring Megatron's ``main_grad`` buffer.

    Megatron's DDP accumulates into a contiguous ``param.main_grad`` buffer rather than the
    autograd-populated ``param.grad``; return whichever is present so the check works under both.
    """
    main_grad = getattr(param, "main_grad", None)
    if main_grad is not None:
        return main_grad
    return param.grad


def assert_cross_layer_tying(model, config, *, check_grad: bool = False) -> None:
    """Verify at run time that every slave layer is still tied to its group master.

    Extends the construction-time :func:`validate_cross_layer_experts` with checks that matter once
    training is under way -- run it periodically (e.g. from the training loop) or after a checkpoint
    reload, where tying can silently break:

      - level 1 (identity + storage): the slave and master expert modules are the same object and
        every expert parameter shares the same ``data_ptr`` (and router too, when shared);
      - level 2 (sentinel): writing a value into the master pool is observed through the slave,
        proving they are literally the same tensor (value restored under ``no_grad``);
      - level 3 (optimizer de-dup): the shared pool's parameters appear exactly once in the de-dup
        ``model.parameters()`` (so the optimizer counts them once) yet ``len(group)`` times in the
        raw listing (so every grouped layer references them) -- this is the mechanism the optimizer
        and DDP rely on to accumulate every layer's gradient into one entry;
      - level 4 (gradient), when ``check_grad``: the shared pool's gradient buffer exists and is
        finite. This confirms the tied pool owns a single, live gradient buffer; the *dynamic* proof
        that gradient actually flows from every layer is added by ``cross_layer_moe.debug``.

    Raises ``AssertionError`` on the first violated invariant. No-op without a sharing group.
    """
    if not getattr(config, "cross_layer_expert_sharing_groups", None):
        return

    m = unwrap_model(model)
    shared_router = getattr(config, "cross_layer_expert_sharing_shared_router", False)
    layers = _layers_by_global_index(m)

    # Occurrence counts for level-3: parameters() de-dups (the optimizer's view); the raw listing
    # keeps every reference, so a correctly tied pool appears once de-dup and len(group) times raw.
    dedup_counts = Counter(id(p) for p in m.parameters())
    raw_counts = Counter(id(p) for _, p in m.named_parameters(remove_duplicate=False))

    # Level 1: object + storage identity for every slave/master pair.
    for slave_idx, master_idx in config.cross_layer_idx_to_master.items():
        if slave_idx == master_idx:
            continue
        if slave_idx not in layers or master_idx not in layers:
            continue  # split across pipeline stages; tie_cross_layer_experts already rejects this.
        s_mlp, m_mlp = layers[slave_idx].mlp, layers[master_idx].mlp
        assert s_mlp is not m_mlp, (
            f"layer {slave_idx} must keep its own MoELayer wrapper (only sub-modules are tied)"
        )
        assert s_mlp.experts is m_mlp.experts, (
            f"layer {slave_idx} experts are no longer tied to master layer {master_idx}"
        )
        for ps, pm in zip(s_mlp.experts.parameters(), m_mlp.experts.parameters()):
            assert ps is pm and ps.data_ptr() == pm.data_ptr(), (
                f"layer {slave_idx} expert storage diverged from master layer {master_idx}"
            )
        if shared_router:
            assert s_mlp.router is m_mlp.router, (
                f"layer {slave_idx} router is no longer tied to master layer {master_idx}"
            )
        else:
            assert s_mlp.router is not m_mlp.router, (
                f"layer {slave_idx} router must stay independent from master layer {master_idx}"
            )

    # Levels 3 + 4: per group, check the shared pool's optimizer de-dup and (optionally) gradient.
    for grp in config.cross_layer_expert_sharing_groups:
        if not grp:
            continue
        master_idx = min(grp)
        size = len(grp)
        if master_idx not in layers:
            continue
        shared_modules = [layers[master_idx].mlp.experts]
        if shared_router:
            shared_modules.append(layers[master_idx].mlp.router)
        for module in shared_modules:
            for p in module.parameters():
                assert dedup_counts[id(p)] == 1, (
                    f"group master {master_idx}: a shared param is counted "
                    f"{dedup_counts[id(p)]} times by the optimizer (expected 1)"
                )
                assert raw_counts[id(p)] == size, (
                    f"group master {master_idx}: a shared param is referenced by "
                    f"{raw_counts[id(p)]} layers (expected {size}); tying is incomplete"
                )
                if check_grad:
                    grad = _grad_of(p)
                    assert grad is not None, (
                        f"group master {master_idx}: shared param has no gradient buffer; "
                        f"the tied pool is not receiving gradient"
                    )
                    assert torch.isfinite(grad).all(), (
                        f"group master {master_idx}: shared param gradient is non-finite"
                    )

    # Level 2: sentinel write/read on the first group, proving the slave sees the master's tensor.
    _sentinel_check(config, layers)


def _sentinel_check(config, layers) -> None:
    """Write a value into a master expert tensor and read it back through a slave (then restore)."""
    for slave_idx, master_idx in config.cross_layer_idx_to_master.items():
        if slave_idx == master_idx:
            continue
        if slave_idx not in layers or master_idx not in layers:
            continue
        master_params = list(layers[master_idx].mlp.experts.parameters())
        slave_params = list(layers[slave_idx].mlp.experts.parameters())
        if not master_params:
            return
        mp, sp = master_params[0], slave_params[0]
        with torch.no_grad():
            flat_m = mp.data.reshape(-1)
            flat_s = sp.data.reshape(-1)
            original = flat_m[0].clone()
            try:
                flat_m[0] = original + 1234.5
                # Read both sides back from storage so any dtype rounding (e.g. fp8) is identical;
                # if the tensor is shared the slave observes exactly the master's stored value, and
                # the value must have actually changed from the original.
                observed_master = flat_m[0].item()
                observed_slave = flat_s[0].item()
                assert observed_slave == observed_master, (
                    f"sentinel written to master layer {master_idx} not visible through slave "
                    f"layer {slave_idx} (slave={observed_slave}, master={observed_master}); "
                    f"pool not shared"
                )
                assert observed_master != original.item(), (
                    f"sentinel write to master layer {master_idx} had no effect; cannot verify "
                    f"sharing with slave layer {slave_idx}"
                )
            finally:
                flat_m[0] = original
        return  # one sentinel pair is enough
