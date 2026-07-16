# Copyright (c) 2026 Tohoku University. All rights reserved.

"""Interactive tying-inspection tool for Cross-Layer Expert Sharing (CLES).

This is a *debugger tool*, not part of the training loop: set a breakpoint anywhere in the training
process (``torch.distributed.breakpoint(rank=0)`` or a plain ``breakpoint()``), then call
``check_cross_layer_tying`` with whatever ``model`` is in scope to confirm that slave layers are
still tied to their group master.

    (Pdb) from cross_layer_moe.debug import check_cross_layer_tying
    (Pdb) check_cross_layer_tying(model)

It wraps ``cross_layer_moe.sharing.assert_cross_layer_tying`` for convenient interactive use --
unwrapping DDP/Float16 wrappers, handling a list of pipeline/VPP chunks, resolving the config from
the model, auto-detecting whether gradients are present, and printing a readable per-group report --
and covers the structural levels:

  - level 1 (identity / storage): slave and master expert modules are the same object and share
    ``data_ptr`` (router too, when shared);
  - level 2 (sentinel): a value written into the master pool is observed through the slave;
  - level 3 (optimizer de-dup): the shared pool appears once in de-dup ``parameters()`` (the
    optimizer's view) yet ``len(group)`` times in the raw listing (every layer references it);
  - level 4 (gradient): when gradients are present (i.e. called after backward) the shared pool's
    gradient buffer exists and is finite.
"""

from collections import Counter

import torch

from cross_layer_moe.sharing import _grad_of, _layers_by_global_index, assert_cross_layer_tying
from megatron.core.utils import unwrap_model


def _rank_tag() -> str:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return f"[rank {torch.distributed.get_rank()}] "
    return ""


def _as_chunks(model):
    return list(model) if isinstance(model, (list, tuple)) else [model]


def _resolve_config(base, config):
    return config if config is not None else getattr(base, "config", None)


def _grads_present(base, config) -> bool:
    """True if any shared expert pool already has a gradient buffer (i.e. backward has run)."""
    layers = _layers_by_global_index(base)
    for grp in config.cross_layer_expert_sharing_groups:
        if not grp:
            continue
        master_idx = min(grp)
        if master_idx not in layers:
            continue
        for p in layers[master_idx].mlp.experts.parameters():
            if _grad_of(p) is not None:
                return True
    return False


def check_cross_layer_tying(model, config=None, *, check_grad="auto", verbose=True) -> bool:
    """One-shot tying check for use from a debugger; returns True iff every group is healthy.

    ``model`` may be a single module or a list of pipeline/VPP chunks; wrappers (DDP/Float16Module)
    are unwrapped automatically. ``config`` is read from the model when omitted. ``check_grad``:
    ``"auto"`` includes the gradient check only when gradients are present (so the call works both
    before and after backward); pass ``True``/``False`` to force it. Prints a per-group report when
    ``verbose``; the first violated invariant is reported (and returned as ``False``).
    """
    all_ok = True
    for chunk_i, m in enumerate(_as_chunks(model)):
        base = unwrap_model(m)
        cfg = _resolve_config(base, config)
        if not getattr(cfg, "cross_layer_expert_sharing_groups", None):
            if verbose:
                print(f"{_rank_tag()}[CLES] chunk {chunk_i}: no sharing groups; nothing to check")
            continue

        cg = _grads_present(base, cfg) if check_grad == "auto" else bool(check_grad)
        err = None
        try:
            assert_cross_layer_tying(base, cfg, check_grad=cg)
        except AssertionError as e:
            err = e
            all_ok = False
        if verbose:
            _print_report(base, cfg, chunk_i, cg, err)
    return all_ok


def _print_report(base, config, chunk_i, check_grad, err) -> None:
    shared_router = getattr(config, "cross_layer_expert_sharing_shared_router", False)
    layers = _layers_by_global_index(base)
    dedup = Counter(id(p) for p in base.parameters())
    raw = Counter(id(p) for _, p in base.named_parameters(remove_duplicate=False))
    tag = _rank_tag()

    print(f"{tag}[CLES] chunk {chunk_i}: tying report (check_grad={check_grad})")
    for grp in config.cross_layer_expert_sharing_groups:
        if not grp:
            continue
        master_idx = min(grp)
        size = len(grp)
        if master_idx not in layers:
            print(f"{tag}  group L{grp}: master L{master_idx} not on this rank")
            continue
        master_exp = layers[master_idx].mlp.experts
        p0 = next(master_exp.parameters(), None)
        dcount = dedup[id(p0)] if p0 is not None else -1
        rcount = raw[id(p0)] if p0 is not None else -1
        tied = all(layers[s].mlp.experts is master_exp for s in grp if s in layers)
        grad_status = "n/a"
        if check_grad and p0 is not None:
            g = _grad_of(p0)
            grad_status = (
                "none" if g is None else ("finite" if torch.isfinite(g).all() else "NON-FINITE")
            )
        router_status = "shared" if shared_router else "independent"
        print(
            f"{tag}  group L{grp} (master L{master_idx}, size {size}): "
            f"experts_tied={tied}, opt_dedup={dcount}(exp 1), raw_refs={rcount}(exp {size}), "
            f"grad={grad_status}, router={router_status}"
        )
    print(f"{tag}  => {'PASS' if err is None else f'FAIL: {err}'}")
