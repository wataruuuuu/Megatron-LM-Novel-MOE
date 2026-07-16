# Copyright (c) 2026 Tohoku University. All rights reserved.

"""Cross-layer (pooled) MoE aux loss for Cross-Layer Expert Sharing (CLES).

By default each grouped layer computes its own per-layer load-balancing (aux) loss over the shared
expert pool. With ``--cross-layer-expert-sharing-cross-layer-aux-loss`` the group's routed-load is instead
pooled across all ``L`` layers and a *single* aux loss is attached once, at the group's tail layer:

    L_cross = (E * coeff / L^2) * sum_i (sum_l f^l_i) * (sum_k P^k_i)

where ``f^l_i`` is layer ``l``'s (constant, non-differentiable) routed frequency for expert ``i`` and
``P^k_i`` is layer ``k``'s differentiable mean softmax score. This couples every grouped layer's
router to the whole group's congestion ``sum_l f^l_i`` (see ``docs/issue_20260714_aux_loss_gradient_4cases.md``).

Exactness with minimal memory. ``switch_load_balancing_loss_func`` only uses ``probs.sum(0)``, so the
loss is bilinear in ``(scores, tokens_per_expert)``. We therefore accumulate, across the group's
layers, only:

  - ``_score_sum`` = sum_k scores^k.sum(0)   -- a ``[pool_size]`` tensor that KEEPS its autograd graph
    back to every layer's router (this is what makes the tail-only attach exact for all L layers);
  - ``_token_sum`` = sum_l tokens^l          -- a ``[pool_size]`` detached (constant) tensor.

No full ``[num_tokens, pool_size]`` score tensors are retained; the per-layer backward graphs are
alive until the backward pass regardless, so the extra memory is just ``L`` small ``[pool_size]``
vectors. At the tail layer the single scalar is built and attached via the router's own
``attach_and_log_load_balancing_loss`` (reusing Megatron's per-token-loss scaling / DP averaging /
per-layer logging path). Because ``MoEAuxLossAutoScaler`` seeds the aux-loss backward with a constant
gradient independent of the activation it is attached to, a single attach at the tail correctly
distributes gradient to all L routers through the accumulated ``_score_sum`` graph.

Installed post-construction (like ``cross_layer_moe.sharing``): each grouped layer's router is given a
reference to its group's accumulator via ``router._cross_layer_aux_agg``, which the (generic,
opt-in) hook in ``TopKRouter._apply_aux_loss`` delegates to.

Caveats: assumes each grouped router fires exactly once per group per microbatch, in layer order
(exact for the supported PP1 / no-recompute / no-VPP setting; activation recompute or CUDA graph
would re-run the router forward and double-fire, a general MoE+recompute caveat).
"""

from typing import Dict

import torch

from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.transformer.moe.moe_utils import (
    get_tokens_per_expert_and_token_count,
    switch_load_balancing_loss_func,
)


class _CrossLayerAuxAggregator:
    """Pools one MoE group's aux loss across its layers and fires once at the tail layer.

    A single instance is shared by every router in a group (the same object in shared-router mode,
    or the ``L`` independent routers otherwise). It is called from ``TopKRouter._apply_aux_loss`` via
    the ``router._cross_layer_aux_agg`` hook, once per grouped layer per microbatch.
    """

    def __init__(self, group_layer_numbers):
        # 1-based global layer numbers of the group (as carried by TransformerLayer.layer_number).
        self._members = sorted(group_layer_numbers)
        self._head = self._members[0]
        self._tail = self._members[-1]
        self._num_layers = len(self._members)
        # Accumulators for the current microbatch; reset after the tail fires.
        self._score_sum = None  # sum_k scores^k.sum(0), keeps grad -> every layer's router
        self._token_sum = None  # sum_l tokens^l, detached (constant load)
        self._seen = 0

    def _reset(self):
        self._score_sum = None
        self._token_sum = None
        self._seen = 0

    def __call__(self, router, probs, scores_for_aux_loss, routing_map, with_padding_mask, coeff):
        tokens_per_expert, local_num_tokens, total_num_tokens = (
            get_tokens_per_expert_and_token_count(
                routing_map=routing_map,
                reduce_group=router.tp_cp_group,
                topk=router.topk,
                with_padding_mask=with_padding_mask,
            )
        )

        # Per-expert score sum (bilinearity: switch loss only needs probs.sum(0)).
        score_sum = scores_for_aux_loss.sum(dim=0)
        token_sum = tokens_per_expert.detach()

        # A fresh accumulation starts at the group head (defensive reset guards against a stale
        # partial accumulation left by an interrupted microbatch).
        if router.layer_number == self._head or self._score_sum is None:
            self._reset()
            self._score_sum = score_sum
            self._token_sum = token_sum.clone()
        else:
            self._score_sum = self._score_sum + score_sum
            self._token_sum = self._token_sum + token_sum
        self._seen += 1

        # Per-layer diagnostic logging (every grouped layer, not just the tail). This records each
        # layer's *own* switch load-balancing loss -- computed from that layer's f^k and P^k, exactly
        # what a non-shared MoE layer would report -- under the standard "load_balancing_loss" name.
        # It is NOT an optimization target: get_moe_metrics_tracker().record() detaches the value, so
        # no gradient flows from it (the trained loss is the pooled one attached at the tail). This
        # lets the individual balance trajectory of every shared layer be inspected and compared
        # against a non-shared baseline run under the same metric key, distinct from the group's
        # pooled loss (logged once at the tail under "cross_layer_load_balancing_loss").
        per_layer_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=router.topk,
            num_experts=router.config.num_moe_experts,
            moe_aux_loss_coeff=coeff,
            fused=router.config.moe_router_fusion,
        )
        # num_layers must match the tracker entry size force-initialized in report(), which is
        # num_layers here (MTP is out of scope for cross-layer expert sharing).
        get_moe_metrics_tracker().record(
            "load_balancing_loss",
            per_layer_loss / coeff,  # raw imbalance metric (coeff-independent), as the core path logs
            router.layer_number,
            router.config.num_layers,
            reduce_group=router.tp_cp_group,
            needs_dp_avg=True,
        )

        # Non-tail layers skip the pooled attach/logging; their P^k stays in the accumulated graph.
        if router.layer_number != self._tail:
            return probs

        # Tail: build the single pooled aux loss. Mirror switch_load_balancing_loss_func's
        # normalization (E * coeff / (topk * total_num_tokens^2)) and divide by L^2 so that
        #   sum_k g_k == L_cross  and  the magnitude stays comparable to a single layer's aux.
        num_experts = router.config.num_moe_experts
        norm = num_experts * coeff / (
            router.topk * total_num_tokens * total_num_tokens * self._num_layers * self._num_layers
        )
        aux_loss = norm * torch.sum(self._score_sum * self._token_sum.to(self._score_sum.dtype))

        probs = router.attach_and_log_load_balancing_loss(
            probs,
            coeff,
            aux_loss,
            "cross_layer_load_balancing_loss",
            router.tp_cp_group,
            valid_token_count=local_num_tokens,
        )
        self._reset()
        return probs


def _layers_by_global_index(model) -> Dict[int, torch.nn.Module]:
    # TransformerLayer.layer_number is a global 1-based index; return a 0-based map.
    return {layer.layer_number - 1: layer for layer in model.decoder.layers}


def install_cross_layer_aux_loss(model, config) -> None:
    """Attach a shared pooled-aux aggregator to every grouped layer's router.

    Called in the model builder right after ``tie_cross_layer_experts`` (before the optimizer / DDP
    wrappers). No-op unless both a sharing group and ``cross_layer_expert_sharing_cross_layer_aux_loss``
    are set.
    """
    if not getattr(config, "cross_layer_expert_sharing_groups", None):
        return
    if not getattr(config, "cross_layer_expert_sharing_cross_layer_aux_loss", False):
        return

    layers = _layers_by_global_index(model)
    for grp in config.cross_layer_expert_sharing_groups:
        if not grp:
            continue
        # Every layer of the group must live on this pipeline stage (same constraint as tying).
        if any(idx not in layers for idx in grp):
            raise RuntimeError(
                "Cross-layer pooled aux loss needs all layers of a sharing group on the same "
                f"pipeline stage, but group {grp} is split across pipeline ranks."
            )
        group_layer_numbers = [idx + 1 for idx in grp]  # 0-based indices -> 1-based layer numbers
        aggregator = _CrossLayerAuxAggregator(group_layer_numbers)
        for idx in grp:
            layers[idx].mlp.router._cross_layer_aux_agg = aggregator
