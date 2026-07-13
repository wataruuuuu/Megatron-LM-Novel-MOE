# Copyright (c) 2026 Tohoku University. All rights reserved.

"""CLI arguments for Cross-Layer Expert Sharing (CLES) in MoE.

Cross-Layer Expert Sharing ties one MoE expert pool across several decoder layers; it is a
distinct concept from a MoE *shared expert* (an always-on expert inside a single layer).

Argument ``dest`` names match ``CrossLayerExpertSharingConfig`` fields so that
``core_transformer_config_from_args`` collects them automatically.
"""


def _parse_sharing_groups(value):
    """Parse ``"4,5,6;10,11"`` into ``[[4, 5, 6], [10, 11]]``; empty string into ``[]``."""
    value = value.strip()
    if not value:
        return []
    groups = []
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        groups.append([int(tok) for tok in chunk.split(",") if tok.strip() != ""])
    return groups


def add_cross_layer_expert_sharing_args(parser):
    """Register Cross-Layer Expert Sharing arguments and return the parser."""
    group = parser.add_argument_group(title="cross-layer-expert-sharing")
    group.add_argument(
        "--cross-layer-expert-sharing-groups",
        type=_parse_sharing_groups,
        default=[],
        help='Cross-layer expert sharing groups as "i,j,k;l,m" (0-based layer indices). Layers '
        "in a group share a single MoE block whose expert pool is the union of the group's "
        "per-layer experts: pool_size = len(group) * --num-experts. Each layer routes top-k over "
        "the shared pool. Presence of this flag enables the cross-layer expert sharing path.",
    )
    group.add_argument(
        "--cross-layer-expert-sharing-shared-router",
        action="store_true",
        default=False,
        help="Also share the router (gate) across a sharing group, so every layer routes over the "
        "shared pool with the same gate. When unset (default), only the experts are shared and "
        "each layer keeps an independent router over the shared pool.",
    )
    return parser
