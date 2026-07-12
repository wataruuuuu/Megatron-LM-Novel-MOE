# Copyright (c) 2026 Tohoku University. All rights reserved.

"""CLI arguments for layer-wise FFN redistribution and cross-layer FFN sharing.

Argument ``dest`` names match ``LayerwiseFFNTransformerConfig`` fields so that
``core_transformer_config_from_args`` collects them automatically.
"""


def _parse_sharing_groups(value):
    """Parse ``"0,1,2;5,6"`` into ``[[0, 1, 2], [5, 6]]``; empty string into ``[]``."""
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


def add_layerwise_ffn_args(parser):
    """Register layer-wise FFN arguments and return the parser."""
    group = parser.add_argument_group(title="layerwise-ffn")
    group.add_argument(
        "--ffn-hidden-dims",
        type=int,
        nargs="+",
        default=None,
        help="Per-layer FFN hidden size (one value per decoder layer). A value of 0 removes "
        "the FFN from that layer (attention-only). When unset, the model stays homogeneous "
        "and uses --ffn-hidden-size. Presence of this flag enables the layer-wise FFN path.",
    )
    group.add_argument(
        "--ffn-sharing-groups",
        type=_parse_sharing_groups,
        default=[],
        help='Cross-layer FFN sharing groups as "i,j,k;l,m" (0-based layer indices). Layers in '
        "a group share the smallest-index layer's FFN parameters.",
    )
    group.add_argument(
        "--share-pre-ffn-norm",
        action="store_true",
        default=True,
        help="Share the pre-MLP norm together with the FFN. With Transformer Engine the norm is "
        "fused into linear_fc1, so this is always effectively on.",
    )
    return parser
