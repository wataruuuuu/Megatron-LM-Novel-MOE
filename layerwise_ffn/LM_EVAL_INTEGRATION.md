# lm-eval integration for `extra_args`-based architectures

This note documents how checkpoints trained with **architecture flags passed via
`extra_args_provider`** are loaded and evaluated through `lm-evaluation-harness`'s `megatron_lm`
backend, and — most importantly — **the procedure to follow when a new architecture is added**.

Two architectures currently go through it:

| feature | package | flags | eval-side work |
| --- | --- | --- | --- |
| layer-wise FFN redistribution / sharing | `layerwise_ffn` | `--ffn-hidden-dims`, `--ffn-sharing-groups`, `--share-pre-ffn-norm` | config subclass + block-spec wrap + tying |
| cross-layer MoE expert sharing (CLES) | `cross_layer_moe` | `--cross-layer-expert-sharing-groups`, `--cross-layer-expert-sharing-shared-router`, `--cross-layer-expert-sharing-cross-layer-aux-loss` | config subclass + tying (no spec change) |

CLES needs no spec change because the MoE module spec reads the expert count from the config at
build time; overriding per-layer `num_moe_experts` through the heterogeneous hook is enough.

It corresponds to the *adapter-only* recovery strategy (no change to Megatron core).

---

## 1. Background: why a special step is needed

The eval backend
(`lib/lm-evaluation-harness/lm_eval/models/megatron_lm.py`) builds the model with
`--use-checkpoint-args`, which restores the model structure from the checkpoint. However,
`megatron/training/checkpointing.py::load_args_from_checkpoint` restores only a **fixed allow-list**
of arguments (`_set_arg('num_layers')`, `_set_arg('ffn_hidden_size')`, …).

Architecture flags added at training time through `extra_args_provider`
(e.g. `pretrain_gpt.py` → `layerwise_ffn.arguments.add_layerwise_ffn_args`:
`--ffn-hidden-dims`, `--ffn-sharing-groups`, `--share-pre-ffn-norm`) are **not** in that allow-list.

Consequences if they are not restored:

- per-layer FFN widths default to homogeneous,
- attention-only (dim 0) layers are rebuilt *with* an FFN,
- shared FFNs are rebuilt as independent modules.

The first two change tensor shapes, so the module tree no longer matches the checkpoint and
`load_checkpoint(..., strict=True)` fails (or, worse, loads a structurally different model).

The third is milder and worth stating precisely, because the same reasoning applies to
cross-layer expert sharing. `state_dict` walks the module tree **by name, not by object
identity**, so a tied module is written out once per layer that references it — as byte-identical
copies, since they are literally one storage. The checkpoint therefore always carries a full set
of per-layer keys, and an un-tied rebuild still loads successfully with the correct values. What
is lost is only the structural invariant and the memory saving (`len(group)` redundant copies
resident). Tying is still applied at eval time so the eval module tree is identical to the
trained one.

The values **are** saved — they live in the checkpoint's `state_dict['args']` namespace — they are
just not copied onto the runtime `args`. The integration below re-reads them.

---

## 2. Change set (all in the eval backend, Megatron core untouched)

File: `lib/lm-evaluation-harness/lm_eval/models/megatron_lm.py`

### 2.1 Recovery of extra model args

- **`_CHECKPOINT_EXTRA_MODEL_ARGS`** (module-level tuple)
  The list of argparse `dest` names to recover from the checkpoint. **This is the extension point**
  (see §4).

- **`_recover_checkpoint_extra_model_args(args)`** (module-level)
  Re-reads the checkpoint's saved `args` via
  `megatron.training.checkpointing._load_base_checkpoint(args.load, args, rank0=True)`
  (rank-0 metadata only — no weights are loaded) and copies each field in
  `_CHECKPOINT_EXTRA_MODEL_ARGS` onto the live `args` when the checkpoint provides a non-`None`
  value.

- **Call site**: `MegatronLMEval._initialize_megatron`, immediately after `args = get_args()` and
  **before** `get_model(...)`, so the recovered values are visible when the model is built.

### 2.2 Layer-wise FFN model construction (mirrors `gpt_builders.gpt_builder`)

- **`_use_layerwise_ffn(args)`** (module-level) — gate identical to the training-side helper:
  `bool(args.ffn_hidden_dims or args.ffn_sharing_groups)`.

- Inside the nested `model_provider`:
  - **config** — when layer-wise FFN is active, build the config with
    `core_transformer_config_from_args(args, config_class=LayerwiseFFNTransformerConfig)` so the
    per-layer FFN hook (`heterogeneous_block_specs` / `get_config_for_layer`) is enabled.
  - **layer spec** — wrap the single-layer spec with
    `build_layerwise_ffn_block_spec(config, transformer_layer_spec)` so FFN-less (dim 0) layers get
    a no-op MLP. Done **before** the `attn_mask_type` override so that override iterates the
    per-layer specs.
  - **sharing** — after `GPTModel(...)` is constructed, call `tie_ffn_across_layers(model, config)`
    so slave-layer FFNs reference their group master. Must run **before** `load_checkpoint`.

The eval-specific behavior is preserved (`parallel_output=False`, the `attn_mask_type=arbitrary`
override, etc.); only the three layer-wise branches are added.

### 2.3 Cross-layer expert sharing model construction (mirrors `gpt_builders.gpt_builder`)

- **`_use_cross_layer_expert_sharing(args)`** (module-level) — gate identical to the training-side
  helper: `bool(args.cross_layer_expert_sharing_groups)`.

- Inside the nested `model_provider`:
  - **config** — build with
    `core_transformer_config_from_args(args, config_class=CrossLayerExpertSharingConfig)`, in the
    same `if/elif` chain as the layer-wise branch and in the same order as the training builder.
    The subclass sets `heterogeneous_block_specs = True` and its `get_config_for_layer` overrides
    per-layer `num_moe_experts` to the group's pool size (`len(group) * num_experts`).
  - **layer spec** — *no change*. CLES implies MoE, so the existing `if args.num_experts:` branch
    already selects `get_gpt_decoder_block_spec`, the same call the training builder makes, and the
    MoE module spec reads the expert count from the config at build time.
  - **sharing** — after `GPTModel(...)`, call `tie_cross_layer_experts(model, config)`.
    `install_cross_layer_aux_loss` is deliberately **not** called: the pooled aux loss is a
    training-only concern.

- **Post-load check**: after `load_checkpoint(...)`, `assert_cross_layer_tying(model, config,
  check_grad=False)` re-verifies object/storage identity, the sentinel write-through, and the
  `parameters()` de-dup counts. The config used to build the model is stashed as
  `self._model_config` inside `model_provider` so it is reachable there.

If the config subclass is not applied, grouped layers are rebuilt with the base `num_experts` and
the checkpoint's pool-sized expert tensors fail to load — a shape mismatch under `strict=True`,
which is the reliable regression signal for this feature.

---

## 3. Data flow at eval time

```
eval_ckpt_1gpu.sh
  -> lm_eval --model megatron_lm ... (extra_args=--use-mcore-models)
       MegatronLMEval._initialize_megatron:
         initialize_megatron(...)                      # load_args_from_checkpoint (allow-list only)
         args = get_args()
         _recover_checkpoint_extra_model_args(args)     # <-- restores ffn_hidden_dims /
                                                        #     ffn_sharing_groups /
                                                        #     cross_layer_expert_sharing_groups ...
                                                        #     from checkpoint state_dict['args']
         get_model(model_provider):
           _use_layerwise_ffn(args)?                    # gate driven by the recovered args
             config = LayerwiseFFNTransformerConfig(...)        # per-layer FFN width hook
             spec   = build_layerwise_ffn_block_spec(...)       # dim0 -> no-op MLP
             GPTModel(...); tie_ffn_across_layers(...)          # shared FFNs
           _use_cross_layer_expert_sharing(args)?       # gate driven by the recovered args
             config = CrossLayerExpertSharingConfig(...)        # per-layer num_moe_experts hook
             spec   = get_gpt_decoder_block_spec(...)           # unchanged MoE path
             GPTModel(...); tie_cross_layer_experts(...)        # shared expert pool
         load_checkpoint(..., strict=True)              # module tree now matches the checkpoint
         assert_cross_layer_tying(..., check_grad=False)  # CLES only: tying survived the load
```

For a **standard checkpoint** (no extra flags saved), `_recover_...` sets nothing, every gate is
`False`, and the original homogeneous build path runs unchanged — fully backward compatible.

---

## 4. Extension procedure for a NEW architecture

When you add a new architecture whose configuration is passed at **training time** through an
`extra_args_provider` flag and must be reproduced at eval time, follow these steps. The design goal
is that steps 3–4 are the *only* eval-side edits needed.

1. **Training side (as usual).** Register the flag with an `extra_args_provider` and route it into
   the model config (e.g. a `TransformerConfig` subclass and/or a custom layer/block spec), exactly
   as `layerwise_ffn` does. Make sure the argparse `dest` name matches the config field name so
   `core_transformer_config_from_args` picks it up.

2. **Confirm the flag is *not* in Megatron's allow-list.** Check
   `megatron/training/checkpointing.py::load_args_from_checkpoint`. If it is already restored there,
   no eval-side work is needed. Custom flags almost never are.

3. **Register the `dest` name for recovery.** Add the argparse `dest` string(s) to
   `_CHECKPOINT_EXTRA_MODEL_ARGS` in
   `lib/lm-evaluation-harness/lm_eval/models/megatron_lm.py`. This is enough for the value to be
   copied from the checkpoint onto the eval-time `args` automatically — the recovery function is
   generic and needs no other change.

4. **Wire the construction into `model_provider`.** In the same file's nested `model_provider`,
   add the branch(es) that consume the recovered args, mirroring the training-time builder:
   - build the config with the right `config_class` (if you use a `TransformerConfig` subclass),
   - select / wrap the layer spec,
   - apply any post-construction tying/patching **before** `load_checkpoint`.
   Guard each branch with a small `_use_<feature>(args)` helper that mirrors the training-side gate,
   so standard checkpoints are unaffected.

5. **Verify.** Run `scripts/eval_ckpt_1gpu.sh` against a checkpoint trained with the new flag. A
   successful `strict=True` load is the signal that the eval-time module tree matches training.
   The recovery function logs each restored field
   (`Recovered extra model arg from checkpoint: <name>=<value>`).

### Checklist (per new architecture)

- [ ] argparse `dest` == config field name (training side).
- [ ] `dest` added to `_CHECKPOINT_EXTRA_MODEL_ARGS`.
- [ ] `model_provider` branch added (config / spec / post-construction), guarded by a gate helper.
- [ ] `strict=True` checkpoint load succeeds in eval.

---

## 5. Caveats / notes

- **Private helper dependency.** Recovery uses
  `megatron.training.checkpointing._load_base_checkpoint` (leading underscore). If a future Megatron
  update changes its signature/return, update `_recover_checkpoint_extra_model_args` accordingly.
  It currently returns `(state_dict, checkpoint_name, release, ckpt_type)` and accepts
  `(load_dir, args, rank0=...)`.
- **Two metadata reads.** The checkpoint's rank-0 metadata is read once inside
  `initialize_megatron` and once again in the recovery function. `rank0=True` keeps this cheap
  (no weights).
- **Pipeline parallelism.** The eval backend forces `PP=1`, which satisfies
  `build_layerwise_ffn_block_spec`'s offset slicing and the "master and slave on the same stage"
  requirement of both `tie_ffn_across_layers` and `tie_cross_layer_experts`.
- **Expert parallelism with CLES.** A grouped layer's pool is `len(group) * num_experts`, so an
  eval run with `expert_model_parallel_size > 1` needs EP to divide that pool size. EP=1
  (`eval_ckpt_1gpu.sh`) is unconstrained.
- **`None` vs falsy.** Recovery copies a field when its checkpoint value `is not None` (not by
  truthiness), so a meaningful `False` / `0` / `[]` from a future flag is restored faithfully.
- **Alternative strategies (not used here).**
  - *Megatron core 2-line change*: add `_set_arg('ffn_hidden_dims')` / `_set_arg('ffn_sharing_groups')`
    to `load_args_from_checkpoint`. Single source of truth (also helps training resume) but edits
    upstream core.
  - *`extra_args` on the eval CLI*: pass the flags manually via `extra_args` (requires
    `extra_args_provider=add_layerwise_ffn_args`). Works for space-separated `--ffn-hidden-dims`, but
    `--ffn-sharing-groups` values contain commas that collide with lm-eval's comma-separated
    `--model_args`, and it duplicates the training config (drift risk).
