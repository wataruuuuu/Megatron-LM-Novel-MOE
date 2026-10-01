# Layer-wise FFN Redistribution / Sharing (Megatron-LM 実装)

lingua `apps/layerwise_ffn_redistribution/transformer.py` の 2 機能を Megatron-LM (GPT / Transformer Engine バックエンド) に移植したもの。

- **層別 FFN 次元調整**: 層ごとに FFN 中間次元を変える。`0` を指定するとその層の FFN を除去し attention-only 層にする。
- **FFN の層間共有**: 指定した層グループで FFN パラメータ(TE では融合された pre-MLP Norm も含む)を共有する。

`megatron/core` は一切変更していない。追加はプロジェクト直下の `layerwise_ffn/` パッケージと、`gpt_builders.py` / `pretrain_gpt.py` への最小限の分岐のみ。

---

## 設計方針

### なぜ Megatron の heterogeneous 機構に乗せるか

Megatron には既に「層ごとに異なる config を適用する」仕組みがある。`TransformerConfig.heterogeneous_block_specs = True` のとき、`TransformerBlock._build_layers`(`megatron/core/transformer/transformer_block.py`)が層ごとに `config.get_config_for_layer(layer_number)` を呼び、その層専用の `TransformerConfig`(`ffn_hidden_size` などを上書き済み)でモジュールを構築する。

本実装はこのフックに相乗りすることで、**層別 FFN 次元を core 改変なしで実現**している。FFN 除去(dim=0)は、層 spec 側で MLP を `IdentityOp` に差し替えることで表現する。

### なぜ Transformer Engine (TE) 固定か

- baseline (`scripts/train_baseline_570M_24L.sh`) が TE で動作しており(`--transformer-impl` 未指定=既定 `transformer_engine`)、コンテナ(`nvcr.io/nvidia/pytorch:26.04-py3`)にも TE が同梱。比較可能性と速度の観点で TE を維持する。
- TE では **pre-MLP LayerNorm が `linear_fc1` に融合**され、`TransformerLayer.pre_mlp_layernorm` は `IdentityOp` になる。そのため FFN(`layer.mlp`)を共有するだけで pre-MLP Norm も自動的に共有され、lingua の `share_pre_ffn_norm=True`(デフォルト)と一致する。実装が単純になる。
- 逆に「FFN は共有するが Norm は層別」(`share_pre_ffn_norm=False`)は TE では表現できないため、明示的に禁止(assert)している。

---

## 追加・変更ファイル一覧

| ファイル | 種別 | 役割 |
| --- | --- | --- |
| `layerwise_ffn/__init__.py` | 新規 | パッケージ公開 API |
| `layerwise_ffn/config.py` | 新規 | `LayerwiseFFNTransformerConfig`(層別次元 + 共有マップ) |
| `layerwise_ffn/layer_specs.py` | 新規 | 層別ブロック spec(FFN 無し層を `IdentityOp` 化) |
| `layerwise_ffn/sharing.py` | 新規 | FFN 層間共有の後処理 + 検証 |
| `layerwise_ffn/arguments.py` | 新規 | CLI 引数追加 |
| `gpt_builders.py` | 変更(分岐追加) | config クラス選択 / spec 選択 / 共有適用 |
| `pretrain_gpt.py` | 変更(数行) | `extra_args_provider` に引数登録を合成 |
| `scripts/train_layerwise_ffn_570M_24L.sh` | 新規 | 実行例スクリプト |

---

## 各モジュールの詳細

### `config.py` — `LayerwiseFFNTransformerConfig`

`TransformerConfig` を継承した薄いサブクラス。

- 追加フィールド
  - `ffn_hidden_dims: list[int]` — 層数分の FFN 中間次元(`0` で FFN 除去)。未指定時は `ffn_hidden_size` で一様化。
  - `ffn_sharing_groups: list[list[int]]` — 共有する層グループ(0-based)。各グループの最小 index が master。
  - `share_pre_ffn_norm: bool` — TE では実質 True 固定(`_validate` で強制)。
- `__post_init__`
  - `heterogeneous_block_specs = True` をセットし、Megatron の per-layer フックを起動。
  - `_validate()` で整合性検証(次元数=層数、グループ重複なし、グループ内次元一致、dim=0 を共有グループに含めない、`share_pre_ffn_norm=False` を禁止)。
  - `ffn_idx_to_master`(slave→master の写像)を構築。
- `get_config_for_layer(layer_number)`
  - 1-based の層番号を受け取り、`ffn_hidden_dims[i] > 0` の層だけ `ffn_hidden_size` を上書きした **plain な `TransformerConfig`** を返す(再帰的な heterogeneous 処理を避けるため base 型で返す)。
  - 既存 `HeterogeneousTransformerConfig.get_config_for_layer` と同じ「`asdict` → base フィールドで絞る」方式。

### `layer_specs.py` — `build_layerwise_ffn_block_spec`

標準 TE 層 spec(`base_layer_spec`)を土台に、層ごとに `TransformerBlockSubmodules` を構築する。

- `ffn_hidden_dims[i] > 0` の層 → 標準 TE 層 spec を再利用(次元は `config` 側が層別供給)。
- `ffn_hidden_dims[i] == 0` の層 → spec を deepcopy し、`pre_mlp_layernorm=IdentityOp` / `mlp=IdentityOp` / `mlp_bda=IdentityFuncOp` に差し替え(attention-only)。
- パイプライン用に `offset:offset+num_layers` でスライス(PP=1 では全層)。

### `sharing.py` — `tie_ffn_across_layers`

GPTModel 構築後に呼び、slave 層の `mlp` を master 層の `mlp` オブジェクトへ **参照差し替え**する(= パラメータ共有)。

- `TransformerLayer.layer_number`(グローバル 1-based)で層を引き当てる。
- master/slave が別 PP rank に分かれている場合はエラー(同一 rank でなければモジュール共有不可)。
- `validate_ffn_sharing` で `is` 比較による共有チェック(lingua の `validate_ffn_sharing` 相当)。

**なぜこのタイミングか**: builder 内(optimizer 構築前・DDP / Float16 ラップ前)で差し替えるため、optimizer も DDP も共有後のモジュールツリーを見る。`torch.nn.Module.parameters()` は共有パラメータを重複排除するので、共有 FFN は DDP バケットに 1 度だけ載り、複数層から呼ばれた勾配は同一 `.grad` に累積される。master の初期値が保持され、slave 側の重複初期化分は破棄される(lingua と同挙動)。

### `arguments.py`

`--ffn-hidden-dims`(nargs, int)、`--ffn-sharing-groups`(`"0,1;2,3"` 形式をパース)、`--share-pre-ffn-norm` を追加。dest 名を config フィールドに合わせているため、`core_transformer_config_from_args` が自動収集する。

### `gpt_builders.py` の分岐(3 か所)

`--ffn-hidden-dims` または `--ffn-sharing-groups` のいずれかが指定されたときだけ経路を切り替える(`_use_layerwise_ffn(args)` で判定。baseline はどちらも無指定なので影響なし)。共有グループのみ指定した場合、`ffn_hidden_dims` は None のまま `ffn_hidden_size` で一様化される(全層同一次元 + 一部共有)。

1. config 構築: `core_transformer_config_from_args(args, config_class=LayerwiseFFNTransformerConfig)`。
2. spec 選択: `build_layerwise_ffn_block_spec(config, base_layer_spec, vp_stage=vp_stage)`。
3. GPTModel 構築直後: `tie_ffn_across_layers(model, config)`。

### `pretrain_gpt.py`

`extra_args_provider` を「layer-wise FFN 引数 + (存在すれば)modelopt 引数」を合成する関数に置き換え。

---

## 使い方

### CLI

```bash
python -m torch.distributed.run ... pretrain_gpt.py \
    ... \
    --ffn-hidden-dims 4480 4480 ... 0 0 0 0 \   # 層数分(例: 24 個)。0 は FFN 除去
    --ffn-sharing-groups "0,1;2,3"              # 層 0,1 と 2,3 でそれぞれ FFN 共有
```

- `--ffn-hidden-dims` の要素数は `--num-layers` と一致必須。
- 共有グループ内の層は同じ FFN 次元でなければならない(検証で assert)。
- 共有グループに dim=0 の層を含めてはならない。

### 実行例スクリプト

`scripts/train_layerwise_ffn_570M_24L.sh` は baseline を複製し、例として

- 後半 4 層(20–23)を attention-only(dim=0)
- 層 0,1 / 2,3 でそれぞれ FFN を共有

を設定している。`--ffn-hidden-dims` / `--ffn-sharing-groups` の値は実験パラメータなので目的に応じて差し替える。

---

## 注意点・制約

- **TE 固定**: `--share-pre-ffn-norm False`(FFN 共有・Norm 層別)は TE では未対応で、指定すると assert で停止する。必要になったら local(非 TE)経路を別途追加する。
- **パイプライン並列**: 共有グループの master/slave は同一 PP stage に置く必要がある。異なる rank に分かれると `tie_ffn_across_layers` がエラーを出す(対象設定は PP=1 なので問題なし)。
- **分散チェックポイント**: 共有 FFN でも state_dict は層プレフィックスごとに保存されるため、保存時は重複するが読込は整合する(研究用途で許容)。
- **共有グループの制約**: グループ間で層 index の重複不可、グループ内で FFN 次元一致必須。

---

## 検証状況

- 構文チェック(`py_compile`)と純粋ロジック(`_parse_sharing_groups`)はローカルで確認済み。
- torch / Transformer Engine / megatron.core の実行時 import はコンテナ(Singularity)内でのみ可能なため、**モデル構築〜学習の end-to-end 動作はコンテナ内での実行が必要**。
- コンテナ内での推奨確認手順:
  1. `--train-iters` を 1〜2 に絞って起動し、モデル構築が通ること。
  2. パラメータ数が baseline より「共有 + dim 縮小/除去」分だけ減っていること。
  3. `tie` 後に master/slave の `id(layer.mlp)` 一致、dim=0 層の MLP が `IdentityOp` であること。
  4. loss が下がり始めること、共有 FFN の grad が正しく累積されること。
