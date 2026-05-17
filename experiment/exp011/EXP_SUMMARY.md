# exp011 EXP_SUMMARY (2026-04-29 更新: Phase 番号ベースに再編)

## 目的

**timm + SED non-Perch 系の主力モデル** を構築。Discussion 683791 の Boredom (Master rank 2) / Salman / Ali Ozan のレシピを **1 Phase = 1 変更** で ablation 駆動的に積み上げ、最終的に exp010 NB3 (Perch + ProtoSSM, LB 0.914) と blend する。

**目標 LB**: 累積で Phase 13 まで進めて **0.94-0.95**。

## 管理ルール (2026-04-29 確定)

- **Phase 番号一本で管理**。Step 表記は廃止
- **1 Phase = 1 変更** (`feedback_isolate_changes.md` 厳守)
- 同 Phase 内の bugfix / PP 追加 / smoke test 修正は **v1, v2, v3...**
- LB 劣化したら次 Phase に進まず原因究明 or 別案
- **dual val (Val-A 既知site / Val-B 未知site)** と **推論 PP (file_conf_scale + adaptive δ-shift)** は Phase 2 以降の標準装備、差分カウントに含めない

旧 Phase 2 (Step 1) で 4 要素同時投入 → LB 0.819 (-0.035) で犯人不明だった反省から、ablation 駆動の Phase 体系に再編。

## ロードマップ (Phase 1 → 13)

| Phase | 前 Phase からの差分 (1 変更) | 累積 LB 見込 | 状態 |
|-------|----|---|---|
| **Phase 1** | (baseline) 10s, BCE, mel cache, spec mixup, author split, single seed | v2=0.854 ✅ | 完了 |
| ~~旧 Phase 2 (Step 1)~~ | (棄却) 4要素同時投入 | v1=0.819 ❌ | 教訓化 |
| **Phase 2** | + **20s 化のみ** (mel cache/BCE/spec mixup 維持) | v2=0.854 ✅ | 完了 |
| ~~Phase 3~~ | ~~+ Soft CE + Sigmoid infer~~ | v3=0.824 ❌ | 棄却 |
| **Phase 4** | + **raw waveform mixup** (ogg 直読に切替) | 0.86-0.88 | 進行中 |
| **Phase 5** | + **入力解像度 256×384** | 0.88-0.89 | 未着手 |
| **Phase 6** | + **過去年 pretrain** (BC25+BC21+AnuraSet → BC2026) | 0.89-0.91 | 未着手 |
| **Phase 7** | + **2025 公開 ckpt init** | 0.90-0.92 | 未着手 |
| **Phase 8** | + **Multi-context head** (5/10/20s 並列) | 0.91-0.92 | 未着手 |
| **Phase 9** | + **Pink Noise + Time Stretch aug** | 0.91-0.92 | 未着手 |
| **Phase 10** | + **iter pseudo R1** (threshold 0.4) | 0.92-0.94 | 未着手 |
| **Phase 11** | + **iter pseudo R2** (threshold 0.45) | 0.93-0.94 | 未着手 |
| **Phase 12** | + **blend with exp010 NB3** + per-class threshold | 0.93-0.95 | 未着手 |
| **Phase 13** | + **multi-seed (3 seed)** + isotonic calibration | 0.94-0.95 | 未着手 |

## Phase 1 設計 (まず動かす最小構成)

### モデル

| 要素 | 値 | 根拠 |
|------|-----|------|
| Backbone | **`tf_efficientnetv2_b0`** (timm) | Salman 単一 0.937 確定実績、Fused-MBConv で学習安定、推論速い |
| Head | SED + framewise attention pool (Linear→Tanh→Linear→softmax) | Salman: framewise max > LSEHead |
| Output | Linear(1280 → 234) | |

### 学習

| 要素 | 値 | 根拠 |
|------|-----|------|
| Input | mel cache 直読み, **10秒** (5秒×2 連結) | Salman/hengck23: long input で long-tail 緩和 |
| Mel | n_mels=256, sr=32000, hop=512, fmin=20, fmax=16000 | mel cache 既存仕様と一致 |
| Loss | **CrossEntropy on soft labels** (BCE フォールバック) | Salman: CE→Sigmoid at infer |
| Mixup | **Spectrogram mixup** (α=0.5) | mel cache 由来で raw waveform 持たないため。Phase 2 で raw 追加検討 |
| SpecAugment | freq_mask=30, time_mask=40 | exp006 流用 |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-4 | exp006 流用 |
| Scheduler | CosineAnnealingLR, T_max=20 | |
| Epochs | **20** | Salman 推奨 25 より少なめで様子見 (overconfident で AUC↓ 警告) |
| Batch | 64 (T4x2) | mel 256×625 ≈ メモリ余裕 |
| Fold | **Single fold (val=10% random split)** | Boredom 流 |
| Multi-seed | **なし** (単一モデル) | Boredom 流 |
| Mixup ルール | train_audio 同士 + (train_audio × labeled_SS) のみ。**unlabeled_SS との mixup 禁止** | D.M. 警告: unlabeled signal が混入してラベル不正化 |

### 推論 (Phase 1 nb2_submit)

| 要素 | 値 |
|------|-----|
| TTA | **なし** (Boredom 流) |
| Post-processing | file_confidence_scale (top_k=2, power=0.4) のみ |
| 目標推論時間 | < 30 min (CPU 90min 制約余裕) |

## 旧 Phase 2 (Step 1) 教訓 — 2026-04-28〜29

| 項目 | 内容 |
|------|------|
| 構成 | 20s + Soft CE + raw mixup + ogg 直読 + 256×384 (4-5要素同時投入) |
| 学習結果 | Wall 178 min, Best Val-A 0.8420 / Val-B 0.9233 @ Ep 4 (早期収束) |
| LB | **0.819** (Phase 1 v2 0.854 から **-0.035**) |
| 棄却理由 | 4 要素同時のため犯人切り分け不能 |
| 教訓 | `feedback_isolate_changes.md` 違反。以後 Phase 番号で 1 変更ずつ ablation |
| 学習 NB | https://www.kaggle.com/code/maekeso/birdclef2026-exp011-train-phase2-step1 |
| 推論 NB | https://www.kaggle.com/code/maekeso/birdclef2026-exp011-submit-phase2-step1 |

## Phase 2-13 の各 Phase 設計

### Phase 2: 20s 化のみ
- Phase 1 v2 のコードベースで chunk_duration を 10s → 20s
- mel cache (256×625) のままで 20s ぶん crop (時間軸 ~1251)
- BCE/spec mixup/author split 維持。**dual val は導入** (Val-A=labeled SS hold-out 16 files, Val-B=author 10%)
- 推論 PP は Phase 1 v2 を流用

### Phase 3: + Soft CE
- Loss を BCE → Soft CE on multi-hot (Salman レシピ)
- 推論時 Sigmoid を適用 (Salman 推奨: CE training + Sigmoid infer)

### Phase 4: + raw waveform mixup
- ogg 直読 + GPU mel transform に切替 (mel cache 卒業)
- raw waveform mixup α=0.5、ラベルは max OR
- 9h セッション内に収まらなければ waveform cache 7NB を作り直す

### Phase 5: + 入力解像度 256×384
- 時間軸を 384 に固定 (mel resize)

### Phase 6: + 過去年 pretrain
- Stage A: BC2025 + BC2021 + AnuraSet v2 (既存 mel cache 流用)、234-class head + 背景音
- Stage B: BC2026 train_audio + labeled_SS finetune
- 重複種: BC25 ∩ BC2026 ≈ 42, BC21 ≈ 34, AnuraSet ≈ 17
- Stage A: epochs=15, lr=5e-4, Stage B: epochs=15

### Phase 7: + 2025 公開 ckpt init
- Kaggle Models / Datasets を探索。見つからなければ自前 BC25 で代替 (= スキップ判定)

### Phase 8: + Multi-context head
- 5/10/20s 並列 head、共有 backbone (hengck23)
- 出力は 3 head の平均

### Phase 9: + Pink Noise + Time Stretch aug
- raw audio aug を追加

### Phase 10: + iter pseudo R1
- Phase 9 weight で `train_soundscapes/` 10,658 × 12 windows 推論
- threshold 0.4、各種 max 1000 pseudo positive、sample_weight = pseudo_score
- Phase 9 から finetune epochs=10

### Phase 11: + iter pseudo R2
- threshold 0.45、Phase 10 weight から finetune epochs=8

### Phase 12: + blend with exp010 NB3 + per-class threshold
- exp010 NB3 (LB 0.914) と Phase 11 を rank average blend
- 開始重み: PW=0.5 / SW=0.5 → Perch 苦手な非鳥類は SED 重視で per-class 調整
- per-class threshold は OOF で最適化

### Phase 13: + multi-seed + isotonic calibration
- Phase 11 を 2 seed 追加学習 (合計 3 seed)
- OOF で isotonic calibration

## データソース

| データ | Kaggle Dataset | mel cache slug | サイズ | ライセンス |
|--------|---------------|---------------|--------|-----------|
| BirdCLEF 2026 | `competitions/birdclef-2026` | n/a (オリジナル使用 + 別途 cache) | 15GB | コンペ規約 |
| BirdCLEF 2025 | (確認要) | `birdclef2026-mel-cache-bc2025-256-v2` ✅ | ? | コンペ規約 |
| BirdCLEF 2024 | (確認要) | `birdclef2026-mel-cache-bc2024-256-v2` ✅ | ? | コンペ規約 |
| BirdCLEF 2023 | (確認要) | `birdclef2026-mel-cache-bc2023-256` ✅ | ? | コンペ規約 |
| BirdCLEF 2022 | (確認要) | `birdclef2026-mel-cache-bc2022-256` ✅ | ? | コンペ規約 |
| BirdCLEF 2021 | (確認要) | `birdclef2026-mel-cache-bc2021-256` ✅ | ? | コンペ規約 |
| iNatSounds 2024 抜粋 | `shadowdude/train-recordings` | `birdclef2026-mel-cache-inat-256` ✅ | 121GB | CC-BY-NC系 (要確認) |
| AnuraSet v2 | `bengtlueers/anuraset-v2-raw` | `birdclef2026-mel-cache-anuraset-256` ✅ | 8.5GB | MIT |
| Caiman yacare | xeno-canto から収集 → 新規 Kaggle Dataset | (Phase 1 では不使用) | TBD | xeno-canto ルール |

mel cache 仕様: **sr=32000, n_fft=2048, hop=512, n_mels=256, fmin=20, fmax=16000, uint8 量子化 [-80,20]dB→[0,255], 全長保存**

## Coverage 参考 (外部データで救える非鳥類73種)

| | Amphibia 35 | Insecta 28 | Mammalia 8 | Reptilia 1 |
|--|--|--|--|--|
| AnuraSet v2 | 17 (49%) | 0 | 0 | 0 |
| iNatSounds | 24 (69%, 少量) | 3 (11%) | 4 (50%) | 0 |
| **Union** | 27 (77%) | 3 (11%) | 4 (50%) | 0 |
| Caiman 追加後 | 27 | 3 | 4 | **1** |

**Insecta sonotype 25種 + Amphibia 8種 + Mammalia 4種 は外部データ無し** → BC2026 train + iter pseudo + (Phase 4) Perch zero-shot blend で対応。

## ファイル構成

**方針 (2026-04-27 確定)**: 全コードを Kaggle NB 内に inline (no src/ modules)。`_gen_*.py` がローカル雛形、生成された `.ipynb` をそのまま push。

```
experiment/exp011/
├── EXP_SUMMARY.md               # このファイル
├── src/
│   ├── inat_subset.py               # 既存: iNat → primary_label マッピング
│   ├── inat_subset_mapping.csv      # 既存
│   └── inat_subset_filelist.csv     # 既存
├── notebook/
│   ├── _gen_nb1_train_p1.py         # ★Phase 1 学習 NB 生成 (12 cells, all inline)
│   ├── nb1_train_p1.ipynb           # ★Phase 1 学習 (T4x2)
│   ├── push_nb1_p1.py               # ★SDK push (machine_shape=NvidiaTeslaT4)
│   ├── _gen_nb1_train_p2.py         # 後続: Phase 2 学習 NB
│   ├── _gen_nb1_train_p3.py         # 後続: Phase 3 学習 NB
│   ├── _gen_nb2_submit.py           # 後続: 共通推論 NB 生成
│   └── nb2_submit.ipynb             # 後続: CPU 90min 推論
└── output/
    └── weights/                      # 各 Phase の best.pth (Kaggle Dataset 化して引き継ぎ)
```

**Kaggle slug 規約 (2026-04-27 改訂)**:
- SDK push の slug は **title から派生** する (memory `feedback_kaggle_slug_from_title.md`)
- 命名パターン: `birdclef2026-exp011-{phase}-{kind}` (slug は title を slug 形式に揃えて確定)
- 採用済み: `birdclef2026-exp011-train-phase1-sed` (Phase 1 学習 NB)
- 計画: `birdclef2026-exp011-train-phase2-sed`, `-train-phase3-sed`, `-submit-sed`
- 学習 weight は `birdclef2026-exp011-weights-phase{N}` の Dataset として登録
- 一度確定した slug は **変更不可** (memory `feedback_kaggle_slug_reservation.md`)

## 進捗

### Phase 0 (準備)
- [x] Scaffolding
- [x] iNat 抽出マッピング作成
- [x] mel cache 生成 NB × 7 (BC2021/22/23/24/25 + AnuraSet + iNat) 全 COMPLETE

### Phase 1 (Boredom 最小構成) — **完了 2026-04-28**
- [x] 構成方針確定: 全コード NB inline (no src/ modules)
- [x] `_gen_nb1_train_p1.py` + `nb1_train_p1.ipynb` (12 cells)
- [x] Phase 1 学習 NB push (Version 4 = num_workers=0, single GPU, smoke test 入り)
- [x] Phase 1 学習完走 (Wall **70.8 min**、Best Val Macro AUC **0.9645 @ Ep17**, 187/234 cls)
- [x] best.pth は Dataset 化せず **kernel_sources で直接参照**
- [x] `_gen_nb2_submit_p1.py` + `nb2_submit_p1.ipynb` (CPU 90min, 12 windows × 10-sec context)
- [x] Phase 1 推論 NB push (Version 1) → submit
- [x] **Phase 1 LB = 0.840** (期待 0.84-0.89 の下限)

#### Phase 1 教訓
- **val→LB ギャップ -0.124** が想定より大きい
- 原因 (a): 47/234 cls が val で評価されていない (希少種が author group split で除外)
- 原因 (b): author group split が site shift を吸収できず、LB regime と乖離
- 原因 (c): 推論 PP (file_conf_scale, δ-shift smoothing 等) を入れていない
- → **Phase 2 で dual validation (Val-A 既知site / Val-B 未知site) は必須**
- → 推論 NB に **file_confidence_scale** + **adaptive δ-shift smoothing** を最低限入れる (NB3 v20 で +0.010 実証済み)

### 旧 Phase 2 (Step 1) — **棄却 2026-04-29**
- [x] 4 要素同時投入で LB 0.819 (-0.035) → 教訓化
- [x] Phase 番号一本に再編

### Phase 2 (20s 化のみ) — **学習 NB push 済 2026-04-29**
- [x] `_gen_nb1_train_p2.py` + `nb1_train_p2.ipynb` (12 cells、chunk_duration=20.0、dual val 導入)
- [x] Phase 2 学習 NB push (`birdclef2026-exp011-train-phase2`, Version 1)
- [x] `_gen_nb2_submit_p2.py` + `nb2_submit_p2.ipynb` (Phase 1 v2 PP 流用、20s context、ローカル生成のみ)
- [x] 旧 Step 1 関連ファイルは `*_legacy_step1.py` に rename して保存
- [ ] Kaggle で Run All (T4x2、期待 wall ~3-4h)
- [ ] Phase 2 推論 NB push → submit
- [ ] Phase 2 LB 取得 (目標 0.86-0.87)

### ~~Phase 3 (+ Soft CE)~~ — **棄却 2026-05-04**
- [x] Phase 2 ベース + Loss を Soft CE に変更
- [x] 学習 NB: https://www.kaggle.com/code/maekeso/birdclef2026-exp011-train-phase3
- [x] 推論 NB: v3 (rglob fallback 入り) で LB 確定
- [x] **Phase 3 v3 LB = 0.824 (-0.030)** → 棄却。原因: raw mixup なしの CE は学習 softmax / 推論 sigmoid 分布乖離
- 教訓: `feedback_soft_ce_needs_raw_mixup.md`

### Phase 4 (+ raw waveform mixup) — **進行中 2026-05-04**
- [x] `_gen_nb1_train_p4.py` + `nb1_train_p4.ipynb` 作成 (13 cells)
- [x] `push_nb1_p4.py` 作成 (slug: birdclef2026-exp011-train-phase4-sed)
- [x] `_gen_nb2_submit_p4.py` + `nb2_submit_p4.ipynb` 作成 (9 cells)
- [x] `push_nb2_submit_p4.py` 作成 (slug: birdclef2026-exp011-submit-phase4-sed)
- [ ] Kaggle push → 学習実行 (T4x2, 期待 wall ~120-160 min)
- [ ] 推論 NB push → submit
- [ ] Phase 4 LB 取得 (目標 0.86-0.88)

**Phase 4 設計メモ**:
- mel cache 直読 → ogg 直読 + GPU MelSpectrogram (Phase 2 推論と同じパイプライン)
- spec mixup (fixed lambda=0.5) → raw waveform mixup (Beta(0.5, 0.5) lambda, label=max OR)
- BCE clip+frame 維持 (Soft CE 棄却教訓から)
- target_size (256, 256) 維持 (Phase 5 で 256x384 化)
- Val-B primary early stop (feedback_small_val_holdout.md 教訓から)
- num_workers=4 (ogg I/O bound、旧 Step 1 動作実績)

### Phase 5 (+ 256×384)
- [ ] target_size を (256, 384) に変更
- [ ] Phase 5 LB 取得

### Phase 6 (+ 過去年 pretrain)
- [ ] Stage A 学習データ統合スクリプト (BC25/21 + AnuraSet)
- [ ] Stage A → Stage B の checkpoint 引き継ぎ
- [ ] Phase 6 LB 取得 (目標 0.89-0.91)

### Phase 7 (+ 2025 公開 ckpt init)
- [ ] Kaggle Models / Datasets で 2025 ckpt 探索
- [ ] Phase 7 LB 取得 (見つからなければスキップ)

### Phase 8 (+ Multi-context head)
- [ ] 5/10/20s 並列 head 実装
- [ ] Phase 8 LB 取得

### Phase 9 (+ Pink Noise + Time Stretch)
- [ ] raw aug 追加
- [ ] Phase 9 LB 取得

### Phase 10 (+ iter pseudo R1)
- [ ] Pseudo label 生成スクリプト (Phase 9 weight で train_soundscapes 推論)
- [ ] Phase 10 LB 取得 (目標 0.92-0.94)

### Phase 11 (+ iter pseudo R2)
- [ ] Phase 11 LB 取得 (目標 0.93-0.94)

### Phase 12 (+ blend with NB3 + per-class threshold)
- [ ] 推論 NB に NB3 出力読み込み + rank average blend
- [ ] Per-class threshold OOF 最適化
- [ ] Phase 12 LB 取得 (目標 0.93-0.95)

### Phase 13 (+ multi-seed + isotonic calibration)
- [ ] Phase 11 を +2 seed 追加学習
- [ ] isotonic calibration
- [ ] Phase 13 LB 取得 (目標 0.94-0.95)

### 後回し (取り組まないかも)
- [ ] iNat full pretrain (旧 Stage 1, ~183種)
- [ ] Caiman yacare 収集
- [ ] B0_NS への切替実験 (`tf_efficientnet_b0.ns_jft_in1k`)
