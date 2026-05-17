# BirdCLEF 2026 改善バックログ

---

# 🚨 2026-05-17 21:45 Recon Update (CRITICAL)

## LB 現状 (top 30 取得済)

| 順位 | LB | 帯 |
|---|---|---|
| 1 | 0.962 | gold 確実 |
| 2-3 | 0.959 | gold 確実 |
| 4-7 | 0.957-0.958 | gold 圏内 |
| 8-11 | 0.956 | gold 圏内 |
| 12-15 | 0.954-0.955 | **gold border 推定** |
| 16-20 | 0.953 | silver 高位 |
| ~~ | ~~ | ~~ |
| **我々** | **0.947** | **silver 下位 → bronze 落ちリスク** |

→ gold = **+0.007-0.008 LB 必須**、silver 安泰 = +0.003-0.005、bronze 死守 = 現状維持

## 公開 NB 動向 (top 30 公開 kernels)

新しい trend が公開 NB に出てる:
- **0.948 帯**: ProtoSSM + SED ベース、forkers 多数 (youssefmo942009、mtoshidesu、複数)
- **`BirdNET Third Branch + Site-Hour Prior Restore`** (yaroslavkholmirzayev) — **我々が未着手の path、競合は実装中**
- **`Deep ProtoSSM & SED Dynamic Blend`** (raunakdey07 v6/v7) — blend 動的最適化
- **`EoS.3/EoS.4`** (nina2025) — 新規 stream、要分析
- **`Ensemble of solutions`** (anthonytherrien) — 複数解集約
- **`Acoustic Time-Window Rank Fusion`** (pilkwang) — 時間窓 ensemble
- **`Power optimization`** (karnakbaevarthur) — calibration

→ 競合 trend: **BirdNET、filename signals、dynamic blend 集中**。我々の Multi-teacher R3 は独自軸

## 🔥 緊急アクション (明日から)

### Submit-driven (毎日 5 枠)

1. **明日 #1**: exp020 R2 V5 standalone submit
2. **明日 #2**: exp027 ProtoSSM only submit
3. **明日 #3**: 公開 0.948 NB fork submit ← **silver 保険**として最優先 (前回 user 不要判断だったが、bronze 落ちリスク踏まえ再検討)
4. **明日 #4-5**: 結果次第 (R2 swap blend or 別軸)

### Multi-teacher R3 pseudo (進行中)

- [x] l0r2 ✅ COMPLETE
- [x] e17 ✅ COMPLETE
- [ ] hgnetv2-tucker (running V3、GPU)
- [ ] hgnetv2-r1 (running V3、GPU)
- [ ] exp015 convnext_pico (running)
- [ ] exp016 regnety_008 (running)
- [ ] **全完了後**: Multi-teacher 平均化 NB 作成
- [ ] **R3 学習 NB** (Colab Blackwell、4-6h)

## 🎯 Gold 達成 3 軸 (全実行が必要)

### 軸 A: Multi-teacher R3 pseudo (進行中) → +0.003-0.006

### 軸 B: 独自 data 軸
- [ ] **day-of-year signal** (NB4 内追加、~+0.003-0.008 NB4 単独)
- [ ] **session smoothing** (同 site/hour 連続 file)
- [ ] **wet/dry class-level** (species-level 失敗確認済、class-level retry)

### 軸 C: 公開 NB 流用 / 取り込み
- [ ] **0.948 NB fork submit** (silver 保険)
- [ ] **BirdNET 3rd 軸 integration** (Tucker と重複懸念だが、競合が試してる)
- [ ] **EoS.4 / Dynamic Blend / Ensemble of solutions** の Cell 解析 (新規 trick 探索)

## 📊 更新版 Gold 確率

| 構成 | gold 確率 |
|---|---|
| 現状 0.947 (何もしない) | **0%、bronze 落ちリスク** |
| + exp020 R2 swap (R2 V5 OK 前提) | 10-20% |
| + Multi-teacher R3 | 35-50% |
| + day-of-year + Sonotype 拡張 | 50-60% |
| **+ BirdNET 3rd 軸 (0.948 base)** | **60-75%** |
| + AVES paradigm 軸 | 70-80% |

## 重要な判断点

1. **公開 0.948 fork は silver 保険として実行必須** (前回 user 「不要」判断、recon 後 reconsider 推奨)
2. **BirdNET 3rd 軸は競合がやってる**、我々も追随必要
3. **2.5 週間で 3 軸全部実行**するには 1 軸/数日のペース
4. exp020 R2 V5 結果が明日の方向性決定の鍵

---


>
> **メダル境界 (user 確認)**: 銅 0.944 / 銀 0.945 / 金 ~0.952+ (推定)
> 銅まで **+0.005**、銀まで **+0.006**、金まで **+0.013+**
>
> **進行中**:
> - **AVES embedding 抽出 NB1f**: T4 GPU で実行中 (3-5h)、3rd blend axis 候補
> - **BirdNET embedding 抽出 NB1g**: T4 GPU で実行中 (2-3h)
> - **CLAP NB1h**: GPU quota reset 後 (5/9) push 予定
> - **exp012 自前 5-fold (Colab)**: 優先度低下 (Tucker public で代替可能)
>
> **Gold までの公式** (公開 NB の分解から):
> ```
> 0.939 帯 = NB4 v7 (Perch+ProtoSSM+MLP) + Tucker public 5-fold SED       ← 我々の現状
> 0.943 帯 = ↑ + train_audio LinearHead (konbu17 公開重み流用)             ← 即実装可
> 0.946 帯 = ↑ + G26 Per-class Isotonic + G27 Cross-branch gate           ← 公開 NB 0.946 由来、Day 1
> 0.948 帯 = ↑ + AVES 3rd axis blend (Spearman 0.417 で独立性確認済)       ← Week 1
> 0.952 帯 = ↑ + pseudo-label R1 (F23 site-stratified + H1 Power Scaling p=1.82) ← Week 2
> 0.955 帯 = ↑ + pseudo-label R2-R3 (F25 curriculum + H1 Power Scaling)         ← Week 2-3
> 0.957 帯 = ↑ + H12 Auxiliary 700-class head (1位 Babych の +0.003 実測)        ← Week 3+
> ```

---

## 凡例

- **実装コスト**: 小=半日以下 / 中=1-2日 / 大=3日以上
- **処理時間**: 推論時の追加時間（CPU 90分制約への影響）。"なし"=学習時のみ
- **Score Gap**: 実測 LB 差分 / 期待値 (未実装は期待値のみ)
- **Status**:
  - ✅ **採用済** (実装 + LB 改善確認、現行 baseline に組込)
  - 🟢 **試行中** (push 済 / 学習中、LB 結果待ち)
  - ⚠️ **試行→却下** (LB 悪化 or noise 圏 → revert)
  - ❌ **未実装**
  - 🔵 **新規候補** (EDA / Discussion から導出、未試行)

---

## 主要 LB ログ

### exp010 NB3 (Perch + ProtoSSM submission)

| Version | 追加要素 | LB | Score Gap | Status |
|---|---|---|---|---|
| v6 | baseline (Perch + ProtoSSM + 66 labeled SS) | 0.884 | - | ✅ |
| v7 | site/hour emb + TTA shifts=[-1,0,1] | 0.886 | +0.002 | ✅ |
| v8 | Prior Tables (site/hour shrinkage λ=0.3) | 0.905 | **+0.019** | ✅ |
| v11 | Month embedding | -0.017 vs v8 | -0.017 | ⚠️ revert (OOD `nn.Embedding`) |
| v17/18 | KMeans+PCA cluster + Month 同時 | -0.007 | -0.007 | ⚠️ revert (1 push 1 変更違反) |
| **v20** | file_confidence_scale top-K=2 power=0.4 | **0.914** | **+0.009 vs v8** | ✅ clean baseline |
| v21 | Adaptive δ-shift smoothing | 0.915 | +0.001 noise | ⚠️ revert |
| v22 | Class-specific T (Aves 1.10 / Amph 0.95) | 0.914 | ±0 | ⚠️ revert |
| v23 | ResidualSSM 2nd-pass (α=0.35) | 0.867 | **-0.047** | ⚠️ revert |
| v24 | MixUp on Perch embedding | 0.911 | -0.003 | ⚠️ revert |
| v25 | Iter pseudo R1 | 0.880 | **-0.034** | ⚠️ revert |
| v26 | LightProtoSSM cross-attn | 0.921 | +0.007 | ✅ |
| v29 | retrieval (TTA 5-shift) | **0.921** | - | ✅ 現行 |

### exp010 NB4 (MLP + ProtoSSM blend)

| Version | 追加要素 | LB | Score Gap | Status |
|---|---|---|---|---|
| v1 | baseline | 0.923 | - | ✅ |
| v2 | SWA 単独 | 0.923 | ±0 | ⚠️ multi-seed と機能重複 |
| v3 | Focal γ=2.0 | 0.920 | -0.003 | ⚠️ pos_weight と干渉 |
| v5 | SWA + KD λ=0.15 | 0.923 | ±0 | ⚠️ KD 効果なし |
| **v7** | TA retrieval pool LAMBDA=0.05 | **0.924** | **+0.001** | ✅ 現行 baseline |
| v8 | LAMBDA=0.10 | 0.922 | -0.002 | ⚠️ 過剰 |
| v9 | + AnuraSet/iNat 外部 pool | 0.924 | ±0 | ⚠️ 上乗せ無し |
| v10 | + class-specific LAMBDA | 0.924 | ±0 | ⚠️ 上乗せ無し |
| **v11** | + E19 file-level consistency boost (max β=0.2) | **0.926** | **+0.002** | ✅ 単独 NB に効く (blend では効果消失) |

### exp010 NB5/NB6 (Noisy Student) — 失敗確定

| Version | 内容 | LB | Status |
|---|---|---|---|
| NB5 | 5 ラウンド学習 (66 SS labeled + 10,592 pseudo SS) | - | ✅ 完走 |
| NB6 v1 | NB5 重み load + 推論 | 0.923 | ⚠️ -0.001 vs NB4 v7、自己蒸留の天井 |

### exp010 NB7/NB8 (train_audio MLP pretrain) — 失敗確定

| Version | 内容 | LB | Status |
|---|---|---|---|
| NB7 v1 | focal 265k で MLP pretrain (primary one-hot) | - | ✅ 完走 |
| NB8 v1 | NB7 v1 重み + 推論 | 0.914 | ⚠️ -0.010 vs NB4 v7、focal vs SS ドメインギャップ |
| NB7 v2 | + multi-hot (primary+secondary) | - | ✅ |
| NB8 v2 | NB7 v2 重み + 推論 | 0.916 | ⚠️ -0.008 |
| NB7 v3 | + Perch sigmoid soft target | - | ✅ |
| NB8 v3 | NB7 v3 重み + 推論 | (未実測 / 路線 abandon) | ⚠️ |

### exp012 (Tucker Distilled SED 自前再現)

| Step | 内容 | LB / 状態 | Score Gap | Status |
|---|---|---|---|---|
| 1 | Tucker single fold (Kaggle T4) | 0.890 | - | ✅ |
| 2 | Colab Pro A100 で 5-fold 学習 | - | - | 🟢 進行中 |
| 3 | 5-fold ensemble + Gaussian smooth | - | +0.02-0.03 | ❌ |
| 4 | 20s window (Salman 流) | - | +0.005 | ❌ |
| 5 | wave mixup 強化 (Boredom 流) | - | +0.003 | ❌ |
| 6 | unlabeled SS pseudo-label R1 | - | **+0.012** | ❌ |
| 7 | gate-fake008 と blend | - | **0.95+ 期待** | ❌ |

### Blend 系 (2026-05-08 更新)

| 構成 | LB | Score Gap | Status |
|---|---|---|---|
| NB4 v7 単独 | 0.924 | - | ✅ |
| NB4 v7 × exp012 fold0 (70:30) | 0.929 | +0.005 | ✅ |
| **NB4 v7 × Tucker public 5-fold ONNX (50:50)** | **0.939** | **+0.015** | ✅ **現行最高** |
| NB4 v7 × Tucker (40:60) | 0.937 | -0.002 | ⚠️ 50:50 が ratio 最適 |
| NB4 v11 (E19) × Tucker (50:50) | 0.939 | ±0 | ⚠️ E19 は Tucker smoothing と機能重複 |

---

## 🥇 Medal Path (2026-05-08 更新, 現状 0.939)

### Phase 1: 銅 (0.944) → 銀 (0.945)、本日〜5/15

| 優先 | 仮説 | 期待 LB | Status | コスト |
|---|---|---|---|---|
| 🥇 | **train_audio LinearHead** (konbu17 公開重み流用) | +0.002-0.003 | ❌ | 小 (1h) |
| 🥇 | **AVES 3rd axis blend** (Spearman 0.417 確認済) | +0.005-0.010 | 🟢 抽出 NB push 済 | 中 (2-3日) |
| 🥈 | **Max Ensemble** (mean→max 集約) | +0.000-0.005 (再評価、BACKLOG 楽観値の半分) | ❌ | 極小 (5分) |
| 🥈 | **A3 Retrieval pool dedup** (TA pool 同 author×種 削減) | +0.002-0.005 | ❌ | 小 (30分) |
| 🥉 | **E20 Session smoothing** ((site, date) 連続性) | +0.001-0.003 | ❌ | 小 (1h) |
| ❌ | A1 家畜 hard-zero | +0.000-0.001 | EDA 検証で多くが false positive 候補、効果小 | 5分 |

### Phase 2: 銀 (0.945) → 金 (0.952+)、5/15〜5/29

| 優先 | 仮説 | 期待 LB | Status | コスト |
|---|---|---|---|---|
| 🥇 | **F23 Site-stratified pseudo R1 + H1 Power Scaling (p=1.82)** (Sites 01/02/13 = SS の 63% / labeled ゼロ。1位 Babych の決定打 = Power Scaling 必須セット) | +0.008-0.015 | ❌ | 中 (2日) |
| 🥇 | **F25 Curriculum pseudo R2-R3 + H1 Power Scaling** (5位 Noir 3-stage と整合) | +0.010-0.020 | ❌ | 大 (3日) |
| 🥈 | **H12 Auxiliary 700-class head** (1位 Babych 0.930→0.933 実測、公開 inference NB に実装あり) | +0.002-0.005 | 🔵 | 中 |
| 🥈 | **H3 SED head fine-tuning (backbone 凍結)** for 非鳥類 73 種 | +0.002-0.005 | 🔵 | 小 (半日) |
| 🥈 | **BirdNET 4th axis** (gap 0.836、独立 0.660) | +0.002-0.005 | 🟢 抽出 NB push 済 | 中 |
| 🥇 | **E21/G26 Per-class Isotonic + F1 threshold** (公開 0.946 NB に具体実装あり、コピペ可) | +0.004-0.008 | 🔵 | 小 (1-2h) |
| 🥇 | **G27 Cross-branch agreement gate** (fake_only/proto_cont/sed_only) | +0.002-0.005 | 🔵 | 小 (1h) |
| 🥈 | **G28 Student-t temporal kernel** (df=1.5, fat-tail) | +0.001-0.003 | 🔵 | 小 (30分) |
| 🥈 | **E22 Test soundscape clustering** | +0.003-0.008 | ❌ | 中 (1日) |
| 🥉 | **C13 Negative mining "never-in-SS" 159種** | +0.005-0.010 | ❌ | 中 (半日) |
| ⭐ | **D15 Site classifier auxiliary head** | +0.005 | ❌ | 中 (1日) |
| ⭐ | **D18 Cross-domain contrastive learning** | +0.007 | ❌ | 大 (2日) |

### 過去候補で却下/効果小

| 仮説 | 結果 | 教訓 |
|---|---|---|
| ⚠️ Max Ensemble (BACKLOG 旧見積 +0.010-0.020) | 期待値を下方修正 | 公開 NB top帯では mean blend が defacto |
| ⚠️ A1 家畜 hard-zero (旧 +0.001-0.003) | EDA で houspa/osprey/bbwduc など Pantanal 出現あり判明 | 効果小、リスク中 |
| ⚠️ B6 セミドローン合成 | 学習が必要、ROI 中 | 後回し |
| ⚠️ C11 Per-class loss weight | exp010 構造では学習データ 66 SS のみ | F23/F25 の方が ROI 高 |

---

## Tier 1: 最優先（銅〜銀メダル必達施策）

| 手法 | 詳細 | LB / Score Gap | Status | 実装コスト |
|---|---|---|---|---|
| **推論後処理3点セット** | Rank-aware power scaling + Delta shift smoothing + Per-class threshold (OOF) | v20=0.914 (+0.009), v21=+0.001 noise, v22=±0 | ✅ file_conf v20 / ⚠️ δ smooth v21 / ⚠️ class T v22 / ❌ per-class threshold | 小 |
| **複数ラウンド疑似ラベル** | NB3 v25 R1=-0.034 (構造的ミスマッチ)、NB5/6 自己蒸留=-0.001 (66 SS teacher 弱い)。**SED 系 + site-stratified が本命**。**1位 Babych = 4 ラウンド + Power Scaling (H1) でノイズ抑制が決定打**、5位 Noir = 3-stage self-distillation で同様の効果 | v25=-0.034 / NB6=-0.001 / 期待 +0.012-0.020 (F23/F25) | ⚠️ NB3/Self-distill 却下 → ❌ SED 系で必須 | 中-大 |
| **SoftAUCLoss** | AUC 直接最適化、BCE+SoftAUC 混合。**9位 "finally not overfitting" 採用 (overfitting 耐性、soft label 対応)**。pairwise 差分 + log-loss 形式 | 期待 +0.003-0.007 (旧 +0.005-0.010 から下方修正) | ❌ | 小 |
| **OpenVINO/ONNX 推論最適化** | Perch ONNX、`intra_op_num_threads=4` + `ThreadPoolExecutor` | -30~40% 推論時間 | ✅ NB2/NB3 v9/v6 以降 | 中 |
| **Perch emb + メタクラスタリング** | Tucker Arrants 0.925 解釈 (KNN だった、KMeans でない) | 期待 +0.005-0.010 (旧 +0.010-0.025 から下方) | ❌ (現状 NB4 v7 retrieval で部分実現) | 中 |
| **SED (CNN) 系 branch 追加** | Tucker public 5-fold ONNX 採用済 | **+0.015 実測 (NB4 単独 0.924 → blend 0.939)** | ✅ Tucker public 経由で実現 (exp012 自前は不要) | 完了 |
| **🆕 train_audio LinearHead 公開重み (konbu17)** | `konbu17/bird26-train-audio-head-v1` を blend NB に追加 | 期待 +0.002-0.003 (公開 0.943 NB で実証) | ❌ | 小 (1h) |
| **🆕 AVES 3rd axis blend** | wav2vec2 系列 embedding (Spearman 0.417, gap 0.785 で独立確認) | 期待 +0.005-0.010 | 🟢 抽出 NB push 済 | 中 (2-3日) |

---

## Tier 2: 重要（モデル・パイプライン改善）

| 手法 | 詳細 | LB / Score Gap | Status | 実装コスト |
|---|---|---|---|---|
| ResidualSSM 2nd-pass | BiSSM(d=64) + α=0.35 zero-init | -0.047 | ⚠️ revert (`feedback_residualssm_overfit.md`) | 中 |
| Cross-Attention (Perch↔SSM) | NB3 v26 LightProtoSSM | +0.007 vs v20 | ✅ v26 採用 | 中 |
| **Multi-seed 5モデル ensemble** | seed=[42,123,777,2024,9999] | NB3 v8 以降 累積貢献 | ✅ 採用 | 小 |
| Month metadata embedding | OOD `nn.Embedding` 危険 | -0.017 | ⚠️ revert | 小 |
| 2本目 backbone 蒸留 | AudioMAE/BEATs と Perch 並列 | 期待 +0.005-0.010 (下方修正、Tucker public 採用済で類似効果) | ❌ | 大 |
| **TTA 3-shift** | `TTA_SHIFTS=[-1,0,1]` | NB3 v8 以降貢献 | ✅ 採用 | 小 |
| TTA 2.5秒 half-shift | sub-window TTA。**1位 Babych 実測 0.91→0.922 (+0.011)** で確度高 | 実測 +0.011 (1位)、期待 +0.005-0.010 | ❌ | 小 |
| TTA 5-shift (circular) | shifts=[0,1,-1,2,-2] roll/逆roll/avg | NB3 v27 同値 | ⚠️ ±0 |  小 |
| 昆虫・両生類専用モデル | Non-bird 73種に EffNet-B0 専用 (1位 Babych 0.930→0.933 = +0.003 実測)。**公開 inference NB に 700-class head の具体実装あり** (H12 参照、`multilabel_pred_to_train_preds` で 234 種空間に index remap) | 期待 +0.002-0.005 (実測あり、Pantanal Insecta 25 sonotype は外部に無く再学習対象限定的) | ❌ | 中-大 |
| BirdNET アンサンブル | BirdNET ONNX + Perch blend (Spearman 0.660 中独立、gap 0.836) | 期待 +0.002-0.005 (再評価、AVES より priority 低) | 🟢 抽出 NB push 済 | 中 |
| UDA (DANN/MMD) | train_audio ↔ SS 分布整合 | 期待 +0.005-0.015 | ❌ | 大 |
| SpecAugment 強化 | freq/time_mask 30→40-50 | 期待 +0.005 | ❌ | 小 |
| 正則化強化 | Dropout 0.2-0.3, DropPath 0.1-0.2 | 期待 +0.005 | ❌ | 小 |
| **Perch TFLite 変換** | TF→TFLite で Perch 60% 短縮 | -60% Perch 推論 | ❌ (現状 ONNX で十分) | 小 |
| train_audio Perch 埋め込み | 46k 音声で emb 抽出 + LinearHead | 期待 +0.002-0.003 (konbu17) | ❌ konbu17 公開重み流用が ROI 高 | 中 |
| **長窓入力 (10-20s)** | Salman 20s で 0.937 | 期待 +0.005-0.015 | 🟢 exp012 step 4 計画 | 中 |
| **Raw wave Mixup** | spec mixup より優 | exp012 で実装済 | ✅ exp012 採用 | 小 |
| **CE loss + Sigmoid inference** | Salman 0.922 | exp012 で実装済 | ✅ exp012 採用 | 小 |
| **KLD 蒸留 10ep → FT 12ep** | EliKal 0.906-0.909 | exp012 で実装済 (Perch MSE 蒸留) | ✅ exp012 採用 | 中 |
| **Multi-context head** | hengck23: 異なる窓長 head 並列 | 期待 +0.005-0.010 | ❌ | 中 |
| logit std 監視ロギング | overfit 検出 | 間接 | ❌ | 小 |
| **Sonotype mirroring** | needless090: MIRROR_PAIRS 内 max 伝播 | 期待 +0.001 | ❌ | 小 |
| **2025 pretrained checkpoint 転移** | Ali Ozan: VSydorskyy NFNet/EfficientNetV2-S +0.01 | 期待 +0.01 | ❌ (アーキ変更必要、見送り中) | 大 |

### Tier 2 追加候補 (DCASE Task 5 発、再評価)

| 手法 | 詳細 | Score Gap | Status | 実装コスト |
|---|---|---|---|---|
| Supervised Contrastive Learning (SupCon) | Moummad DCASE2023 2位 | 期待 +0.003-0.008 (下方) | ❌ | 小〜中 |
| **AVES wav2vec2 backbone 並列** | Bordoux DCASE2024 7位、EDA で Spearman 0.417 / gap 0.785 で実用性確認 | **期待 +0.005-0.010 (実測根拠あり)** | 🟢 抽出 NB push 済 | 中 |
| TS-VAD 流用 multi-task | Du DCASE2023 1位 | 期待 +0.005-0.010 (下方) | ❌ | 大 |
| PCEN + ΔMFCC 特徴追加 | QianHu系 SED 補強 | 期待 +0.001-0.003 (下方、Tucker SED で代替) | ❌ | 小 |
| IFPDA Domain Adaptation | Latifi DCASE2024 | 期待 +0.005-0.015 | ❌ | 大 |
| Template Matching + DTW | Wilkinghoff DCASE2023 稀種対策 | 期待 +0.001-0.003 (下方、retrieval で部分代替) | ❌ | 中 |
| BEATs transformer embeddings | Gelderblom DCASE2023 | 期待 +0.003-0.008 (再評価) | ❌ | 中 |

---

## Tier 3: 小改善・検証候補 (再評価)

| 手法 | 詳細 | Score Gap | Status | 実装コスト |
|---|---|---|---|---|
| Silero-VAD 人声除去 | train_audio 前処理 | 期待 +0.001-0.003 (下方) | ❌ | 小 |
| Max Ensemble | mean → max 集約 | 期待 +0.000-0.005 (大幅下方修正、公開 NB top では不採用) | ❌ | 小 |
| Secondary labels 活用 | train.csv secondary を MLP 特徴量化、NB7 v2 で hard label に追加するも -0.008 | 旧期待 +0.010 → 実測効果薄 | ⚠️ NB7 v2 で実測 | 中 |
| Focal Loss (NB3) | BCE → Focal Loss | NB4 v3 -0.003 | ⚠️ revert (BCE+pos_weight 環境では悪化) | 小 |
| Checkpoint Soup / SWA | epoch 重み平均 | NB4 v2 ±0 | ⚠️ multi-seed=5 と重複 | 小 |
| **データ品質フィルタ** | RMS/STD/パワーで低品質除外 | 期待 +0.001-0.003 (下方) | ❌ | 小 |
| HGNetV2-B0 バックボーン | timm 実装済、GhostConv で FLOPs 半減 | SED 性能依存 | ❌ (Tucker public で代替) | 小 |
| 入力長 7s/10s 化 | OpPrime 20s 報告 | 要検証 | ❌ | 中 |

---

## Tier 4: パラメータチューニング (構造変更後)

| 手法 | 詳細 | Score Gap | Status | 実装コスト |
|---|---|---|---|---|
| Power Scaling (inference) | `p ** power` grid=[0.5, 0.7, 1.0, 1.5, 2.0] を**推論出力**に適用。**理論上 AUC は rank 不変で ±0**、blend/smoothing との二次効果のみ | 試行済、有意な改善なし。training-time pseudo-label target への適用 (H1) とは別物 | ⚠️ 試行→却下 | 小 |
| Temperature grid | T=[0.8, 0.9, 1.0, 1.1, 1.2, 1.5] | NB3 v22 ±0 | ⚠️ 効果薄 | 小 |
| アンサンブル比率調整 (Perch×Tucker) | grid (40:60-60:40) | 50:50 で最適 (40:60=-0.002 確認) | ✅ blend 50:50 採用 | 小 |
| カーネル平滑化 | 5-tap [0.1,0.2,0.4,0.2,0.1] / gaussian σ=0.65 | exp012/Tucker public で採用 | ✅ Tucker 標準 | 小 |
| Quantile-Mix | mean + rank α=0.5 | 期待 +0.001-0.003 (下方) | ❌ | 小 |
| Prior Tables λ tune | grid `λ=0.3, strength=(8,8,4)` | NB3 v8 で +0.019 | ✅ 現行値で最適 | 小 |

---

## EDA 駆動の新仮説 (2026-05-08 追加)

> Discussion / 公開 NB に未出の 25 仮説。データ構造解析から導出。
>
> **重要な EDA 発見** (詳細は memory `project_eda_findings.md` 参照):
> - テスト soundscape の **99.2% が夜間録音 (18:00-06:00)** — 昼間 88 件 (0.8%) のみ
> - train_audio の **4.2% のみが Pantanal 1000km 圏内**、16.1% は 5,000km 以上 (Gallus 17,125km、Pandion 7,339km)
> - **Sites 01/02/13 は SS の 63% を占めるが labeled SS ゼロ** (Site 22 が labeled の 60.6%)
> - labeled SS 上位 11 種が全て Amphibia/Insecta、4 種は train_audio ゼロ (517063=626 win/0 TA、47158son25=168/0、1491113=158/0)
> - Insecta 28 種中 train_audio あるのは 3 種のみ
> - **隣接 window の Jaccard 類似度 0.918** = 同一ファイル内 species 構成は静的
> - 同一 author × 同一種で 10件以上が **415 組** (重複バイアス)
> - iNat ID の **28.6% が ±10 以内に連続** (同一フィールドセッション)

### A. データ品質・選別 (再評価)

| ID | 仮説 | 根拠 | Score Gap | Status | 実装コスト |
|---|---|---|---|---|---|
| **A1** | 家畜・遠方種 hard-zero prior | EDA 検証: 距離 > 5000km 14 種のうち多くが Pantanal 出現あり (houspa, osprey, bbwduc, bkhpar 等) | **+0.000-0.001 (下方修正、リスク中)** | 🔵 | 5分 |
| **A2** | iNat 短ファイル除外 (dur<1s) | 0.036s 等のゴミファイル混入 (3%程度) | +0.001-0.002 | 🔵 | 5分 |
| **A3** | 同一 author×同一種 重複削減 (推論時 retrieval pool filter) | 415組で 10件+、TA pool に直接適用可 | +0.002-0.005 | 🔵 | 30分 |
| **A4** | iNat 連続 ID で session group fold | 28.6% が同セッション、OOF 用 | CV 信頼度向上 | 🔵 | 1時間 |
| **A5** | Pantanal 1000km 圏内 fine-tune | 4.2% が test domain に近い | +0.003-0.008 (下方) | 🔵 | 学習+30% |

### B. Augmentation

| ID | 仮説 | 根拠 | Score Gap | Status | 実装コスト |
|---|---|---|---|---|---|
| **B6** | セミドローン合成背景ノイズ | 全 SS が Quesada gigas ドローン支配 | +0.005-0.012 | 🔵 | 1日 |
| **B7** | 遠距離シミュレーション (reverb+LPF) | focal 近距離 vs SS 遠距離 | +0.003-0.008 | 🔵 | 半日 |
| **B8** | 時間反転 chorus 合成 | labeled SS 660 windows (45%) が 5+種同時 | +0.002-0.005 | 🔵 | 半日 |
| **B9** | Type 別 augmentation | song vs call 最適 aug 異なる | +0.001-0.003 | 🔵 | 半日 |
| **B10** | Frequency-band masking | taxa 別に周波数帯異なる | +0.002-0.005 | 🔵 | 1時間 |

### C. Loss 関数

| ID | 仮説 | 根拠 | Score Gap | Status | 実装コスト |
|---|---|---|---|---|---|
| **C11** | Per-class loss weight by SS freq | 517063 = SS 626/TA 0、BCE 学習信号皆無 | +0.003-0.008 | 🔵 | 1時間 |
| **C12** | Distance-weighted loss | Pantanal 近距離録音優先 | +0.002-0.005 | 🔵 | 1時間 |
| **C13** | Negative mining "never-in-SS" 159種 | val SS 75種のみ、159種は壊滅でも気付けない | +0.005-0.010 | 🔵 | 半日 |
| **C14** | Calibration loss (KL to SS prior) | 出力分布が SS prior と乖離 | +0.001-0.003 | 🔵 | 半日 |

### D. アーキテクチャ・補助タスク

| ID | 仮説 | 根拠 | Score Gap | Status | 実装コスト |
|---|---|---|---|---|---|
| **D15** | Site classifier auxiliary head | 全 SS に site ラベル無料、Insecta sonotype が site-specific | +0.005-0.010 | 🔵 | 1日 |
| **D16** | Multi-context head (5/10/20s) | 短 call と長 song で最適 context 異なる | +0.003-0.008 | 🔵 | 2日 |
| **D17** | Chorus / Quiet binary detector | SS labels 45% が chorus、11% が quiet | +0.002-0.005 | 🔵 | 半日 |
| **D18** | Cross-domain contrastive learning | focal と pseudo SS の同種 pair 対照 | +0.005-0.010 | 🔵 | 2日 |

### E. Inference / Post-Processing (再評価)

| ID | 仮説 | 根拠 | Score Gap | Status | 実装コスト |
|---|---|---|---|---|---|
| **E19** | File-level species consistency boost (max β=0.2) | 隣接 window Jaccard 0.92 | NB4 v11 単独 +0.002 / blend ±0 (Tucker と機能重複) | ⚠️ 単独 NB のみ採用 | 30分 |
| **E20** | Session-level smoothing | (site, date) セッション ~4.4h 連続 | +0.001-0.003 | 🔵 | 1時間 |
| **E21** | Adaptive per-class threshold via OOF | needless090 0.934 で実現 | +0.003-0.007 | 🔵 (OOF 必須) | 半日 |
| **E22** | Soundscape clustering @ inference | 同 site/時刻/月の SS は同 species 構成 | +0.003-0.008 | 🔵 | 1日 |

### F. Pseudo-label 詳細

| ID | 仮説 | 根拠 | Score Gap | Status | 実装コスト |
|---|---|---|---|---|---|
| **F23** ⭐ | Site-stratified pseudo-labeling | Sites 01/02/13 = 6,719 unlabeled / labeled ゼロ | +0.008-0.015 | 🔵 | 2日 |
| **F24** | Multi-teacher pseudo-label | Tucker 5-fold + Perch 異質系統で robust | +0.003-0.008 | 🔵 | 1日 |
| **F25** ⭐ | Curriculum pseudo-labeling | Round 1: TA>=50 種 → R2: TA>=10 → R3: 全種 | +0.010-0.020 | 🔵 | 3日 |

---

## 🆕 0.946 NB 解析所見 (yaroslavkholmirzayev/0-946-replay-with-robust-inputs, 2026-05-09)

> exp_043 (0.941) → exp_044 (0.944) → **exp_044c (0.946)** の ablation chain。
> Vyanktesh Dwivedi の base に乗せた public 限定の comprehensive recipe。

### 主要新規要素 (公開 NB 標準を超える部分)

| ID | 要素 | 期待 LB | Status | 実装コスト |
|---|---|---|---|---|
| **G26** ⭐ | **Per-class Isotonic + F1 threshold sharpening** (E21 の具体実装が公開) | +0.004-0.008 (本人記載) | 🔵 公開実装あり、コピペ可 | 小 (1-2h) |
| **G27** ⭐ | **Cross-branch agreement gate** (fake_only / proto_cont / sed_only) | +0.002-0.005 | 🔵 | 小 (1h) |
| **G28** | **Student-t kernel temporal smoothing** (df=1.5, scale=1.20, fat-tail) | +0.001-0.003 | 🔵 Tucker Gaussian の代替 | 小 (30分) |
| **G29** | **Rare-taxon suppression を入れない** (V8 が `vals<mean+0.05 → ×0.9` で Amph/Mam/Rep を抑制 → 解除で +0.002) | +0.002 (実測 0.944→0.946) | ✅ 既に未実装、確認のみ | - |
| **G30** | **MLP probe upgrade** (PCA 1536→64, hidden (128,64), min_pos=8→5, alpha=0.4) | +0.003-0.006 | 🔵 exp010 系の MLP 改修 | 中 (半日) |

### G27 cross-branch gate 具体定義 (実装用メモ)
```python
fake_only  = (Proto > 0.50) & (SED < 0.05)                              → Proto rank ×1.08
proto_cont = (smoothed_Proto rank > 0.88) & (Proto > 0.75) & (SED < 0.12) → Proto rank ×1.15
sed_only   = (SED rank > 0.95) & (Proto < 0.80)                          → SED rank ×1.12
```
Gate kernel: Student-t `(1 + (offs/1.20)^2 / 2)^(-1.5)` over offs=[-3..3]

### G26 per-class threshold 具体手順
1. OOF 出力に対し各 class で `IsotonicRegression` fit
2. 各 class で grid `[0.25, 0.30, ..., 0.70]` から F1 最適 threshold を選択
3. 推論時: 上回る → 1 寄り (sharpen)、下回る → 0 寄り (suppress)

### Phase 1 への組込み優先度 (現状 0.939 → 銀 0.945 まで +0.006)
- G26 (+0.004-0.008) + G27 (+0.002-0.005) + G28 (+0.001-0.003) で **+0.007-0.016** → 銀〜金射程に乗る
- いずれも公開 NB 内に実装あるため工数 1 日以内、銀越え最速ルート

---

## 🆕 BirdCLEF 2025 上位 13 解法の学び (2026-05-09)

> 1-14 位 (8位欠番) の write-up を横断調査。Kaggle write-up は JS レンダリング必須で直接 fetch 不可なため、CEUR-WS Vol-4038 paper_256 (2位 Sydorskyy)、各位 GitHub README、tekkix.com 上位5解説、SpeakerDeck 25位 Ryushi 比較考察、YouTube Walkthrough を統合した結果。

### 共通パターン (上位 5 位以上が全員実施)

1. **EfficientNet 系 + SED head が defacto** — V2-S/B3/B0、NFNet-L0 が多数派。**Transformer (BEATs/AST) は上位非採用** (OpenVINO 互換性悪)
2. **疑似ラベル多ラウンド反復** — 名前違うが本質同じ (Noisy Student / Self-distillation / Semi-supervised distillation)、3-4 ラウンド
3. **外部データ補完** — Xeno-Canto, iNaturalist, CSA。**非鳥類クラスの補強が決定打**
4. **OpenVINO fp16 推論最適化** — CPU 90 分制約のため必須、ONNX → IR 形式
5. **Temporal smoothing** — `[0.1, 0.2, 0.4, 0.2, 0.1]` カーネル defacto

### ユニーク要素 (1 人が採用して効いた手法)

| ID | 要素 | 出典 | 期待 LB | Status | 実装コスト |
|---|---|---|---|---|---|
| **H1** ⭐ | **Power Scaling on pseudo-label TARGETS during training** (`target = pred ** 1.82`) — pseudo-label の noisy floor を強烈に削る label-noise filter。**重要: inference 後処理ではなく学習時の target 加工**。inference 時の `pred ** k` は AUC rank 不変で効かない (我々が複数回試行して空振り、Tier 4 参照)。**前提条件**: ① 自前 SED 学習ループ ② 多ラウンド iterative ③ 信頼度閾値併用 | 1位 Babych | +0.003-0.008 (training-time 適用時のみ、inference では ±0) | 🔵 F25/F23 自前 SED 学習と必須セット (Tucker public 経由では適用不可) | 小 (30分、ただし F23/F25 学習基盤が前提) |
| **H3** | **SED head fine-tuning (backbone 凍結)** for レアクラス | 12位 223223223 | +0.002-0.005 | 🔵 非鳥類 73 種への適用候補 | 小 (半日) |
| **H4** | **20-fold CV for model selection** (通常 5-fold) | 2位 Sydorskyy (CEUR-WS) | CV 信頼度向上 | 🔵 OOF 構築時に検討 | - (CV 設計) |
| **H5** | **Manual audio review (<30 サンプル種)** で品質管理 | 1位 Babych | 学習データ品質 | 🔵 Pantanal Insecta 25 sonotype 候補 | 中 (1日) |
| **H6** | **長ウィンドウ for レア種** (通常 5s、レアは最大 60s) | 1位 Babych | +0.001-0.003 | 🔵 学習時のみ | 中 |
| **H8** | **ImageNet-21k pretrain init** (V2-S/B3) | 2位 Sydorskyy | Aves 学習底上げ | 🔵 学習時 init 変更のみ | 小 |
| **H9** | **SqrtBalancing + MinorOverSampleV1** クラスバランス | 2位 Sydorskyy | rare 種 AUC 改善 | 🔵 | 小 |
| **H12** | **Auxiliary 700-class head** (Insecta/Amphibia 700 種で B0 学習 → 234 種空間に index remap、`multilabel_pred_to_train_preds` で合流) | 1位 Babych (公開 inference NB `nikitababich/birdclef2025-1st-place-inference` に実装あり) | +0.002-0.005 (実測根拠あり) | 🔵 既存 "昆虫・両生類専用モデル" の具体 recipe | 中 |

### 失敗報告 (避けるべき)

- ❌ ナイーブな複数ラウンド pseudo-label (品質管理なし) — 25位 Ryushi 報告。**Power Scaling (H1) や信頼度閾値が必須**
- ❌ 学習 duration 延長 — 過学習リスク (25位)
- ❌ Transformer backbone (BEATs/AST 等) — OpenVINO 変換の互換性悪、推論遅
- ❌ Soft CE 単独 (raw mixup なし) — memory `feedback_soft_ce_needs_raw_mixup.md` と整合

### 我々の現状 (LB 0.939) への適用優先度

**Phase 1 (銀 0.945 まで +0.006)**
- **H1 Power Scaling** (30分): F23 site-stratified pseudo + 必須セットとして組込
- **既存 G26/G27/G28 (公開 0.946 NB 由来)** が Phase 1 の本命で、H 系は補強

**Phase 2 (金 0.952+ まで +0.013)**
- **H12 Auxiliary 700-class head**: 公開 inference NB に実装あり、recipe 抽出済 (再学習データ収集が課題、Pantanal Insecta 25 sonotype は外部に無く対象限定)
- **H7 (= F25 と統合) Self-distillation 3-stage**: F25 curriculum と本質同等、5位 Noir 流の 3-stage 命名で再整理
- **H3 SED head fine-tuning frozen backbone**: 非鳥類 73 種専用 fine-tune として F25 後に追加候補
- **H5 Manual review**: 「未知種クラスタの発掘」(前回議論) と組合せれば Pantanal Insecta 25 sonotype の seed label 生成可能
- **H8/H9** は学習時 init/balance の細部、銀帯到達後の上積み

### 出典

- 1位 Nikita Babych — https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n / 公開 inference NB: `nikitababich/birdclef2025-1st-place-inference` / Walkthrough: https://www.youtube.com/watch?v=jivW1JBxV8s
- 2位 Volodymyr Sydorskyi — CEUR-WS Vol-4038 paper_256 / GitHub: https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place
- 4位 Dylan Liu — Walkthrough https://www.youtube.com/watch?v=G2IiEi3Ck2s
- 5位 Noir (myso1987) — https://github.com/myso1987/BirdCLEF-2025-5th-place-solution
- 12位 223223223 — SpeakerDeck 25位 Ryushi の比較考察に間接記載
- tekkix 上位 5 解説 — https://tekkix.com/articles/ai/2025/07/birdclef-2025-overview-of-the-competition-a
- SpeakerDeck Ryushi 25位 — https://speakerdeck.com/ryushi496/...

---

## 公開 NB / Discussion 知見

### 公開 NB スコア表 (2026-05-08 更新)

| NB / 著者 | LB | コア手法 | 我々への適用状況 |
|---|---|---|---|
| Gate (ulyanovantonamaranta) | 0.941 | ProtoSSM + Tucker SED 5-fold rank avg | ✅ blend tucker v1 = 0.939 で同等再現 |
| konbu17 | 0.943 | Gate + train_audio LinearHead 5%-15% | ❌ LinearHead 未実装 (公開重み `konbu17/bird26-train-audio-head-v1` 流用可、+0.002 期待) |
| mattiaangeli | 0.943 | Gate + temporal continuity gate | ❌ continuity gate 未実装 |
| needless090 | 0.934 | ProtoSSM v5 (d_model=320, 4 SSM layers) + 8 model SED ensemble + sonotype mirroring | ❌ アーキ拡張・mirroring 未実装 |
| Tucker Arrants Distilled SED | 0.898-0.920 | EfficientNet-B0 SED + Perch MSE 蒸留 | ✅ Tucker public 5-fold ONNX を直接 blend (我々の 0.939 構成) |
| HGNetV2 baseline (ttahara) | 0.876-0.898 | LSEHead + distillation (有無で +0.022) | ❌ |
| imaadmahmood Perch+ProtoSSM | 0.925 | ProtoSSM ベース | ✅ exp010 NB3 で同水準 (NB4 で 0.924) |

### Discussion 683791 「Best single model LB」要点

| Kaggler | 単一 LB | 構成 | 我々への含意 |
|---|---|---|---|
| **Boredom (cudacoding)** | 0.937-0.939 (no unlabeled) / **0.950+** (R1 pseudo) | timm SED non-Perch | unlabeled R1 で +0.012、R2-R3 で 0.96+ 期待 |
| Ali Ozan Memetoglu | 0.941 (no unlabeled) | 2025 pretrained checkpoint 転移 | NFNet/EffV2-S init、アーキ変更必要 |
| Salman Ahmed | 0.937 (no unlabeled) | EffNet v2B0, 20s, raw-wave Mixup, CE+Sigmoid | exp012 で実装済 (ほぼ同構成) |
| Tucker Arrants | 0.925 | DL なし: Perch emb + cluster + meta | exp010 NB3 v8 prior tables (+0.019) で同類効果 |

**ceiling**: labeled only ~0.94-0.945 (Boredom)、unlabeled R1 で 0.95、R2-R3 で 0.95-0.96+。

---

## 完了 / 却下済み (2026-05-08 更新)

| 手法 | 結果 | 教訓 |
|---|---|---|
| TTA (time-flip mel 反転) | exp006: 0.878→0.877 | ⚠️ 効果なし |
| Backbone B0→B2 | exp008 で B3 へジャンプ | ⚠️ B2 スキップ |
| 上位解法調査 | 2025 1-5位調査済 | ✅ |
| Dual-Model Pseudo (exp007) | 0.878→0.863 | ⚠️ 少量データ別モデル擬似ラベルは逆効果 (`feedback_exp007_lesson.md`) |
| Site/Hour embedding + TTA (NB3 v7) | +0.002 | ✅ 採用 |
| **Prior Tables (NB3 v8)** | **+0.019** | ✅ 採用 |
| Month embedding (NB3 v11) | -0.017 | ⚠️ OOD `nn.Embedding` 危険 (`feedback_ood_embedding.md`) |
| Cluster + Month 同時 (NB3 v18) | -0.007 | ⚠️ 1 push 1 変更原則 (`feedback_isolate_changes.md`) |
| **file_confidence_scale (NB3 v20)** | **+0.009 vs v8** | ✅ clean baseline |
| Adaptive δ-shift smoothing (v21) | +0.001 noise | ⚠️ revert (v20 baseline で後処理頭打ち) |
| Class-specific T (v22) | ±0 | ⚠️ revert |
| ResidualSSM 2nd-pass (v23) | -0.047 | ⚠️ 792 sample で zero-init head でも train residual 暗記 (`feedback_residualssm_overfit.md`) |
| MixUp on Perch embedding (v24) | -0.003 | ⚠️ frozen logit を mix すると物理的に存在しない入力 |
| Iter pseudo R1 (NB3 v25) | -0.034 | ⚠️ frozen Perch + 小 head 構造的ミスマッチ。SED 系で実施 |
| LightProtoSSM cross-attn (v26) | +0.007 | ✅ 採用 |
| SWA 単独 (NB4 v2) | ±0 | ⚠️ multi-seed=5 と機能重複 (`feedback_swa_redundant_with_multiseed.md`) |
| Focal Loss (NB4 v3) | -0.003 | ⚠️ BCE+pos_weight 環境では悪化 (`feedback_focal_with_pos_weight.md`) |
| Soft CE (exp011 Phase 3) | -0.030 | ⚠️ raw mixup なしでは破滅的 (`feedback_soft_ce_needs_raw_mixup.md`) |
| **Noisy Student (NB5/NB6)** | -0.001 | ⚠️ 自己蒸留の天井、66 file × 12 window で構造的限界 |
| **train_audio MLP pretrain (NB7/NB8)** | v1 -0.010 / v2 -0.008 / v3 abandon | ⚠️ focal vs SS ドメインギャップ深刻、catastrophic forgetting |
| TA retrieval LAMBDA tune (v8/v10) | v7 0.05 が最適 | ✅ |
| 外部 non-Aves pool 追加 (AnuraSet+iNat) (v9) | ±0 | ⚠️ 効果薄、retrieval は頭打ち |
| class-specific LAMBDA non-Aves 3x (v10) | ±0 | ⚠️ retrieval 軸完全頭打ち |
| **E19 (NB4 v11)** | 単独 +0.002 / blend ±0 | ⚠️ 単独 NB のみ有効、blend は Tucker smoothing と重複 |
| **blend ratio (NB4 v7 × Tucker SED)** | 50:50 = 0.939、40:60 = 0.937 | ✅ 50:50 確定 |
| Tucker Arrants 手法解釈 | KMeans でなく **KNN** が本人発言 | ⚠️ `feedback_verify_source_method.md` |
| exp011 Phase 1 学習 + 推論 | LB 0.840 (Best Val 0.9645)、val→LB ギャップ -0.124 | ⚠️ dual val 必須 (`feedback_small_val_holdout.md`) |
| exp011 Phase 2 (20s 化のみ) | LB 0.854 ±0 | ⚠️ neutral (target_size=256 で時間軸圧縮) |
| **exp012 Tucker SED 自前再現 fold0** | **LB 0.890** | ✅ pipeline 確認、Tucker public で代替可能 |
| **NB4 v7 × exp012 fold0 70:30 blend** | **LB 0.929** | ✅ 銅射程到達 (旧 best) |
| **NB4 v7 × Tucker public 5-fold ONNX (50:50)** | **LB 0.939** | ✅ **現行最高、+0.015 vs NB4 単独** |

## 重要な発見 (2026-05-08, EDA NB v4 で測定)

### Audio embedding model 相関 (vs Perch)

| 順位 | モデル | Spearman ↓ | gap_norm ↑ | 判定 |
|---|---|---|---|---|
| reference | Perch v2 | — | 1.059 | 基準 |
| 🥇 | **AVES** (wav2vec2-base) | 0.417 | 0.785 | **独立 × 有用 = blend 第一候補** |
| 🥈 | BirdNET v2.4 (6522d classifier) | 0.660 | 0.836 | 中独立 × 高有用 = 第二候補 |
| 🥉 | CLAP (laion-htsat-unfused) | 0.624 | 0.490 | 中独立 × 中有用 |
| ❌ | AST | 0.722 | 0.781 | Perch 冗長、blend 効果なし |
| ⚠️ | YAMNet | 0.510 | **0.213** | 種シグナル無し (ノイズ)、不採用 |

**重要な学び**: Spearman 低 ≠ blend に有効。Spearman + gap の 2 軸で判定が必須。Spearman だけ見て YAMNet 採用すると失敗していた。

---

## 見送り（現状優先度低）

| 手法 | 見送り理由 |
|---|---|
| Nocall 検出 | ProtoSSM 時系列モデルで既にカバー |
| CutMix | 2024 1位「Mixup hurt」報告 |
| 独自 CNN fine-tune 単体 | SED + 疑似ラベリングでカバー |
| d_model 拡大 128→192 (旧 NB3) | 学習データ 66 ファイルで過学習リスク (needless090 320 は別計算) |
| XenoCanto 事前学習 | 学習コスト高、他施策後 |
| Multi-fold 学習 (k>1) for NB3 | 学習時間 5 倍 |
| AudioMAE / BEATs フル推論 | CPU 90分予算超過 |
| **2025 pretrained checkpoint (Ali Ozan)** | NFNet-L0 / EffV2-S 化が必要、exp012 (B0) との差分大 |

---

## 分析・ツール

- [ ] species_auc.csv を exp010 ベースで再作成
- [ ] audio_similarity.ipynb 完成 (perch-meta キャッシュ利用)
- [ ] Streamlit Species AUC Report 更新 (exp010 OOF)
- [ ] EDA 結果を memory `project_eda_findings.md` として保存

---

## 参考: 過去上位解法サマリー

| 年度 | 1位手法 | Private LB |
|---|---|---|
| 2021 | SED + DenseNet121 + Nocall | F1 ベース |
| 2022 | BirdNET + ECA-NFNet + 時間シフト | - |
| 2023 | ConvNeXt + 疑似ラベリング + ONNX | 0.9+ |
| 2024 | EfficientNet-B0 + CE→sigmoid + データ品質フィルタ | 0.689 |
| 2025 | SED + EfficientNet (L0/B4/B3/B0) + RegNetY + 4 ラウンド Multi-Iterative Noisy Student + Power Scaling (p=1.82) + 昆虫/両生類専用 B0 + TTA 2.5s + ImageNet-21k init (2位) | 0.933-0.937 |

---

## 参考: 情報ソース

| ソース | URL | 有用性 |
|---|---|---|
| DCASE 2024 Task 5 Results | [link](https://dcase.community/challenge2024/task-few-shot-bioacoustic-event-detection-results) | ★★★ 最新 few-shot bioacoustics |
| DCASE 2023 Task 5 Results | [link](https://dcase.community/challenge2023/task-few-shot-bioacoustic-event-detection-results) | ★★★ SupCon/TS-VAD/BEATs 出典 |
| BioDCASE 2025 Challenge | [link](https://biodcase.github.io/challenge2025) | ★★ Stefan Kahl 主催 |
| Cornell Lab Research | [link](https://www.birds.cornell.edu/home/bioacoustics-research-program/) | ★★★ コンペ主催者 |
| Google Perch GitHub | [link](https://github.com/google-research/perch) | ★★ |
| Moummad SupCon 実装 | [link](https://github.com/ilyassmoummad/dcase23_task5_scl) | ★★★ |
| Discussion 683791 | [link](https://www.kaggle.com/competitions/birdclef-2026/discussion/683791) | ★★★ Best single model LB |
| BirdCLEF 2025 1位 writeup | [link](https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n) | ★★★ Multi-Iterative Noisy Student |
| VSydorskyy 2025 2位 GitHub | [link](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place) | ★★★ NFNet+EffV2 OpenVINO |
| Tucker Distilled SED 公開 dataset | `tuckerarrants/bc2026-distilled-sed-public` | ★★★ 5-fold ONNX 流用可 |
| konbu17 train_audio head 公開 | `konbu17/bird26-train-audio-head-v1` | ★★★ LinearHead 即流用可 |
