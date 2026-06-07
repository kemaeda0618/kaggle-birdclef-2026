# BirdCLEF 2026 — Post-Processing (PP) 調査結果

**作成日**: 2026-05-28
**現状**: exp090 = LB **0.951** (silver border、gold まで +0.006)
**残期間**: 6 日 / sub 30 枠
**user 制約**: BirdNET / AVES / 新 paradigm 禁止、既存 ckpt 組合せのみで gold push

---

## 0. 結論サマリ

1. **Babych 1st place の主要 PP 技法は exp090 に等価実装済** (gaussian smoothing, session smoothing, half-Babych, TopN ファミリー)
2. **exp090 の PP パラメータ値は anthony 0.950 NB から借用、自前 sweep してない** → 値 sweep の余地大
3. **真に未実装の新規軸**は: `Per-class blend weights (mapped/unmapped)`, `Overlapping framewise avg`, `Day-of-year prior`, `Class-level wet/dry prior`
4. **棄却済**: TopN K=1/POW=1.0 単独 port (dual FCS で overcalibration risk、exp078 ヘッダー警告)、Power Transform γ on pseudo、Wet/dry species-level、Geographic proximity、NFNet weight averaging、単純 Window blend
5. **最有力 sub 候補**: **lambda_prior 値 sweep (0.65 → 0.75 等)**、独立軸で +0.001-0.003 期待

---

## 1. exp090 PP 実装 audit (確定)

### 1.1 ACTIVE pp (値変更で sweep 可能)

| # | PP | 現在値 | 場所 (line) | source | 過去 sweep |
|---|---|---|---|---|---|
| 1 | `lambda_prior` (site + hour) | **0.65** (default 0.4 から up) | 2084, 2136 | anthony 0.950 NB | ❌ 未 |
| 2 | `rank_aware power` | **0.6** (default 0.4 から up) | 2199 | anthony 0.950 NB | ❌ 未 |
| 3 | `BLEND_W_E10` (NB4) | **0.35** | 2551 | exp019 (旧 3-way) | ⚠️ 旧 e17 stream のみ |
| 4 | `BLEND_W_SED` (Tucker) | **0.40** | 2552 | 同上 | ⚠️ 同上 |
| 5 | `BLEND_W_E17` (R3) | **0.25** | 2553 | exp078 で R3 swap | ❌ R3 stream で未 sweep |
| 6 | `FCS_TOP_K` (post-blend) | **2** | 2235 | Karnakbayev V18 | ⚠️ exp048 で K=1 試行 (別 stack +0.001) |
| 7 | `FCS_POWER` (post-blend) | **0.4** | 2236 | Karnakbayev V18 | ⚠️ exp078 ヘッダー警告 "0.4 維持 (overcalibration risk)" |
| 8 | Pre-blend FCS `top_k`/`power` | **2 / 0.4** hard-coded | 2197 | exp078 | ❌ 未 sweep |
| 9 | `adaptive_delta_smooth base_alpha` | **0.20** | 2201 | exp078 | ❌ 未 sweep |
| 10 | `TAX_SMOOTHING genus_alpha` | **0.15** | 2694 | anthony 0.950 NB | ❌ 未 sweep |
| 11 | `TAX_SMOOTHING class_alpha` | **0.05** | 2694 | anthony 0.950 NB | ❌ 未 sweep |
| 12 | `gaussian_filter1d sigma` (Tucker SED stream 内) | **0.65** | 2358, 2533 | Tucker SED default | ❌ 未 sweep |
| 13 | half-Babych ratio (clip vs frame_max) | **0.5 / 0.5** hard-coded | 2356, 2531 | Tucker SED | ❌ 未 sweep |
| 14 | `per_class_thresholds` | auto-calibrate (`calibrate_and_optimize_thresholds`) | 2204 | exp078 | self-tuning |
| 15 | label_smoothing (train 内) | **0.03** | 170 | train | training-time |

### 1.2 DEFINED but NOT CALLED (未使用 helper)

| # | helper | 定義 | 状態 |
|---|---|---|---|
| — | `smooth_predictions(probs, alpha=0.3)` | line 662 | ⚠️ helper だが呼ばれてない (代わりに `adaptive_delta_smooth` 使用) |
| — | `Temporal smoothing helper` | line 661 | ⚠️ 同上 |

### 1.3 ACTIVE pp (実装の意味確認済)

| # | PP | 実装 |
|---|---|---|
| 16 | MAPPED/UNMAPPED concept | ✅ Perch head 内専用 (Genus proxy 用)、3-way blend では **使ってない** |
| 17 | Genus proxy for unmapped species | ✅ Perch head 内 |
| 18 | class_weights (cap=10.0) | ✅ training 用 |

---

## 2. 未実装 PP 候補 (新規追加)

### 2.1 既知 source あり、実装すれば +0.001-0.005

| # | PP | 期待 lift | 実装コスト | source / 根拠 |
|---|---|---|---|---|
| A | **Per-class blend weights (mapped/unmapped 別比)** | **+0.002-0.005** | 中 (50 行) | lb-0948 公開 NB (mapped 50/30/20 vs unmapped 20/40/40) |
| B | **Overlapping framewise avg (neighbor chunks)** | +0.002-0.003 | 高 (20s stream 必要) | Babych 1st 直接 |
| C | **Padded center alignment** (first/last chunk centering) | +0.001 | 低-中 | Babych 1st 直接 |
| D | **Day-of-year smoothing** | +0.001-0.005 | 中 | filename_signals_map memo (未開拓) |
| E | **Class-level wet/dry season prior** (species-level は drag、class-level 未試) | +0.001-0.003 | 中 | exp023 教訓 |
| F | **Recorder/session metadata prior** | +0.001 | 中 | filename_signals_map memo |
| G | **Conditional class blending** (per-class 動的 weight) | +0.002-0.005 | 中 | lb-0948 派生 |
| H | **Per-class temperature** | +0.001-0.003 | 中 | 一般 |
| I | **TopN per-source then blend** | +0.001-0.003 | 中 | exp029 memo "per_source_mirror_idea" |
| J | **Stream-wise normalization before blend** | +0.001 | 低 | 一般 |
| K | **Geometric mean blend** (vs arithmetic) | ±0-0.001 | 低 | 一般 |
| L | **WeightedRandomSampler on pseudo (Babych 1st)** | training-time (+0.001-0.005) | 中 | Babych writeup 直接 |
| M | **Cell-level pseudo threshold (Jiacheng/2nd place style)** | training-time (+0.001-0.003) | 中 | Jiacheng は "効かず" 報告 |

---

## 3. 棄却済 PP (再挑戦してはダメ)

| # | PP | 実装 | LB 変化 | 根拠 |
|---|---|---|---|---|
| ✗1 | **TopN K=1, POW=1.0 単独 port to exp090** | exp048 で実装 | exp048 LB **0.950** (+0.001 vs base) | ⚠️ exp090 は dual FCS + rank_aware 0.6 で **既に強抑制**、port すると overcalibrate drag risk。**exp078 ヘッダーに明示警告** |
| ✗2 | Power Transform γ on pseudo (γ=1.2 Babych spec) | exp046 | **-0.005 LB** (val 0.9359 vs 0.9409) | 5-fold avg ensemble pseudo に over-suppress |
| ✗3 | Chunk filter > 0.2 on pseudo | exp045 | **-0.0062 val** (0.9347 vs 0.9409) | Aves silence drop で signal 失う |
| ✗4 | Wet/dry species-level prior (ALPHA=50, CLAMP [0.7, 1.5]) | exp023 | **-0.018 LB (大失敗)** | labeled SS 1478 segs では sparse すぎ、species-level 過剰 |
| ✗5 | Geographic proximity (Pantanal 内) | exp067 | **-0.001 LB** | test 全 Pantanal 固定で proximity が constant、site prior と redundant |
| ✗6 | NFNet weight averaging (5-fold ckpt state_dict 平均) | exp026 | **-0.018 LB** | 非線形性で broken model 化、output averaging 一択 |
| ✗7 | 単純 Window blend (5s + 10s/20s) | (user 記憶) | drag | logit-fix なしの単純混ぜで drag、R1 段階 |
| ✗8 | BirdNET 3rd axis | (user 禁止) | — | feedback_no_birdnet_aves_axis memo |
| ✗9 | AVES | (user 禁止) | — | 同上 |
| ✗10 | Asymmetric Loss / Focal BCE / CE (training loss) | failed methods memo | — | discussion 701938 chesteryuan 報告 |
| ✗11 | Delta TTA (frame ±2) 追加 | exp068 | **±0** (LB 0.950 = base) | half-Babych に既に内蔵 |
| ✗12 | Cross-Dialect Pantanal proximity | exp067 | **-0.001** | 固定地域で redundant |

---

## 4. Babych BC25 1st place PP との対応関係

**Source**: https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n
**Inference NB (実コード)**: https://www.kaggle.com/code/nikitababich/birdclef2025-1st-place-inference

| Babych 1st の PP | 我々 exp090 status | 備考 |
|---|---|---|
| **Overlapping framewise avg (neighbor 20s chunks)** | ❌ 未実装 | Babych の core PP、+0.002-0.003 lift 報告。要 20s stream |
| **5-point smoothing kernel [0.1, 0.2, 0.4, 0.2, 0.1]** | ✅ **等価実装** | `gaussian_filter1d sigma=0.65` (Tucker 内蔵) で近似実現 |
| **Padded center alignment** | ❓ 未確認 (おそらく未) | first/last chunk centering、+0.001 |
| **Delta shift TTA (frame ±2)** | ✅ half-Babych 内蔵 | exp068 で再確認、追加 ±0 |
| **half-Babych (50% clip + 50% frame_max)** | ✅ active | Tucker SED 内蔵 |
| **Power Transform on pseudo (γ 1, 1.54, 1.82, 1.67)** | 🚫 drag | exp046 で -0.005 確認、棄却 |
| **WeightedRandomSampler on pseudo** | ❓ 未確認 (おそらく未) | Babych writeup 直接、未試 |
| **Stochastic Depth drop_path=0.15** | ⚠️ 部分実装 | R1/R2/R3 NB は drop_path=0.1 固定、Babych Stage 別 (R1=0/R2+=0.15) 未実装 |

### BC2026 add-on (Babych BC25 には無い、BC2026 公開 NB 由来)

| BC2026 add-on PP | 我々 exp090 status | source |
|---|---|---|
| Site prior (lambda_prior) | ✅ 0.65 | anthony 0.950 NB |
| Hour prior | ✅ exp037 内蔵想定 | anthony |
| TAX_SMOOTHING | ✅ α=0.15/0.05 | anthony |
| Rank-aware blend | ✅ power=0.6 | anthony |
| Per-class thresholds | ✅ auto-calibrate | exp078 |

---

## 5. 値 sweep ROI ranking (今日 / 残期間で実装候補)

### 5.1 既存 param の値 sweep (1 行変更、純粋 ablation)

| Rank | PP sweep | 候補値 | 期待 lift | Risk | 根拠 |
|---|---|---|---|---|---|
| 🥇 1 | **lambda_prior 0.65 → 0.50 / 0.55 / 0.75 / 0.85** | 4 値 | +0.001-0.003 | 最小 | anthony 値、自前最適化未、elasticity 期待 |
| 🥇 2 | **rank_aware power 0.6 → 0.4 / 0.5 / 0.75 / 0.9** | 4 値 | +0.001-0.003 | 最小 | 同上 |
| 🥈 3 | **gaussian sigma 0.65 → 0.4 / 0.85 / 1.0** | 3 値 | +0.001-0.002 | 小 | Tucker default、自前最適化未 |
| 🥈 4 | **adaptive_delta base_alpha 0.20 → 0.10 / 0.30** | 2 値 | +0.001-0.002 | 小 | 未 sweep |
| 🥈 5 | **TAX (genus/class) = (0.15/0.05) → (0.10/0.03), (0.20/0.10)** | 2-3 組 | +0.001 | 小 | 未 sweep |
| 🥈 6 | **Stream weight (NB4/Tucker/R3 比) R3 stream で再 sweep** | 3-4 組 | +0.001-0.003 | 中 | exp019/021 e17 stream sweep 済、R3 stream で再 sweep 余地 |
| 🥉 7 | **half-Babych ratio 0.5/0.5 → 0.3/0.7 / 0.7/0.3** | 2-3 組 | +0.001-0.002 | 中 | 未 sweep |
| 🥉 8 | **Pre-blend FCS top_k/power (現 2/0.4)** | 1-2 組 | +0.001 (or drag) | 中 | overcalibration risk |
| ⚠️ 9 | **Post-blend FCS K=1/POW=1.0 (exp048 spec port)** | 1 組 | +0.001 or **drag** | **中-高** | exp078 ヘッダー警告あり |

### 5.2 新規 PP 追加 (実装コスト中-高)

| Rank | PP | 期待 lift | 実装コスト |
|---|---|---|---|
| 🥇 A | **Per-class blend weights (mapped/unmapped)** | +0.002-0.005 | 50 行 |
| 🥈 B | **Class-level wet/dry season prior** | +0.001-0.003 | 中 |
| 🥈 C | **Day-of-year smoothing** | +0.001-0.005 | 中 |
| 🥉 D | **Overlapping framewise avg (20s stream 経由)** | +0.002-0.003 | 高 (exp082 stream 活用要) |
| 🥉 E | **Conditional class blending** | +0.002-0.005 | 中 |
| 🥉 F | **Per-class temperature** | +0.001-0.003 | 中 |
| 🥉 G | **TopN per-source** | +0.001-0.003 | 中 |
| 🥉 H | **Stream-wise normalization** | +0.001 | 低 |
| ⚠️ I | **Padded center alignment** | +0.001 | 低-中 |
| ⚠️ J | **Recorder/session prior** | +0.001 | 中 |

---

## 6. sub 戦略 (残 6 日)

### 6.1 今日 sub 2 残 2 枠
- **sub 1**: exp081 R2 standalone (Window 軸 diagnostic、確定)
  - 期待 lift: 不明 (user 記憶では blend で drag、standalone も飽和の可能性)
  - 情報量: training axis info (R1 → R2 NS lift が Window 軸で発生したか)
- **sub 2 候補**:
  - 🥇 **lambda_prior 0.65 → 0.75** (値 sweep、+0.001-0.003)
  - 🥈 rank_aware power 0.6 → 0.5 or 0.75
  - 🥈 gaussian sigma 0.65 → 0.85

### 6.2 明日以降 25 sub 枠の使い方
1. **R4 完走後 → exp094 sub** (R3 → R4 swap) = 1 枠
2. **値 sweep 連続: 1 sub = 1 軸変化** = 6-8 軸 sweep 可能 (lambda, rank, sigma, base_alpha, TAX, stream weight, half-Babych ratio, FCS)
3. **新規 PP 実装 sub**: A (per-class blend) を時間あれば実装 = +0.002-0.005 期待
4. **公開 NB fork 余地**: 競合の新 idea を fork して試す

### 6.3 期待 lift 集計
- 値 sweep 6-8 軸 で **redundancy 50-70% 仮定** → 実効 +0.003-0.008
- 新規 PP (A: per-class blend) 成功 → 追加 +0.002-0.005
- R4 swap 効果 → 不明 (NS iter 4 で diminishing returns、+0.000-0.002)
- **gold 圏 (0.957-0.962) 到達可能性: 30-50%**

### 6.4 最悪ケース
- 全 PP sweep が ±0 → exp090 0.951 のまま silver 維持
- bronze 死守ライン (~0.945) は遠く、リスクは silver→silver 圏内
- **「sub 残し禁止」rule で連日 5 枠使う**ことで情報量取り続け

---

## 7. 主要参照 source

### Babych BC25 1st place
- Writeup: https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n
- Inference NB: https://www.kaggle.com/code/nikitababich/birdclef2025-1st-place-inference
- Extra data: https://www.kaggle.com/datasets/nikitababich/birdclef2025-1st-place-extra-data

### BC2026 公開 NB
- anthonytherrien 0.950 NB: lambda_prior 0.65, rank_aware 0.6, TAX_SMOOTHING 実装元
- lb-0.948 (youssefmo942009): Per-class blend (mapped 50/30/20 vs unmapped 20/40/40), 7 Tweaks (memo: reference_public_lb_0948_nb)
- Tucker SED public: bc2026-distilled-sed-public dataset (memo: reference_tucker_sed_public)

### BC2026 公開 discussions (2026-05-22 ~ 28)
- **#683791 — What is your best single model LB score?**:
  - Babych 本人 (2026-05-28): "2-seed ensemble: no pseudo + partial pp = 0.935, with pseudo + **full pp package** = 0.955"
  - Tucker (2026-05-26): "6 rounds pseudo, best 0.946, virtually zero ensemble benefit"
  - Salman (2026-05-26): "single 0.939, pseudo 0.944, ensemble 0.952. 2nd iter pseudo 効かない"
  - Antoine Masq (2026-05-27): "Do you ensemble models trained with same teachers? They might be too correlated"
- **#702366 — OverfitOracle single model 0.946**:
  - SED + pseudo + XC API + "capped clips per species/recordist for bias reduction" + 1-stage combined training
- **#694479 — Distilled SED Baseline (Tucker)**:
  - Tucker (2026-05-28): "Perch logits way too hot, mean prediction 0.7, temperature 必須"
  - Ali Ozan (2026-05-28): "temperature 1 + KL weight 下げる で good results from 1st iter"
- **#681297 — train_soundscapes_labels.csv 重複**:
  - 2026-05-28 確認: CSV まだ重複したまま (1478 rows)、drop_duplicates 必要

### memory 参照
- `reference_birdclef2025_1st_recipe.md`: Babych BC25 code-verified
- `reference_public_lb_0948_nb.md`: lb-0.948 NB 構造分析
- `reference_tucker_sed_public.md`: Tucker SED 仕様
- `project_exp048_topn1_result.md`: exp048 TopN K=1 LB 0.950 +0.001
- `project_exp090_status` (該当): NB build 完了
- `feedback_no_birdnet_aves_axis.md`: BirdNET/AVES/新 paradigm 禁止
- `project_filename_signals_map.md`: filename 軸の未開拓 +0.003-0.010
- `feedback_wd_prior_too_aggressive.md`: species-level wet/dry 棄却
- `project_exp067_result.md`: geographic proximity 棄却
- `project_exp046_gamma_ablation.md`: γ=1.2 pseudo 棄却
- `project_exp045_filter_ablation.md`: chunk filter > 0.2 棄却
- `project_exp026_v2_result.md`: NFNet weight averaging 棄却
- `feedback_failed_methods_discussion_701938.md`: ASL/Focal/CE 棄却

---

## 8. open question

1. **exp081 R2 standalone を本当に sub すべきか？** user 記憶では Window blend が drag、standalone も飽和の可能性。情報量のみ取得 sub なら他軸 (B/C/D) の方が ROI 高い可能性
2. **per-class blend weights (mapped/unmapped) 実装に時間投資すべきか？** +0.002-0.005 期待、実装 50 行 + テスト 1-2 時間
3. **R4 完走後の exp094 sub の lift 期待値**: NS iter 4 (R3 → R4) は他者報告 (Tucker / Salman / Boredom) で plateau 確認、+0.000-0.002 想定
4. **Antoine 指摘 "same teacher correlation"** (exp037 内 e17 ↔ exp029 R3) → R3 削除 sub の価値 (期待 ±0、R3 stream 削ぐ drag リスク)
5. **train_soundscapes_labels.csv drop_duplicates** が我々の val 計算で適用されてるか確認 (val→LB 相関に影響)

---

**最終更新**: 2026-05-28
**次回更新タイミング**: sub 結果取得時 / 新軸発見時 / 公開 NB に新 trend 出た時
