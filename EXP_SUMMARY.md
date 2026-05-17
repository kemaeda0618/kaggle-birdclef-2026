# BirdCLEF 2026 実験サマリー

最終更新: 2026-05-16

## 現状ベスト

**🥈 LB 0.945** (exp010 blend、5-way / 3-way 同点 plateau)
- 構成例 v5: NB4 0.40 / Tucker SED public 0.45 / exp014 R2 0.15
- 構成例 5way v3: NB4 0.35 / Tucker 0.40 / exp014 R2 0.15 / exp016 R2 0.10
- 銀メダル境界 (0.946+) まで -0.001、現 5 model 集合では ratio 調整で突破不可
- 突破には **新規 ingredients (新 backbone / 別 modality / submission TTA)** が必須

## 全実験サマリー

| Exp | 概念 / Backbone | 自己ベスト LB | 状態 |
|---|---|---|---|
| exp001 | EfficientNet 動作確認 (3 epoch / 1000 sample) | - | クローズ (sanity check) |
| exp002 | EfficientNet-B0 SED (2 epoch) | 0.721 | クローズ |
| exp003 | Perch v2 + exp002 アンサンブル | - | 実行時間切れ提出断念 |
| exp004 | Perch v2 + ProtoSSM + MLP Probe + Prior Fusion | 0.912 | exp010 系列に発展 |
| exp005 | 独自 CNN SED + pseudo label (5s) | 0.733 | クローズ (pretrain 無 CNN が原因) |
| exp006 | EfficientNet-B0 SED + pseudo label | 0.878 | クローズ |
| exp007 | Dual-Model Pseudo Label (B0、少量データ別モデル教師) | 0.863 | クローズ (`feedback_exp007_lesson.md` で教訓化) |
| exp008 | EfficientNet-B3 + DataParallel | - | 実行失敗 |
| exp009 | 5-Fold SED + Mel Cache (n_mels=128) | 0.868 | クローズ (mel cache パラ不一致) |
| **exp010** | **Perch + ProtoSSM/MLP blend + Tucker SED + 自前 SED 多 backbone blend** | **0.945** | 🎯 **現 best、blend 主体実験** |
| exp011 | timm SED non-Perch (Phase 1-13 ablation) + gate-fake008 | 0.943 | gate-fake008 は exp010 blend に吸収済 |
| exp012 | Tucker SED 自前再現 (EffNet-B0、5-fold 計画) | 0.890 (single fold) | クローズ (Tucker public ONNX で代替可と判明) |
| exp013 | Babych F25 R1 (pseudo gen + SED student + 3-way blend) | - | NB1/NB2 生成完了、推論未到達で停滞 |
| **exp014** | **HGNetV2-B0 + Perch distill SED** (自前 SED 主軸) | **0.907 (R2 ep4)** | R2-fast (Perch precompute、25 min/ep) 走行中 |
| **exp015** | **convnext_pico + Perch distill SED** | **0.910 (R2)** | R2 完走、4-way blend 投入済 |
| **exp016** | **regnety_008 + Perch distill SED** | **0.903 (R2)** | R2 完走、pseudo で gap 縮小実証 (-0.011) |
| **exp017** | **eca_nfnet_l0 + Perch distill SED** (BC25 1位 Nikita 採用 backbone) | **0.921 (R2)** | R3 val plateau (-0.0006)、R4 推奨せず |

## メダル境界と差分

| メダル | 必要 LB | 現状 (0.945) からの差 |
|---|---|---|
| 🥉 銅 | 0.929+ | **到達 ✅** |
| 🥈 銀 | 0.946+ | **+0.001** |
| 🥇 金 | 0.952+ 推定 | +0.007 |

締切: 2026-06-03 (残り 18 日)

---

## 系列別の進捗

### 系列 A: Perch + ProtoSSM/MLP (exp004 → exp010 NB4)

**exp010 NB4 v7 単独 LB 0.924** (loss 軸頭打ち)

- 構造: Perch v2 + ProtoSSM + MLP head の blend + Prior Tables (site/hour) + multi-seed=5 + Perch embedding TTA (`roll([-1,0,1])`) + retrieval (TA pool K=20, λ=0.05)
- 改良履歴: v1 baseline 0.923 → v7 で TA retrieval pool 追加 +0.001 → v8-v10 飽和
- Noisy Student (NB5/6) は teacher 越えず、自己蒸留の天井確認
- AVES / BirdNET の embedding 置換 path は **完全 close** (LB 0.700 / 0.745、Perch 事前知識喪失が致命的)

### 系列 B: Tucker SED public 5-fold ONNX 流用

**Tucker SED single ~0.93、blend 主軸**

- `tuckerarrants/bc2026-distilled-sed-public` の 5-fold ONNX (各 ~20MB)
- 公開 0.94 帯 NB の事実上スタンダード
- 入力: mel (256×313, sr=32000, n_fft=2048, hop=512, n_mels=256, fmin=20, fmax=16000)
- 推論: `0.5*sigmoid(clip) + 0.5*sigmoid(frame_max)` per fold → 平均 → Gaussian smoothing σ=0.65
- exp010 NB4 v7 × Tucker 50:50 rank blend = **LB 0.939-0.942** が中核

### 系列 C: 自前 SED 多 backbone (exp014/015/016/017)

**diversity-first 戦略、各 ~5-9M params の軽量 backbone を並列学習**

共通 recipe (Babych BC25 1位 流):
- Perch v2 distill (MSE on 1536-d emb)
- R1: focal 0.85 / labeled_sc 0.15 (pseudo 無し)
- R2: focal 0.70 / labeled_sc 0.10 / pseudo_sc 0.20
- AdamW lr=3e-4、cosine + warmup 2-3 ep、batch=64-192
- Loss: 0.5*BCE(clip) + 0.5*BCE(frame_max) + 1.0*MSE(distill_emb)

| Exp | Backbone | R1 LB | R2 LB | val→LB gap (R2) |
|---|---|---|---|---|
| exp014 | HGNetV2-B0 (6M) | 0.897 | 0.907 (ep4 peak) | -0.018 |
| exp015 | convnext_pico (9M) | 0.890 | **0.910** | -0.021 |
| exp016 | regnety_008 (6M) | 0.886 | 0.903 | -0.024 |
| exp017 | eca_nfnet_l0 (24M) | TBD | **0.921** ✨ | **-0.004** |

**観測**:
- **exp017 (NFNet) が単体 LB 最高** — NFNet rule (LR=3e-4 据置、sqrt scaling 無) で R2 LB 0.921 (gap -0.004)
- **R1 gap 大 backbone ほど R2 lift 大** — exp016 (gap -0.035→-0.024)、exp015 (-0.032→-0.021)
- **R3 で plateau** — exp017 R3 val 0.9243 (-0.0006)、Babych 経験則通り
- **R2 真の peak は ep4-8** — `feedback_pseudo_long_epoch_overfit.md` で確認、25 ep は memorize で劣化

### 系列 D: exp010 blend (現 best 0.945)

| Version | 構成 | LB |
|---|---|---|
| NB4 × Tucker 50:50 logit | NB4 v7 + Tucker | 0.939 |
| NB4 × Tucker rank 50:50 (v5) | rank blend 化 | 0.941 (+0.002) |
| + Sonotype mirror (v8) | Phase 1 完了 | 0.942 |
| 3way v2 (+ exp014 R1) | 3rd 軸追加 | 0.943 |
| 3way v4 (+ exp014 R2 ep10) | R1→R2 swap | 0.944 |
| **3way v5 (0.40/0.45/0.15)** | e14 weight ↑ | **0.945** 🎯 |
| 5way v3 (NB4/Tucker/e14/e16) | e15=0、e16 0.10 | 0.945 (v5 タイ) |
| 5way v4 (e14 0.10、e16 0.15) | swap | 0.945 (v5 タイ) |

**0.945 plateau 観測** (3 構成同点):
- e15 R1 (LB 0.890、gap -0.032) は 0.05 weight で **-0.001 drag**
- e14 R2 と e16 R2 は **interchangeable** (両方とも 6M、Perch distill、同 recipe → 誤差パターン類似)
- ratio 調整で silver 0.946+ 突破は **不可確定**

### 系列 E: exp011 timm non-Perch (gate-fake008)

**gate-fake008 LB 0.943** — exp011 系列の集大成 (LightProtoSSM + Tucker SED + Prior + Residual)

- Phase 1-4 で 0.854 まで、Phase 5-13 ロードマップ未進行
- gate-fake008 は exp010 blend の素材として吸収済、新規開発は停止

---

## ロードマップ (2026-05-16)

### 短期 (1 週間、銀 0.946+ 射程)

1. **submission TTA 2.5s** — exp010 NB4 (blend NB) で SED 4 model に audio-level offset TTA 実装
   - Babych H4 実測 +0.011 (BC2025 0.911 → 0.922)
   - 既存 `TTA_SHIFTS=[-1,0,1]` は Perch embedding roll のみ、SED audio offset は未適用
   - 注意: Kaggle CPU 90 分制限、まず 1 SED から段階適用
2. **exp014 R2-fast 完走確認** — Perch precompute cache + batch 128 で ~10.5h 1 session
3. **exp017 を blend に追加** — single LB 0.921 (現メンバー最強)、Nelder-Mead で再最適化

### 中期 (2 週間、金 0.952+ 射程)

4. **新軸 ingredients 投入**:
   - AVES embedding 系モデル (Spearman 0.417、最 diversity)
   - Babych H12 Auxiliary 700-class head (Insecta/Amphibia 底上げ)
   - SED 2.5s window TTA + multi-seed 3
5. **Power Scaling は新規 exp018 で初手から** — 既存 R2 への後付けは ROI 薄

### ❌ 非推奨パス

- exp014/015/016 R3 + Power Scaling (R3 plateau、PS でも抜けない)
- exp017 R4 (R3 すら頭打ち、確定で無駄)
- AVES/BirdNET embedding 単独置換 (LB 0.7 帯確定 close)
- 5 軸以上の同時 weight tune (OOF 過学習、LB 乖離)

---

## 主要な学び

### モデリング

- **Perch v2 が圧倒的に強い** — 単体 LB 0.94+ は Perch ベースが必須
- **SED + Perch distill** は backbone 不問で機能、軽量 backbone (6-24M) で十分競合
- **NFNet 系は test 域 generalization 圧倒的** — exp017 が R2 LB 0.921 (gap -0.004)
- **R1 で val→LB gap 大なら R2 で +0.017-0.020 跳ねる** — `feedback_pseudo_helps_domain_shift.md`
- **R3 は diminishing returns** — exp017 で実証、Babych 経験則通り
- **R2 pseudo distill の真の peak は ep4-8** — 25 ep は overfit、`feedback_pseudo_long_epoch_overfit.md`

### Blend / Ensemble

- **rank blend (rank average) > logit blend** — AUC と整合、+0.002 観測
- **Tucker SED public 単独より NB4 と blend** — diversity 寄与 +0.017
- **不均一 LB 構成では equal weight (0.20×5) 失敗** — 強軸 0.35-0.45 死守
- **architecture diversity より input modality diversity** — Spearman 相関で確認 (Perch vs AVES 0.417 が最低)
- **5 軸以上同時 tune は禁忌** — 2-group 分岐まで、`project_blend_ratio_playbook.md`

### Pseudo Label

- 単一モデル + 複数ラウンド反復が正解 (Babych BC25 1位)
- Dual-Model Pseudo (少量データ別モデル教師) は逆効果 (exp007 教訓)
- Power Scaling は **training-time pseudo target でのみ機能**、推論時 `pred ** k` は AUC rank 不変
- Pseudo TTA と Power Scaling は **二者択一** (両方は target 削りすぎ)

### Filename 軸

- Site/Hour prior は exploit 済 (+0.019)、FCS +0.009
- 未開拓: Wet/dry season binary、Session smoothing、Day-of-year (各 +0.001-0.003)
- Month/Date 学習可能 embedding は **NG** (-0.017 観測)、empirical prior 表で実装

### Inference / 提出

- Kernels Only、CPU 90 分制限
- ONNX + Gaussian smoothing σ=0.65 が Tucker 流標準
- mel cache を Dataset 化 (Kaggle FUSE I/O bound 回避)
- TTA は audio offset (2.5s) と embedding roll (±1s) は別物、両立可

### Kaggle インフラ

- KGAT_ token → KAGGLE_API_TOKEN env var → SDK kernels_push 既定
- slug は **必ず "birdclef2026-" prefix** + title 由来で確定
- Kaggle T4 は I/O bound (5340s/epoch、Tucker 比 53 倍遅) → Colab Pro+ A100 に移行
- 重い学習は Colab A100 (Pro+ 500 CU/月で SED 5-fold × 4 round 想定)

---

## ファイル参照

- 各 exp 詳細: `experiment/exp{NNN}/EXP_SUMMARY.md` (exp010/exp011 のみ)
- memory: `MEMORY.md` に exp 別状態ファイル一覧
- BACKLOG: `BACKLOG.md` に F/G/H 系列のロードマップ
