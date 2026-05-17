# exp010 EXP_SUMMARY

## 種構成と過去年データ overlap (species_overlap.ipynb / 2026-04-20)

Kaggle NB: https://www.kaggle.com/code/maekeso/birdclef2026-exp010-species-overlap

### 2026 種構成 (234 species)
- Aves 162 / Amphibia 35 / Insecta 28 / Mammalia 8 / Reptilia 1

### 過去年 BirdCLEF との overlap

| 年 | overlap | 内訳 |
|----|---------|------|
| 2025 | **42** (17.9%) | Aves 39 / Amphibia 2 / Mammalia 1 |
| 2024 | 1 (0.4%) | passer domesticus のみ |
| 2023 | 1 (0.4%) | butorides striata のみ |
| 2022 | 5 (2.1%) | Aves のみ |
| 2021 | **34** (14.5%) | Aves のみ |

- 2025 が唯一の非鳥類 overlap 源 (Amphibia 2 + Mammalia 1)
- 2024/2023 はほぼ無関係 (北米鳥類中心)

### 過去年 pretrain の射程
- **179 / 234 (76.5%) が過去年データ皆無**
  - Insecta 28 全部、Amphibia 33/35、Mammalia 7/8、Reptilia 1/1
- 2025 + 2021 の overlap (Aves 中心) は重複除き ~55-70 種程度
- **n_train < 20 を過去年で救えるのは 1 種のみ** → 過去年 pretrain は low-sample 救済としては効かない（既知 Aves の表現補強が主目的）

### 2026 train サンプル数分布
| n_train | 種数 |
|---------|------|
| 0 | 28 |
| 1-4 | 14 |
| 5-9 | 11 |
| 10-19 | 11 |
| 20-49 | 16 |
| 50-99 | 30 |
| 100+ | 124 |

### 戦略示唆
- **過去年 pretrain (2025+2021)**: 上位 Aves に対して特徴量を強化する目的に限定。+0.013 の効果報告 (memory) は妥当だが non-Aves には届かない
- **非鳥類 73 種 (28 insect + 35 amphibia + 8 mammalia + 1 reptilia)**: 2026 単独学習 + Perch zero-shot 頼み。NB3 の Prior Tables (site/hour shrinkage) が効くのはこの群と推定
- **0サンプル 28 種**: train_audio に音源無し → train_soundscapes pseudo-label か Perch zero-shot 必須

### 出力ファイル
- `/kaggle/working/overlap_species.csv` (83 行: 過去年×2026 重複種マッピング)
- `/kaggle/working/overlap_low_sample.csv` (1 行: 過去年 overlap × n_train<20)

---

## 非鳥類73種の外部データ調査 (Kaggle Dataset / 2026-04-20)

過去年 BirdCLEF が救えない非鳥類73種を Kaggle Dataset で補えるか調査。

### Coverage 結果

| ソース | Amphibia 35 | Insecta 28 | Mammalia 8 | Reptilia 1 |
|--------|---|---|---|---|
| **AnuraSet v2** (`bengtlueers/anuraset-v2-raw`, 8.5GB MIT) | 17 (49%) | 0 | 0 | 0 |
| **iNatSounds 2024** (`shadowdude/train-recordings`, 121GB) | 24 (69%, 1-14sample/種) | 3 (11%) | 4 (50%) | 0 |
| **Union (両方使用時)** | 27 (77%) | 3 (11%) | 4 (50%) | 0 |

- iNatSounds Insecta マッチ: **Quesada gigas (42), Guyalna cuta (1), Prionacris erosa (1)** のみ
- iNatSounds Mammalia マッチ: Panthera onca, Alouatta caraya, Canis familiaris, Bos taurus
- iNatSounds の Aves coverage: 152/162 (94%) ← Aves pretrainにも使える

### 未カバー 38 非鳥類種 (両sourceに無し)

- **Insecta sonotype01-25 全25種**: 学名でなく "sonotype01" 形式の競技独自分類なのでiNatに存在しない
- **Reptilia: Caiman yacare** (1種)
- **Amphibia 8種**: Adenomera guarani, Lysapsus limellum, Physalaemus centralis, Physalaemus albifrons, Pseudopaludicola mystacalis, Chiasmocleis mehelyi, Dermatonotus muelleri, Trachycephalus typhonius
- **Mammalia 4種**: Equus caballus, Sapajus cay, Plecturocebus pallescens, Mico melanurus

### 戦略示唆

- **AnuraSet v2 を Kaggle Dataset source に追加** → amphibia 17種の fine-tune (1種あたり大量サンプル想定、MIT で公開可)
- **iNatSounds は breadth 補強** だがサンプル少 (1-14/種) → 大規模 pretrain 後の few-shot fine-tune が現実的
- **iNatSounds の Aves 152種** は過去年 BirdCLEF と並ぶ Aves pretrainソース候補
- **Insecta sonotype 25種は外部データ完全に無し** → BirdCLEF 2026 train + train_soundscapes pseudo-label + Perch zero-shot に頼るしかない
- **Caiman yacare** は別途 xeno-canto / Macaulay Library を探す必要あり

### 関連ファイル
- `notebook/inat_match.py`: iNatSounds train.json と taxonomy.csv のマッチング検証スクリプト

---

## 実験 NB 全体マップ (2026-05-08 時点)

| NB | 役割 | Kaggle slug | 環境 | LB |
|---|---|---|---|---|
| NB1 | Perch ONNX で全 SS + train_audio の embedding 抽出 | `birdclef2026-exp010-nb1-embedding` | T4×2 | — (基盤) |
| NB1b | AnuraSet (17 amphibia 種) Perch embedding 抽出 | `birdclef2026-perch-anura` | CPU | — (基盤) |
| NB1c | iNat non-Aves 31 種 Perch embedding 抽出 | `birdclef2026-perch-embed-inat-nonaves` | CPU | — (基盤) |
| NB2 v10 | MLP solo 推論 | `birdclef2026-exp010-nb2-mlp` | CPU | 0.918 |
| NB3 v29 | ProtoSSM solo 推論 | `birdclef2026-exp010-nb3-protossm` | CPU | 0.921 |
| **NB4 v7** | **ProtoSSM + MLP blend (exp010 baseline)** | `birdclef2026-exp010-nb4-blend-protossm-mlp` | CPU | **0.924** |
| NB4 v8 | LAMBDA_TA=0.10 増幅 | 同上 | CPU | 0.922 (-0.002) |
| NB4 v9 | + AnuraSet/iNat 外部 non-Aves pool | 同上 | CPU | 0.924 (±0) |
| NB4 v10 | + class-specific LAMBDA non-Aves 3x | 同上 | CPU | 0.924 (±0) |
| **NB4 v11** | **+ E19 file-level consistency boost (max, β=0.2)** | 同上 | CPU | **0.926 (+0.002)** |
| NB5 | Noisy Student 5 ラウンド学習 | `birdclef2026-exp010-nb5-noisy-student` | T4 | — (基盤) |
| NB6 | NB5 重み load + 推論 | `birdclef2026-exp010-nb6-inference-ns` | CPU | 0.923 (-0.001) |
| NB7 | train_audio MLP head pretrain + fine-tune | `birdclef2026-exp010-nb7-pretrain-mlp` | T4 | — (基盤) |
| NB8 v1 | NB7 重み (primary one-hot) で推論 | `birdclef2026-exp010-nb8-inference` | CPU | 0.914 (-0.010) |
| NB8 v2 | + multi-hot (primary+secondary) + 緩和 fine-tune | 同上 | CPU | 0.916 |
| NB8 v3 | Perch sigmoid soft target + hard override | 同上 | CPU | (LB 待ち) |
| **blend v2** | **NB4 v7 + 自前 exp012 fold0 (70:30)** | `birdclef2026-exp010-blend-exp012` | CPU | **🎯 0.929 (銅射程)** |
| **blend tucker v1** | **NB4 v7 + Tucker public 5-fold ONNX (50:50)** | `birdclef2026-exp010-blend-tucker-sed` | CPU | **🎯 0.939 (現 best)** |
| blend tucker v2 | + ratio 40:60 (Tucker heavier) | 同上 | CPU | 0.937 (-0.002) |
| blend tucker v3 | NB4 v11 (E19) × Tucker (50:50) | 同上 | CPU | 0.939 (±0、E19 が Tucker smoothing と重複) |

## NB4 retrieval 軸 ablation 結論
- **TA pool (BC2026 train_audio 265k focal)** は LAMBDA=0.05 が最適、+0.001 のみの寄与
- **外部 pool (AnuraSet+iNat non-Aves)** は ±0、test に該当種が出てないか Perch 自体が非鳥類弱いため信号届かず
- **class-specific LAMBDA** non-Aves 強化も ±0、retrieval 軸は完全頭打ち
- → NB4 v7 (TA pool LAMBDA=0.05) を確定 baseline として固定、別軸で勝負する判断

## NB5/6 Noisy Student 結論
- 5 ラウンド × 5 seeds × (ProtoSSM+MLP) で擬似ラベル → 再学習 を反復
- ラウンド毎 loss 改善 -0.0015 程度の微小、LB に反映されず (NB6 v1 = 0.923)
- 構造的原因: 66 SS teacher が弱すぎる、自己蒸留では teacher を超えられない

## NB7/8 train_audio pretrain 結論
- focal 265k で MLP head pretrain → 66 labeled SS で fine-tune
- v1 (primary one-hot only): -0.010 大悪化 → focal の「他種 absent」仮定が SS multi-label と乖離
- v2 (multi-hot primary+secondary): -0.008、改善あるが不十分
- v3 (Perch sigmoid soft target): pending
- 構造的限界: focal vs soundscape のドメインギャップが深刻

## 🎯 銅射程到達: blend NB (2026-05-08)
- **NB4 v7 (Perch+ProtoSSM+MLP) × exp012 (Tucker SED 自前 fold0) を 70:30 で blend → LB 0.929**
- 単独 LB の比: NB4=0.924, exp012=0.890、ratio 70:30 で gap を吸収
- 過去最高、銅メダル境界 0.929 ジャストタッチ

## 重要発見 (2026-05-08): Tucker public 5-fold SED weights
- `tuckerarrants/bc2026-distilled-sed-public` で sed_fold{0-4}.onnx (各 ~20MB) が公開済
- 公開 NB の 0.941-0.943 帯 (mattiaangeli, konbu17, mtoshidesu) は全部この dataset 経由
- 我々の自前 exp012 (single fold 0.890) より圧倒的に強い (5-fold ensemble、推定 0.92-0.93)
- → blend NB の SED 部分を Tucker public に置換するだけで 0.93-0.94 帯射程

### Tucker SED ONNX inference 仕様
- 入力: mel spectrogram (B, 1, 256, 313)、32kHz × 5sec window
- mel 設定: n_fft=2048, hop=512, n_mels=256, fmin=20, fmax=16000, top_db=80, 標準化必須
- 出力: clip_logits (B, 234) + framewise (B, T, 234)
- per fold: `0.5 * sigmoid(clip) + 0.5 * sigmoid(frame_max)`
- 5 folds 平均後: `gaussian_filter1d(sigma=0.65, axis=0, mode='nearest')`

## 次の一手候補
1. **blend tucker v1 LB 確認** (push 済、結果待ち)
2. **BirdNET 軸追加** (3-way blend) — 公開 baseline `ahmadzulfiqar001/birdclef-2026-birdnet-baseline` 参考、48kHz 3sec の窓ミスマッチ吸収が課題、工数 2-3 日
3. blend ratio スキャン (60:40, 75:25 etc)
4. exp012 自前 5-fold は Colab で学習中だが、Tucker public で代替可なので優先度低下
