# BirdCLEF 2026 上位手法リサーチ (2026-04-05)

調査日: 2026-04-05
調査者: Claude (Sonnet 4.6)

---

## 1. リーダーボード状況

| 順位 | スコア | 備考 |
|------|--------|------|
| 1位 | 0.942 | KaggleClaw |
| 2位 | 0.941 | |
| 3〜4位 | 0.940 | |
| 5〜6位 | 0.939 | |
| Top20ライン | 0.935+ | |

前回(2026-03-24)から変化:
- 1位: 0.937 → 0.942 (+0.005)
- Top20ライン: 0.926 → 0.935 (+0.009, 大幅上昇)

---

## 2. 注目ノートブック調査結果

### 2-1. kamongi/pantanal-distill-birdclef2026 (本プロジェクト exp004 のベース)
- **Public LB**: ~0.926 (v36時点)
- **アーキテクチャ**: Perch v2 (凍結) + ProtoSSM + MLP Probe + Prior Fusion
- **ProtoSSM**: 1536次元 → Linear(128) → 双方向SSM(2層) → Prototypical Classification
  - Selective State Space Model (Mamba系)
  - 12窓(60秒)の時系列文脈を活用
  - コサイン類似度によるprototypical分類
  - Knowledge Distillation: BCE + MSE(Perch蒸留) + 分類群補助損失
- **MLP Probe**: クラスごと独立MLP(hidden=128)、PCA圧縮64次元特徴
- **Prior Fusion**: サイト×時刻の出現確率でBayesian補正
- **推論時間**: CPU 42分 (90分制限内)

### 2-2. yusufmurtaza01/pantanal-distill-birdclef2026-improvement (296 votes)
- **推定LB**: pantanal-distillの改良版 → 0.926〜0.935程度
- **特徴**: pantanal-distillノートブックのフォーク・改良
- **Kaggle直接読み取り不可** (JavaScript rendering required)
- **推測内容**: アンサンブル比率調整、Temperature/Power scaling、ProtoSSMパラメータ拡大等

### 2-3. hideyukizushi/bird26-reproduce-perch-protossm-resssm-inf-train (93 votes)
- **タイトル**: Perch + ProtoSSM + ResSSM 再現実装
- **ResSSM**: ProtoSSMの亜種（残差SSM構造の実験）
- **特徴**: TRAIN付きのため、自前でProtoSSMを再学習可能なノートブック
- **Kaggle直接読み取り不可**

### 2-4. yaroslavkholmirzayev/protossm-v18-maximum-ensemble-artifact (101 votes)
- **タイトル**: ProtoSSM v18 + Maximum Ensemble
- **特徴**:
  - ProtoSSMのバージョン18（多数の改良イテレーション後）
  - Maximum Ensemble: 複数ProtoSSMの予測をmax集約
  - artifactという名前からモデルアーティファクトをKaggle Datasetに保存した可能性
- **推測LB**: 0.930〜0.940程度
- **Kaggle直接読み取り不可**

### 2-5. ttahara/birdclef-2026-hgnetv2-b0-baseline-training (96 votes)
- **アーキテクチャ**: HGNetV2-B0 (PaddlePaddle発のCNNバックボーン)
  - GPU最適化のCNN、階層的特徴抽出
  - GhostConvでFLOPs削減（最大50%）
  - B0はEfficientNet-B0相当の軽量版
- **用途**: 独自CNNバックボーンの探索 (Perch非依存)
- **意義**: Perch probe系から独自CNN学習系への移行の試み
- **Kaggle直接読み取り不可**

### 2-6. chiranjithdharma/0-910-score (164 votes)
- **Public LB**: 0.910
- **公開日**: 2026-03-23
- **推測**: Perch v2 + MLP Probe系の基本実装（本プロジェクト exp003 の参照元）
- **Kaggle直接読み取り不可**

---

## 3. 手法別分析

### 3-1. Perch probe系 (現在の主流)

**特徴**:
- Google Perch v2 (凍結) で1536次元埋め込み抽出
- 上にMLPやSSMを積んで分類
- CPU推論: Perch素で~3.3時間 → TFLite化で~16分

**スコア帯**:
- Perch + simple probe: 0.908 (yashanathaniel)
- Perch + ProtoSSM + MLP: 0.926 (kamongi)
- ProtoSSM v18 ensemble: 0.930〜0.940 (推定)

**天井**:
- Perchの特徴抽出が凍結のため、train_audioから学習できない
- Insecta/Amphibia等非鳥類はPerchのカバレッジ外で弱い
- 0.935+には疑似ラベルかSEDとのアンサンブルが必要

### 3-2. 独自CNN学習系 (BirdCLEF 2025 Gold系統)

**特徴**:
- EfficientNet V2-S / B3 / NFNet-L0をMel spectrogramで学習
- OpenVINO/ONNX変換でCPU推論高速化
- 多ラウンド疑似ラベリングが必須

**スコア帯 (BirdCLEF 2025参考)**:
- 1位 (Nikita): ~0.933 (4ラウンド疑似ラベリング + SoftAUCLoss)
- 2位 (VSydorskyy): Private 0.928 (NFNet + EfficientNetV2 + ONNX/OpenVINO)

**BirdCLEF 2026での適用**:
- HGNetV2-B0 (ttahara) がこの系統の探索
- 上位の0.942チームは独自CNN + 疑似ラベリングの可能性が高い

### 3-3. ハイブリッド系

**概念**:
- Perch で疑似ラベル生成 (Phase A)
- 疑似ラベルでSED (EfficientNet) をColab学習 (Phase B)
- Perch + SED アンサンブルで提出 (Phase C)

**期待スコア**: 0.935〜0.945

---

## 4. 疑似ラベリングの使われ方

### BirdCLEF 2025 実績
- **1位 (Nikita)**: 2ラウンド疑似ラベリング → 0.87→0.91 (+0.04)
  - Power Transform: 1.0 → 1.54 → 1.82
  - Xeno Archive全データで事前学習 → +0.03 (0.84→0.87)
- **2位 (VSydorskyy)**: Focal BCE Loss、信頼度閾値 F2≥0.5, match≥0.1, prob≥0.4
- **5位 (myso1987)**: EfficientNet B0/B3/V2系4モデルアンサンブル、3ラウンド

### BirdCLEF 2026 での適用
- train_soundscapes: ~10,000ファイル (うち66ファイルがラベル付き)
- 未ラベル~10,534ファイルに疑似ラベル付与が大きなゲイン源
- 3ラウンド実施で +0.02〜0.04 見込み

---

## 5. CPU 90分制限への対処法

| 手法 | 推論時間 | 備考 |
|------|---------|------|
| Perch TF (素) | ~3.3時間 | NG |
| Perch TFLite | ~16分 | OK、推奨 |
| BirdSetEfficientNetB1 | ~26分 | OK |
| EfficientNet ONNX | ~20〜30分 | OK |
| EfficientNet OpenVINO | ~15〜20分 | OK、最適 |
| BirdNET | ~72分 | ギリギリ |

**組み合わせ例** (90分以内):
- Perch TFLite (16分) + EfficientNet ONNX (25分) + 後処理 = ~45分
- ProtoSSM on Perch emb (42分, exp004実測) = OK

---

## 6. 0.935+達成のための必要要素

### 現在のギャップ分析
- exp004 (ProtoSSM + MLP): LB 0.912
- Top20ライン: 0.935
- ギャップ: +0.023

### 有効な手法 (優先度順)

1. **疑似ラベリング3ラウンド** (+0.02〜0.04)
   - 最重要。2023年以降全上位が使用
   - train_soundscapesの未ラベル~10,000ファイルを活用
   - Perch → ProtoSSM で疑似ラベル生成 → 再学習 × 3

2. **SED + EfficientNet (独自CNN学習)** (+0.01〜0.03)
   - Colab GPUで学習、OpenVINO変換でCPU高速推論
   - Perch probeの天井を超えるポテンシャル
   - HGNetV2-B0も同系統の選択肢

3. **外部データ追加** (+0.01〜0.03)
   - XC Archive全データで事前学習
   - BirdCLEF 2021〜2024データ追加
   - 過去年度データは種あたり500件上限推奨

4. **アンサンブル多様化** (+0.01〜0.02)
   - ProtoSSM + SED + Perch probe の3系統アンサンブル
   - ProtoSSM v18スタイルの iterative ensemble (max ensemble)

5. **Power Scaling** (+0.01〜0.02)
   - 出力確率の power 乗算: p^power (power = 1.5〜2.0)
   - グリッドサーチで最適値探索

6. **SoftAUCLoss** (+0.005〜0.01)
   - AUC直接最適化。BirdCLEF 2025 1位で使用
   - BCE単独より最終指標に直接効く

---

## 7. 重要な知見まとめ

### Perch probe系 vs 独自CNN学習系
- **現在(2026-04-05時点)**: Perch probe系が0.912〜0.935に密集。上位は独自CNN系と推定
- **主流はPerch probe系**: 実装容易で高スコアが出るため参加者多数
- **Gold圏狙いは独自CNN系が必要**: Perchの天井(推定0.935)を超えるにはEfficientNet/NFNet学習が必要

### BirdCLEF 2026固有の注意点
- **234種** (BirdCLEF 2025: 206種より多い)
- **Insecta/Amphibia/Reptilia含む** (Perchのカバレッジ外種が多い)
- **パンタナール特有の種** (ドメインシフト対策が重要)
- **train_soundscapes_labels.csv あり** (BirdCLEF 2025にはなかった) → Prior Fusionに活用可能

### HGNetV2-B0について
- PaddlePaddle由来のCNN (timm/transformersでも利用可能)
- B0は軽量、B6は75.3Mパラメータ (大型)
- EfficientNet系の代替として速度・精度のバランスが良い
- BirdCLEF 2026での実績は調査中 (ttaharaノートブック)

---

## 8. 参考ソース

- [BirdCLEF 2025 1st Place Solution (Nikita Babych)](https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n)
- [BirdCLEF 2025 2nd Place Solution (VSydorskyy)](https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place)
- [BirdCLEF 2025 5th Place Solution (myso1987)](https://github.com/myso1987/BirdCLEF-2025-5th-place-solution)
- [Distilling Spectrograms into Tokens (arXiv:2507.08236)](https://arxiv.org/html/2507.08236v1)
- [State Space Models for Bioacoustics (arXiv:2512.03563)](https://arxiv.org/html/2512.03563)
- [HGNetV2 Architecture (HuggingFace)](https://huggingface.co/docs/transformers/model_doc/hgnet_v2)
- [BirdCLEF++ 2026 (LifeCLEF)](https://www.imageclef.org/BirdCLEF2026)
- [BirdCLEF 2025 Top-5 Overview (tekkix)](https://tekkix.com/articles/ai/2025/07/birdclef-2025-overview-of-the-competition-a)
- [Top 2% 解法ブログ (BirdCLEF+ 2025)](https://medium.com/@maxme006/how-i-climbed-to-the-top-2-in-birdclef-2025-every-failure-every-lesson-and-why-details-matter-273d781a33df)
