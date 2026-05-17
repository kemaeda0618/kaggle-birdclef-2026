# BirdCLEF 2026 フォーラム調査レポート

調査日: 2026-04-05

## 調査方法の注意

Kaggle フォーラムは JavaScript 動的レンダリングのため、WebFetch での直接取得は不可能。
代替として以下の方法で情報収集:
- WebSearch で個別ディスカッション URL を特定
- GitHub の解法リポジトリ (BirdCLEF 2025 上位解法) から技術詳細を抽出
- 学術論文 (arXiv, CEUR-WS) から検証済み手法を収集
- 公開 Kaggle Notebook のメタデータから存在確認

---

## 1. BirdCLEF 2026 フォーラムで確認されたトピック

### 1-1. Potential Kaggle runtime incompatibility
- URL: https://www.kaggle.com/competitions/birdclef-2026/discussion/684693
- 内容: Kaggle 実行環境の互換性問題に関する議論 (具体的内容は動的コンテンツのため取得不可)
- 重要度: 高 (推論ノートブック作成前に確認必須)
- 推測: TFLite / OpenVINO などの推論最適化ライブラリのバージョン互換性問題の可能性

### 1-2. 確認された公開ノートブック (BirdCLEF 2026)

| ノートブック | スコア (Public) | 備考 |
|---|---|---|
| yashanathaniel/birdclef-2026-perch-v2-0-908 | **0.908** | Perch v2 使用、現時点で確認できる高スコア |
| dedquoc/birdclef-2026-the-robust-submission-starter | 不明 | Robust Starter (2026-03-21公開) |
| emanuellcs/birdclef-2026-training | 不明 | 学習ノートブック (2026-03-18公開) |
| gourabr0y555/birdclef-26-inference-notebook | 不明 | 推論ノートブック |
| meenalsinha/birdclef-2026 | 不明 | スターターノートブック |

### 1-3. 確認された公開データセット (BirdCLEF 2026)

| データセット | 概要 |
|---|---|
| pulkitsahu89/birdtransform-birdclef-2026-transformer-model | CNN+Transformer ハイブリッドモデル |
| tonylica/birdclef-2026-model | 学習済みモデル |
| llkh0a/birdclef-2026-repack | データリパック |

---

## 2. BirdCLEF 2025 上位解法サマリー (2026 への転用可能性)

### 2-1. 1位: Multi-Iterative Noisy Student (Nikita Babych)
- URL: https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n
- スコア: 0.93 AUC (1位/2,025チーム)
- BirdCLEF 2026 への転用可能性: 高

キーアイデア:
- EfficientNet スタック: 4xv2_s, 3xv2_b3, 4xb3_ns, 2xb0_ns (合計13モデル)
- 3段階学習: FocalLoss -> pseudo labels -> 2段階自己蒸留
- SoftAUCLoss: AUC を直接最適化するカスタム損失関数 (ペアワイズ差分+ログロス)
- データ: Xeno-Canto +5,489件 + 昆虫・両生類 +17,197件を外部追加
- 希少クラス (<30 サンプル) は全録音を手動レビューしてノイズ除去
- 疑似ラベル: unlabeled soundscapes の 5秒クリップを予測し soft label として使用
- 実装難易度: 高 (13モデルのアンサンブル + 複数回学習が必要)

### 2-2. 2位: Domain Shift Distillation (VSydorskyy)
- URL: https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place
- スコア: Public 0.925 / Private 0.928
- BirdCLEF 2026 への転用可能性: 高

キーアイデア:
- 2モデルアンサンブル: ECA-NFNet-L0 + TF-EfficientNetV2-S (in21k pretrained)
- 学習: Focal BCE Loss + label smoothing, 50 epochs
- 疑似ラベル閾値: prob >= 0.5, model confidence >= 0.1, prob >= 0.4
- モデル変換: PyTorch -> ONNX -> OpenVINO FP16 (CPU で高速推論)
- HDF5形式で音声特徴量をプリコンピュート (学習速度向上)
- 実装難易度: 中 (コードは公開済み、インフラ要件が高い: RTX 4090, 126GB RAM)

### 2-3. 5位: EfficientNet Ensemble + OpenVINO (myso1987)
- URL: https://github.com/myso1987/BirdCLEF-2025-5th-place-solution
- BirdCLEF 2026 への転用可能性: 高

キーアイデア:
- 4モデル: EfficientNet B0, B3, V2-B3, V2-S
- 2段階疑似ラベル: soundscape 専用スクリプト + 通常の疑似ラベル
- OpenVINO 変換で CPU 推論を最適化
- 希少クラス (<20 サンプル) をオーバーサンプリング
- 実装難易度: 中

### 2-4. 38位 Top 2%: Ensemble CNN + Community SED (Max Melichov)
- URL: https://medium.com/@maxme006/how-i-climbed-to-the-top-2-in-birdclef-2025-every-failure-every-lesson-and-why-details-matter-273d781a33df
- スコア: 0.902 AUC (38位/2,025チーム)
- BirdCLEF 2026 への転用可能性: 高 (シンプルで再現しやすい)

キーアイデア:
- EfficientNet-B0 + 粗めのスペクトログラム (N_FFT=2048) + GeM pooling
- Silero-VAD で人間の声を除去
- Quantile-Mix (alpha=0.5): mean と rank 平均の組み合わせ
- 疑似ラベルで 0.817 -> 0.835 AUC に改善
- 2021-2024 の BirdCLEF データで事前学習してから fine-tuning
- 重要な教訓: コミュニティ共有 SED モデルとのアンサンブルが最大の改善要因
- 実装難易度: 低〜中

### 2-5. Spectrogram Token 蒸留論文 (軽量化手法)
- URL: https://arxiv.org/html/2507.08236v1
- 著者: DS@GT チーム
- BirdCLEF 2026 への転用可能性: 高 (TFLite 変換は 2026 でも有効)

キーアイデア:
- TFLite 変換: Perch 17.0秒/ファイル -> 1.4秒/ファイル (約12倍高速化)
- Best transfer モデル (BirdSetEfficientNetB1): ROC-AUC 0.810 (公開LB), 約26分
- STSG (トークン化): ROC-AUC 0.559 (公開LB), 約6分 (最速だが精度低)

---

## 3. 調査トピックへの回答

### 3-1. 上位解法のヒント

BirdCLEF 2026 現時点の最高スコア (確認できる範囲): Public 0.908 (Perch v2 ノートブック)

BirdCLEF 2025 上位解法からの主要テクニック:
1. EfficientNet アンサンブル (v2_s + v2_b3 + b3_ns + b0_ns) が最強クラス
2. SoftAUCLoss で AUC を直接最適化
3. 反復的ノイジースチューデント (soundscapes 疑似ラベル付け -> 再学習 x 複数回)
4. 2021-2024 年の BirdCLEF データで事前学習してから fine-tuning

### 3-2. Perch v2 関連

- 限界: 素の TF 版では ~17秒/ファイル -> 700ファイルで 3.3時間 (90分制限 NG)
- 改善策: TFLite 変換で 1.4秒/ファイル -> ~16分 (制限内)
- Kaggle での実績: yashanathaniel が Public 0.908 を達成
- 詳細: /survey/perch_v2_inference_time.md 参照

### 3-3. 疑似ラベリング

BirdCLEF 2025 での標準的なアプローチ:
1. 初期モデルをラベル付きデータで学習
2. train_soundscapes の 5秒チャンクに対して予測し soft labels を生成
3. 信頼度閾値でフィルタリング (prob >= 0.4~0.5 程度)
4. 疑似ラベル付きデータと元データを混合して再学習
5. これを 2-3 回繰り返す (反復的ノイジースチューデント)

重要な知見:
- 疑似ラベルは 5秒クリップの「真ん中」部分を使うと最も効果的 (端より中央が安定)
- 疑似ラベル後にサウンドスケープで validation すると public LB との相関が高くなる

BirdCLEF 2026 固有:
- train_soundscapes_labels.csv がローカルに存在 -> ラベル付きサウンドスケープを直接活用可能

### 3-4. CPU 推論高速化テクニック

| 手法 | 速度 | 精度 | 実績 |
|---|---|---|---|
| Perch TFLite | ~1.4秒/ファイル (16分) | 高 | BirdCLEF 2025, 2026 確認 |
| EfficientNet ONNX | ~0.5-2秒/ファイル | 高 | BirdCLEF 2024, 2025 |
| EfficientNet OpenVINO FP16 | さらに高速 | 高 | BirdCLEF 2025 2位 (0.928) |
| BirdNET (1秒ステップ) | ~6.2秒/ファイル (72分) | 中 | BirdCLEF 2025 |
| STSG トークン | ~0.5秒/ファイル (6分) | 低 (0.559) | BirdCLEF 2025 |

推奨: Perch TFLite (単体) + EfficientNet ONNX (アンサンブル時)

### 3-5. 新しいデータソース・前処理

BirdCLEF 2025 で使われた外部データ:
- Xeno-Canto: +5,489件 (鳥類)
- iNaturalist: +17,197件 (昆虫・両生類)
- Colombian Soundscape Archive (CSA)
- 過去の BirdCLEF データ (2021-2024)

前処理の工夫:
- Silero-VAD で人間の声を除去 (重要)
- 希少クラスは全録音を手動レビューしてノイズ・無音区間を削除
- 音声の最初の 30秒 (希少クラスは 60秒) を使用
- 20サンプル未満のクラスはアップサンプリング

### 3-6. train_soundscapes の活用方法

BirdCLEF 2025 での活用パターン:
1. 疑似ラベルのソース: 最も一般的。信頼度 >= 0.4 でフィルタリング
2. Validation set: test と同じドメインのため CV より public LB との相関が高い
3. Fine-tuning: 疑似ラベル付き soundscape で fine-tuning してドメインシフトを解消

BirdCLEF 2026 固有:
- train_soundscapes_labels.csv に明示的なラベルが存在 (2026 の新機能)
- 正式なラベルとして学習データに組み込める (疑似ラベル不要)
- ただし弱ラベルか強ラベルかを確認する必要あり

---

## 4. 推奨アクション (優先度順)

### 優先度: 緊急

1. フォーラムの runtime incompatibility 議論を Kaggle 上で直接確認
   - URL: https://www.kaggle.com/competitions/birdclef-2026/discussion/684693
   - どのライブラリが問題かを把握し、推論ノートブック構築前に対処

2. yashanathaniel/birdclef-2026-perch-v2-0-908 のコードを Kaggle 上で直接確認
   - Public 0.908 の実装を把握してベースラインとする
   - どの Perch バリアントを使っているか確認 (TFLite か TF か)

### 優先度: 高

3. train_soundscapes_labels.csv の内容を確認
   - ローカルに存在するため内容を確認して活用方針を決定
   - 弱ラベルか強ラベルかを確認

4. SoftAUCLoss の実装
   - BirdCLEF 2025 1位が使用。AUC を直接最適化できる
   - class SoftAUCLoss(nn.Module): ペアワイズ差分 + ログロスで実装

5. 疑似ラベリングパイプラインの構築
   - train_soundscapes_labels.csv を seed として使う
   - 反復回数: 2-3 回 (BirdCLEF 2025 1位の手法)

### 優先度: 中

6. EfficientNet (v2_s, v2_b3, b3_ns) で事前学習
   - 2021-2025 年の BirdCLEF データで事前学習してから 2026 データで fine-tuning

7. OpenVINO FP16 変換の検討
   - EfficientNet + NFNet アンサンブル -> ONNX -> OpenVINO FP16
   - 複数モデルアンサンブルを 90 分以内に収めるための最重要最適化

8. Silero-VAD による人間声除去
   - トップ解法で採用。iNat/XC データのノイズ除去に有効

---

## 参考ソース

- BirdCLEF+ 2026 フォーラム: https://www.kaggle.com/competitions/birdclef-2026/discussion
- runtime incompatibility 議論: https://www.kaggle.com/competitions/birdclef-2026/discussion/684693
- Perch v2 0.908 ノートブック: https://www.kaggle.com/code/yashanathaniel/birdclef-2026-perch-v2-0-908
- BirdCLEF 2025 1位解法: https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n
- BirdCLEF 2025 2位解法 GitHub: https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place
- BirdCLEF 2025 5位解法 GitHub: https://github.com/myso1987/BirdCLEF-2025-5th-place-solution
- BirdCLEF 2025 Top 2% (Medium): https://medium.com/@maxme006/how-i-climbed-to-the-top-2-in-birdclef-2025-every-failure-every-lesson-and-why-details-matter-273d781a33df
- Distilling Spectrograms into Tokens (arXiv:2507.08236): https://arxiv.org/html/2507.08236v1
- BirdCLEF++ LifeCLEF 2026 公式: https://www.imageclef.org/BirdCLEF2026
- BirdTransform BirdCLEF 2026 モデル: https://www.kaggle.com/datasets/pulkitsahu89/birdtransform-birdclef-2026-transformer-model
