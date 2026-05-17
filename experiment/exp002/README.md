# exp002 - SED + AttBlockV2（過去上位手法の統合）

## 概要

| 項目 | 内容 |
|------|------|
| モデル | EfficientNet-B0 + SED (Sound Event Detection) + AttBlockV2 |
| 入力 | Mel Spectrogram (1ch, 128mel × 1001frame, **10秒** @ 32kHz) |
| 損失関数 | **CrossEntropyLoss**（主ラベルのクラスインデックス） |
| 推論時活性化 | sigmoid（CE訓練 → sigmoid推論 ← 2024-2025上位手法） |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Scheduler | CosineAnnealingLR + Warmup 1epoch |
| Augmentation | SpecAugment (FreqMask 27, TimeMask 100), Gaussian Noise |
| クラス不均衡 | WeightedRandomSampler: weight = (count/total)^-0.5 |
| Epochs | 30 |
| Batch Size | 32 |
| CV | StratifiedKFold (5-fold), fold0のみ学習 |
| 推論 | 重複推論: 10秒ウィンドウ × 3オフセット (±2.5秒) の平均 |

## exp001 からの改善点

| 手法 | 出典 | 効果 |
|------|------|------|
| SED + AttBlockV2 | BirdCLEF 2021-2025 上位共通 | 時間方向のAttention Pooling → フレームレベル特徴を活用 |
| CrossEntropyLoss → sigmoid推論 | BirdCLEF 2024 1st, 2025 1st | 主ラベルに集中した学習 → 推論時sigmoid で多クラス確率化 |
| 10秒入力 | BirdCLEF 2022-2025 上位 | より長い文脈 → 鳥の鳴き声パターンを捉えやすい |
| WeightedRandomSampler | BirdCLEF 2023-2024 上位 | 希少クラスのサンプリング頻度を増加 |
| SpecAugment | BirdCLEF 2021-2025 標準手法 | 周波数・時間方向のマスクで過学習抑制 |
| 重複推論 (3オフセット) | BirdCLEF 2023-2025 上位 | 複数ウィンドウの平均で予測の安定性向上 |
| Mel変換 GPU化 (torchaudio) | - | 学習速度改善 |

## ファイル構成

```
exp002/
├── notebook/
│   ├── train.ipynb      # 学習ノートブック（Google Colabで実行）
│   └── submission.ipynb # 推論・提出ノートブック（Kaggle上で実行）
├── output/              # 学習成果物（Git管理外）
│   ├── weights/         # best_fold0.pth → Kaggle Datasetへアップロード
│   └── logs/            # 学習ログ・曲線
└── README.md
```

## 実行フロー

```
[1] Colab: train.ipynb 実行
        ↓ best_fold0.pth
[2] Kaggle Dataset: birdclef2026-exp002-weights としてアップロード
        ↓
[3] Kaggle Notebook: submission.ipynb をコンペに紐付けて実行
        ↓ submission.csv を自動生成・提出
```

## アーキテクチャ詳細

```
Input: (B, 1, 128, 1001)  ← 10秒 × 128mel
  ↓
EfficientNet-B0 forward_features (global_pool='')
  ↓
(B, 1280, 4, 31)  ← stride=32 により 128/32=4, 1001/32≈31
  ↓
mean(dim=2)  ← 周波数次元を平均プール
  ↓
(B, 1280, 31)  ← 時系列特徴
  ↓
BatchNorm1d + Dropout(0.3)
  ↓
AttBlockV2: att=(B,234,31), cla=(B,234,31) → sum over T
  ↓
(B, 234)  ← clip-level logits
  ↓
CrossEntropyLoss(logits, primary_label_idx)  ← 訓練時
sigmoid(logits)                               ← 推論時
```

## 設計メモ

- `secondary_labels` は損失計算に含めない（CE loss は主ラベルのみ）
- SpecAugment は GPU 上で適用（torchaudio.transforms）
- 推論時の重複チャンク: center = end_sec - 2.5 ± {-2.5, 0, +2.5} の3点
