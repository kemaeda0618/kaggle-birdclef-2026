# exp001 - EfficientNet-B0 ベースライン

## 概要

| 項目 | 内容 |
|------|------|
| モデル | EfficientNet-B0 (tf_efficientnet_b0_ns, ImageNet21k pretrained) |
| 入力 | Mel Spectrogram (1ch, 128mel × 500frame, 5秒 @ 32kHz) |
| 損失関数 | BCEWithLogitsLoss |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Scheduler | CosineAnnealingLR |
| Augmentation | Mixup, Gaussian Noise, Time Shift |
| Epochs | 30 |
| Batch Size | 32 |
| CV | StratifiedKFold (5-fold), fold0のみ学習 |

## ファイル構成

```
exp001/
├── train.py              # 学習スクリプト（Google Colabで実行）
├── notebook/
│   └── inference.ipynb   # 推論・提出ノートブック（Kaggle上で実行・提出）
├── config/
│   └── config.yaml       # ハイパーパラメータ
├── src/
│   ├── dataset.py        # Dataset, Mixup
│   ├── model.py          # EfficientNet-B0 モデル
│   └── utils.py          # seed, metrics, checkpoint
└── output/               # 学習成果物（Git管理外）
    ├── weights/           # best_fold0.pth → Kaggle Datasetへアップロード
    ├── logs/
    └── submission/
```

## 実行フロー

```
[1] Colab: train.py 実行
        ↓ best_fold0.pth
[2] Kaggle Dataset: 重みをアップロード
        ↓
[3] Kaggle Notebook: inference.ipynb をコンペに紐付けて実行
        ↓ submission.csv を自動生成・提出
```

## Step 1: Google Colab での学習

### 依存ライブラリのインストール

```python
!pip install timm librosa scikit-learn pyyaml
```

### データの配置

```python
# kaggle.jsonをColabにアップロード後
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c birdclef-2026
!unzip -q birdclef-2026.zip -d /content/birdclef-2026
```

### スクリプトの配置

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/birdclef-2026/experiment/exp001 /content/exp001
%cd /content/exp001
```

### 学習の実行

```python
!python train.py --config config/config.yaml
```

出力: `/content/output/exp001/weights/best_fold0.pth`

## Step 2: Kaggle Dataset への重みのアップロード

```bash
# ローカルまたはColab上で実行
kaggle datasets create -p /content/output/exp001/weights \
    --title "birdclef2026-exp001-weights" \
    --slug "birdclef2026-exp001-weights"
```

## Step 3: Kaggle Notebook での提出

1. `notebook/inference.ipynb` を Kaggle Notebook として新規作成またはアップロード
2. Input に以下を追加:
   - コンペデータ: `birdclef-2026`
   - モデル重み: `birdclef2026-exp001-weights`
3. ノートブックをコミット・実行 → 自動的に提出

## 設計メモ

- `primary_label` はeBirdコード（鳥類）またはiNat ID（非鳥類）。train_audio/ のディレクトリ名と対応
- `secondary_labels` もmulti-hotラベルに含める
- `rating` フィルタはベースラインでは無効（min_rating: 0.0）
- 推論時はtest_soundscapeを5秒スライドウィンドウで処理（inference.ipynb参照）
