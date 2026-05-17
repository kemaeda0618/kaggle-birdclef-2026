# Google Perch v2 推論時間調査 (BirdCLEF 2026)

調査日: 2026-03-24

---

## 要約

Perch v2 をそのままCPU上で使うと制限時間 (90分) を大幅に超過する。
BirdCLEF 2025 で実証されたように、TFLite変換により約12倍の高速化が可能で、
16分程度で推論が完了する。BirdCLEF 2026 においても同じ制約・同じ解法が有効と考えられる。

---

## 1. Perch v2 の素の推論時間 (CPU)

### BirdCLEF 2025 での実測 (arxiv:2507.08236 より)

| 形式                  | 1ファイルあたり | 700ファイル合計 | 90分制限内？ |
|-----------------------|---------------|---------------|-------------|
| TensorFlow (素)       | 17.0 秒       | ~3.3 時間     | **NG** (制限の2.2倍) |
| TFLite (最適化後)     | 1.4 秒        | ~16 分        | **OK**       |

- ソース: "Distilling Spectrograms into Tokens: Fast and Lightweight Bioacoustic Classification for BirdCLEF+ 2025"
  https://arxiv.org/html/2507.08236v1
- 測定条件: BirdCLEF 2025 テストセット (705個の1分音声ファイル = 約700分相当)
- 「約10倍の高速化」と論文では述べているが、実測値の比率は 17.0 / 1.4 = 12.1倍

### Raspberry Pi での参考値 (コミュニティ報告)
- Raspberry Pi 5 (4GB): バッチサイズ5で 約1.13 clips/秒 → 1クリップ = 約0.88秒
- Raspberry Pi 4: 約2.25 秒/クリップ
- ソース: https://github.com/tphakala/birdnet-go/discussions/1110
- 注: Kaggle CPU環境とは異なるため参考値

---

## 2. BirdCLEF 2026 テストセット規模の推定

BirdCLEF 2026 の実際のテストセットファイル数は非公開 (推論時のみ配置)。
sample_submission.csv はスタブファイル (1ファイル分のみ)。

BirdCLEF 2025 と同様の規模 (~700ファイル × 1分) と仮定した場合:

| 形式              | 推定合計時間 | 90分制限内？ |
|-------------------|-------------|-------------|
| Perch TF (素)     | ~3.3 時間   | NG          |
| Perch TFLite      | ~16 分      | OK          |

BirdCLEF 2026 が 2025 より規模拡大している場合でも、TFLite であれば余裕あり。

---

## 3. 推論高速化手法の比較

### 3-A. TFLite変換 (検証済み、推奨)

- **実績**: BirdCLEF 2025 で DS@GT チームが実証 (Public ROC-AUC 0.729)
- **高速化率**: 約12倍 (17秒 → 1.4秒/ファイル)
- **仕組み**: TFモデルの計算グラフをTFLiteでコンパイル・最適化
- **実装難易度**: 低〜中 (TFLite変換APIが整備されている)
- **Kaggle Modelの対応**: `google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu` が存在
  URL: https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/perch_v2_cpu/versions/1
- **注意**: Perch v2 は TensorFlow ベースのため、PyTorchの ONNX とは別経路

### 3-B. ONNX変換

- **Perch v2 への直接適用**: 報告なし
  - Perch v2 は TensorFlow/JAX で実装されており、PyTorch ONNX エクスポートと互換性がない
  - TF→ONNX 変換 (tf2onnx) は理論上可能だが、BirdCLEF 文脈での実績報告なし
- **PyTorch CNN (EfficientNet等) への適用**: BirdCLEF 2024 で複数の参加者が実施
  - ONNX変換で 30〜200% の高速化報告 (タスク依存)
  - 参考: https://www.kaggle.com/code/vishalbakshi/birdclef24-pth-onnx-xml-inference-speed-analysis
  - 参考: https://www.kaggle.com/code/zijiangyang1116/birdclef-24-inference-with-onnx

### 3-C. OpenVINO変換

- BirdCLEF 2024 1位解法が使用 (EfficientNetベース)
- BirdCLEF 2024 の制限は120分CPU、OpenVINOで並列メルスペクトログラム計算
- Perch への適用は報告なし

---

## 4. Perch vs CNN (EfficientNet等) の推論時間比較

BirdCLEF 2025 の arxiv 論文 (2507.08236) に掲載されたモデル別速度比較:

| モデル                    | 1ファイルあたり | 700ファイル推定 | 90分制限内？ |
|---------------------------|---------------|----------------|-------------|
| Perch (TFLite)            | 1.4 秒        | ~16 分         | OK          |
| BirdSetEfficientNetB1     | 2.21 秒       | ~26 分         | OK          |
| BirdNET (1秒ステップ)     | 6.2 秒        | ~72 分         | ギリギリ    |
| STSG (スペクトグラムトークン) | 0.5 秒    | ~6 分          | OK (最速)   |
| Perch (TF, 素)            | 17.0 秒       | ~3.3 時間      | NG          |

- ソース: https://arxiv.org/html/2507.08236v1

**重要な含意**: TFLite変換後の Perch (1.4秒) は EfficientNetB1 (2.21秒) より速い。
Perch はモデル精度と速度のバランスが良い選択肢。

BirdCLEF 2024 での EfficientNet ONNX 参考値:
- EfficientVit-b0 (ONNX): 5フォールドで40分 → 約0.5秒/ファイル相当
  (BirdCLEF 2024 の制限は120分で、2026 の 90分より余裕があった点に注意)

---

## 5. BirdCLEF 2026 コンペの議論状況

### フォーラム状況
- Kaggle フォーラム (forumId: 10052152) の具体的なスレッドは検索エンジン経由では取得不可
- 推論時間・タイムアウトに関する明示的な 2026 固有の議論は確認できなかった

### 関連ノートブック (BirdCLEF 2026)
| ノートブック | 調査結果 |
|------------|---------|
| `kdmitrie/birdclef26-google-perch-starter` | Kaggle JS due のみ取得、タイミングデータ不取得 |
| `jaejohn/perch-v2-starter-train-infer` | 同上 |
| `kamongi/pantanal-distill-birdclef2026` | 同上 (更新日: 2026-03-23) |
| `yashanathaniel/birdclef-2026-perch-v2-0-908` | Public Score 0.908 を達成、詳細不明 |
| `antoinemasq/birdclef-2026-pytorch-baseline-inference` | PyTorch ベースライン、タイミング不明 |

注: Kaggle ノートブックの実行出力 (セル結果) はWebフェッチでは取得できなかった。
`yashanathaniel` の 0.908 スコアは Perch v2 使用の可能性が高い。

---

## 6. まとめと推奨アクション

### 推奨アクション (優先度順)

1. **[最優先] Perch v2 + TFLite で推論ノートブックを構築**
   - TFLite 版の Perch は Kaggle Models に既存 (`perch_v2_cpu`)
   - 推論時間: ~16分 (700ファイルの場合) → 90分制限内で安全
   - BirdCLEF 2025 での実績あり

2. **[高] `yashanathaniel/birdclef-2026-perch-v2-0-908` を参照**
   - Public 0.908 という高スコアのノートブックを Kaggle 上で直接確認し、
     TFLite 使用の有無・推論時間を確認する

3. **[中] PyTorch CNN (EfficientNet B0 等) との ensemble 検討**
   - EfficientNet B0 (PyTorch) は ONNX 変換で ~0.5秒/ファイル程度が見込める
   - Perch (TFLite) と ensemble する場合の合計推論時間:
     16分 (Perch) + 20〜30分 (EfficientNet ONNX) = ~50分以内、90分制限内に収まる見込み

4. **[低] ONNX 経由での Perch 変換**
   - TF→ONNX (tf2onnx) は技術的に可能だが、BirdCLEF での実績なし
   - TFLite で十分であれば不要

---

## 参考ソース

- [Distilling Spectrograms into Tokens (arXiv:2507.08236)](https://arxiv.org/html/2507.08236v1) - BirdCLEF 2025 Perch TFLite 速度検証の主要論文
- [Kaggle Models: perch_v2_cpu](https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/perch_v2_cpu/versions/1)
- [BirdCLEF 2024 ONNX/OpenVINO Speed Analysis](https://www.kaggle.com/code/vishalbakshi/birdclef24-pth-onnx-xml-inference-speed-analysis)
- [Google Perch GitHub](https://github.com/google-research/perch)
- [Perch 2.0 Paper (arXiv:2508.04665)](https://arxiv.org/html/2508.04665v1) - アーキテクチャ詳細 (EfficientNet-B3, 12Mパラメータ)
- [BirdNET-Go: Perch v2 CPU discussion](https://github.com/tphakala/birdnet-go/discussions/1110)
- [BirdCLEF 2026: yashanathaniel Perch v2 0.908 notebook](https://www.kaggle.com/code/yashanathaniel/birdclef-2026-perch-v2-0-908)
