# exp012 Colab Pro 学習手順 (Drive 同期版)

`C:\Users\maeke\work\kaggle\birdclef-2026` が Drive 同期されている前提。
Tucker datasets は **Colab 内で kaggle CLI 経由で Drive に DL** する方式 (ローカル PowerShell が不要)。

## ファイル

| ファイル | 役割 |
|---|---|
| `setup_tucker_cache_local.ps1` | (任意) ローカル PC で Tucker dataset を DL (Colab DL を省略したい場合のみ) |
| `_gen_nb_train_colab_drive.py` | NB 生成スクリプト (修正したらこれを再実行) |
| `nb_train_colab_drive.ipynb` | **Colab で開く NB 本体** |

## 推奨手順 (Colab だけで完結)

### 1. Colab で notebook を開く

1. https://colab.research.google.com を開く
2. **ファイル → Google ドライブ → ノートブックを開く**
3. `work/kaggle/birdclef-2026/experiment/exp012/notebook/nb_train_colab_drive.ipynb` を選択

### 2. CPU runtime でデータ準備 (units 節約)

```
ランタイム → ランタイムのタイプを変更 → CPU
```

実行するセル:
- Cell 1 (Setup): Drive マウント、PROJECT_ROOT 自動検出、kaggle.json 設定
- Cell 2 (DL Tucker → Drive): `/content/` で kaggle CLI DL → Drive に rsync
  - 所要 30-60 分
  - 一度成功すれば次回以降 skip
- Cell 3 (Drive → Local): `/content/data/` に copy

CPU runtime は units 消費が極小なので長時間放置 OK。

### 3. A100 runtime に切替えて学習

```
ランタイム → ランタイムのタイプを変更 → A100
→ 接続後、Cell 1 から再実行
  - Cell 1 (Setup) ← Drive 再マウント
  - Cell 2 (DL) ← skip される (Drive にあるため)
  - Cell 3 (Drive→Local) ← rsync ~5-10分
- Cell 4 以降を Run All (学習)
```

**Note**: A100 切替で `/content/` は消えるが Drive のデータは残るので、Cell 3 の rsync で復元される。

学習所要: A100 で 4-8 時間 (5 fold × 25 epoch)。

### 4. 完了後

最後の Cell が `maekeso/birdclef2026-exp012-sed-onnx` を Kaggle Dataset として作成。
これを Kaggle 推論NB の `dataset_sources` に指定して提出 → LB ~0.917 期待。

## 代替: ローカル PC で先に Tucker datasets を DL

もし Colab DL が遅い/不安定なら、ローカルで先に DL → Drive 同期：

```powershell
cd C:\Users\maeke\work\kaggle\birdclef-2026
powershell -ExecutionPolicy Bypass -File experiment\exp012\notebook\setup_tucker_cache_local.ps1
```

完了後 Drive Desktop で同期完了を待つ (数時間〜半日)。
Colab で notebook を開いた時、Cell 2 は skip される。

## トラブルシューティング

### Cell 1 で `birdclef-2026 not found on Drive`
→ プロジェクトフォルダが Drive に同期されていない、または My Drive 直下にない。
`!find /content/drive/MyDrive -maxdepth 6 -name "birdclef-2026" -type d` で実パス確認。

### Cell 2 で `kaggle.json` 関連エラー
→ `/content/drive/MyDrive/kaggle.json` を置く (KGAT_ で始まる token を `key` フィールドに)。

### Cell 2 で 401 エラー
→ kaggle.json の token が古い。Kaggle UI で再生成。

### 学習中にランタイム切断
→ Drive にチェックポイント保存される (Cell の patch 部分)。再接続で Cell 1 から実行 → 自動 resume。

### A100 が取れない
→ V100 / L4 でも動く。Cell 5 (config) の BATCH を 32 に下げる。
