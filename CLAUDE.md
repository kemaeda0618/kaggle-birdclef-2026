# BirdCLEF+ 2026

## コンペ概要

- **主催**: Cornell Lab of Ornithology
- **テーマ**: 南米パンタナールにおける音響種識別 (Acoustic Species Identification in the Pantanal)
- **賞金**: $50,000
- **締切**: 2026-06-03
- **評価指標**: BirdCLEF ROC AUC（カスタムメトリクス）
- **提出方式**: Kernels Only（コード提出、ノートブックで推論を実行、CPU 90分制限）
- **1日最大提出数**: 5回
- **最大チーム人数**: 5人

## タスク

5秒ごとにセグメント化されたサウンドスケープ音声から、234種の生物（鳥類・昆虫・両生類・爬虫類など）の存在確率を予測する多ラベル分類問題。

## 評価指標

- **BirdCLEF ROC AUC**: 各種のROC AUCを算出し、その平均を取る
- row_id は `{ファイル名}_{秒数}` 形式（例: `BC2026_Test_0001_S05_20250227_010002_5`）
- 提出は各 row_id × 234種の確率値

## データ構造

```
birdclef-2026/
├── CLAUDE.md
├── train.csv              # 学習データメタデータ（35,549行）
├── taxonomy.csv           # 種の分類情報（234種）
├── sample_submission.csv  # 提出フォーマットサンプル
├── recording_location.txt # 録音場所情報（パンタナール、ブラジル）
├── train_audio/           # 学習音声（.ogg形式、約46,207ファイル、~15GB）
│   └── {inat_taxon_id}/   # 種ごとのディレクトリ（206ディレクトリ）
│       └── *.ogg          # iNat or XC プレフィックスのファイル
├── train_soundscapes/     # 学習用サウンドスケープ音声（.ogg形式、10,658ファイル）
│   └── *.ogg              # パンタナール現地録音のサウンドスケープ
├── train_soundscapes_labels.csv  # train_soundscapesの部分ラベル（疑似ラベル生成の参照用）
├── test_soundscapes/      # テスト音声（評価時のみ自動配置、ローカルは空）
│   └── readme.txt
└── experiment/            # 実験管理ディレクトリ
    └── exp001/            # 実験単位のディレクトリ（exp002, exp003... と増やす）
        ├── train.py       # 学習スクリプト（Google Colabで実行）
        ├── notebook/      # Kaggle提出用ノートブック
        │   └── inference.ipynb  # 推論・提出ノートブック（Kaggle上で実行）
        ├── config/        # 設定ファイル
        │   └── config.yaml    # モデル・学習ハイパーパラメータ
        ├── src/           # モジュール群
        │   ├── dataset.py     # Dataset / DataLoader 定義
        │   ├── model.py       # モデルアーキテクチャ定義
        │   └── utils.py       # 共通ユーティリティ
        └── output/        # 学習成果物（Git管理外）
            ├── weights/       # モデルの重み（Kaggle Datasetにもアップロード）
            ├── logs/          # 学習ログ
            └── submission/    # 提出用CSV（ローカル確認用）
```

### train.csv カラム

| カラム | 説明 |
|--------|------|
| primary_label | 主ラベル（inat_taxon_id） |
| secondary_labels | 副ラベル（リスト） |
| type | 音声タイプ |
| latitude / longitude | 録音位置 |
| scientific_name | 学名 |
| common_name | 一般名 |
| class_name | 分類クラス（Aves, Insecta, Amphibia, Reptilia等） |
| inat_taxon_id | iNaturalist 種ID |
| author / license / rating | メタデータ |
| url | 音源URL |
| filename | `{taxon_id}/{ファイル名}.ogg` |
| collection | データ収集元（iNat or XC） |

### taxonomy.csv カラム

`primary_label, inat_taxon_id, scientific_name, common_name, class_name`（234種）

### sample_submission.csv フォーマット

- `row_id` + 234種のカラム（確率値）
- row_id 形式: `{soundscape_filename}_{end_time_seconds}`

## 録音場所

パンタナール（Pantanal）、マトグロッソドスル州、ブラジル、南米
緯度: -16.5 〜 -21.6 / 経度: -55.9 〜 -57.6

## 開発ワークフロー

1. **Claude Code**（ローカル）で `experiment/expXXX/notebook/` に .ipynb を作成・編集
2. **Kaggle MCP**（`mcp__kaggle__save_notebook`）でKaggle Notebookにpush & 実行
3. 学習NB の出力（重み）を **Kaggle Dataset** として登録
4. 提出NB を同様にpush & 実行してコンペに提出

```
[ローカル編集] → [save_notebook でKaggleにpush] → [Kaggle上で自動実行]
  Claude Code         Kaggle MCP                GPU/CPU Notebook
```

### Kaggle NB push ルール

- **ノートブックを作成・更新したら、必ず Kaggle NB に push すること**
- ローカルの .ipynb ファイルもそのまま残す（ローカル＝正、Kaggle＝pushされたコピー）

#### push 方式（2026-04-20 確立、これを既定とする）

`kaggle` Python SDK の `KaggleApi.kernels_push()` を使う。KGAT_ トークンは **`~/.kaggle/kaggle.json` の `key` フィールド**に保存されており、push スクリプトがそこから自動取得して `KAGGLE_API_TOKEN` 環境変数に設定する。

```powershell
$env:PYTHONUTF8=1; python experiment/expXXX/notebook/push_xxx.py
```

スクリプトテンプレ:
```python
import json, os, io, sys, tempfile, shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Windows cp932 回避: PYTHONUTF8=1 未設定なら re-exec
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# KGAT_ トークンを ~/.kaggle/kaggle.json の key フィールドから自動取得
if not os.environ.get("KAGGLE_API_TOKEN"):
    _kgat = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    os.environ["KAGGLE_API_TOKEN"] = _kgat

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB = Path(__file__).with_name("xxx.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-expXXX-xxx"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": "BirdCLEF2026 expXXX xxx",
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,           # 学習NBは True、推論NBは False
        "enable_internet": True,       # 学習NB は True、★ 推論/提出 NB は必ず False
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [],         # 必要なら "user/dataset-slug" を追加
        "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", r.url, "Version:", r.version_number)
```

- 初回も再 push も同じスクリプトで OK（`id` が同じならバージョン更新、新しいSlugなら新規作成）
- 認証注意点:
  - **`~/.kaggle/kaggle.json` の `key` フィールドに `KGAT_` で始まる Bearer token を書く**（このプロジェクトの現在の構成。上記テンプレが自動読み込み）
  - SDK は `kaggle.json` を直接参照すると Basic auth で送って 401 になる。必ず `KAGGLE_API_TOKEN` env var 経由で渡すこと
- Windows では `PYTHONUTF8=1` を付けるか、スクリプト内で `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")` を設定する（cp932 デコードエラー回避）
- `mcp__kaggle__save_notebook` も使えるが、認証/MCP 接続が不安定なときの保険として上記 SDK 方式を既定にする
- **REST API push (requests.post) は新規 kernel 作成不可** — id フィールドが numeric integer 必須で "Could not convert string to integer" エラーになる。新規作成は SDK 一択
- **Slug は title から派生する (重要)** — `id` で `username/slug` を指定しても、Kaggle は title を slugify して URL slug を確定。title と slug が不一致だと "Your kernel title does not resolve to the specified id" 警告が出て title 由来が採用される。**意図した slug にしたい場合、title をそのまま slug 形式で書く** (例: title=`"BirdCLEF2026 exp011 train Phase1 SED"` → slug=`birdclef2026-exp011-train-phase1-sed`)
- **slug は一度確定すると変更不可**、削除後もソフト予約が残る — 命名は初回 push で確定させる

### Internet 設定 (重要)

- **推論/提出 NB (submission) は必ず `"enable_internet": False`** — Kernels Only コンペの提出要件、internet=True の NB は submit fail or 不可
- **学習 NB のみ `"enable_internet": True`** — pip install, HF download 等で必要
- push スクリプト template で submit NB 用の `False` を明示すること。デフォルト True のままにしない
- 例: `python experiment/expXXX/notebook/_push_nb_xxx.py` で submit NB を push する際は metadata の `enable_internet` を必ず確認
- 既存 submit NB を internet=True で push してしまった場合は metadata 修正して再 push 必要

### GPU アクセラレータ設定

- **学習用NB（train）は必ず T4x2 (`NvidiaTeslaT4`) を指定すること**
- **`enable_gpu` (snake_case) / `enableGpu` (camelCase) は絶対に含めない**。`True` 指定すると P100 強制になる。`machine_shape` のみで指定する
- 推論/提出用NB（submission）は CPU のみ（`machine_shape` を指定しない、`enable_gpu` も含めない）
- 有効な `machine_shape` 値: `NvidiaTeslaT4`（T4x2）, `NvidiaTeslaP100`（P100）, `Tpu1VmV38`（TPU）

#### SDK push 時 (kernel-metadata.json) の指定方法

CLI/公式 docs には未掲載だが、SDK は `machine_shape` フィールドを `KernelPushRequest` に pass-through するため **kernel-metadata.json に直接書ける**:

```python
# T4x2 学習 NB
meta = {
    "id": f"{USER}/{SLUG}",
    "title": "...",
    ...
    "machine_shape": "NvidiaTeslaT4",   # ← snake_case で書く (SDK kernel-metadata.json 規約)
    "enable_internet": True,
    # enable_gpu は書かない
}

# CPU 推論 NB
meta = {
    ...
    # machine_shape も enable_gpu も書かない
}
```

#### REST API / MCP save_notebook 時

- REST API payload (camelCase): `{'machineShape': 'NvidiaTeslaT4', 'enableInternet': True}` （`enableGpu` は含めない）
- `mcp__kaggle__save_notebook`: `machineShape: "NvidiaTeslaT4"` パラメータを指定

> ローカル環境はWindows（コード編集のみ）。学習・推論はKaggle Notebookで実行。

## コーディングルール

- **変数の依存関係チェック**: コードを生成・移植する際は、出力前に全変数が定義済みか確認すること。特に既存ノートブックからセルをコピーする場合、元のセルで定義されていた変数（CFG キー、中間変数、マッピング配列等）が移植先でも利用可能かを必ず検証する。不足があれば定義コードも一緒に追加する。

## ルール上の注意点

- **Kernels Only**: Kaggle ノートブック上で推論コードを動かして提出する
- test_soundscapes はローカルには存在せず、Kaggle 評価環境でのみ配置される
- 外部データ使用は許可されているが、公開必須（コンペルールで確認）
- 1日5回の提出制限あり

## Sub 運用ルール (重要)

- **★ Sub は毎日 5 回 reset、温存意味なし** — その日の残り sub は必ず使い切る (sub 残しは情報収集機会の損失)
- **「明日のために save」は禁止** — 明日も 5 sub 来るので今日の残り使い切るのが strictly dominant
- **Sub 価値判断は "情報量 × 残り日数"** — 例えば diagnostic sub (paradigm 確認) も valid な選択肢
- **その日の sub 残数が 0 になるまで考案して使う** — 1 残ったら "M3' fold 0 で robustness 確認" 等の next-best use を立てる
