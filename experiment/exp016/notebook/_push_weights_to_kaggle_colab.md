# exp016 重み Kaggle Dataset 化 (Colab で実行)

`maekeso/birdclef2026-exp016-weights` という Kaggle Dataset として
Drive 上の `ckpt_best_ns22.pth` 等を upload する。

Colab で新規セルを 1 つ作り、以下を貼り付けて Run。
(Drive mount + kaggle.json は train NB の Cell 1 でセット済なのでそのまま動く)

> **注**: Kaggle API は Dataset 非存在時に **403 を返すクセ**があるため、
> 事前に `dataset_view` で存在チェックして分岐する設計にしている。

```python
# ============================================================
# Push exp016 R1 weights to Kaggle Dataset
# ============================================================
import shutil, tempfile, json, time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi(); api.authenticate()

DRIVE_INPUT_DIR = Path("/content/drive/MyDrive/kaggle/birdclef2026")
DRIVE_CKPT_DIR  = DRIVE_INPUT_DIR / "output" / "exp016" / "r1"

USER  = "maekeso"
SLUG  = "birdclef2026-exp016-weights"
TITLE = "birdclef2026 exp016 weights"

assert DRIVE_CKPT_DIR.exists(), f"Drive ckpt dir missing: {DRIVE_CKPT_DIR}"

TARGETS = ["ckpt_best_ns22.pth", "ckpt_best_macro.pth", "ckpt_latest.pth", "history.json"]

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    for fn in TARGETS:
        src = DRIVE_CKPT_DIR / fn
        if not src.exists():
            print(f"  skip (missing): {fn}")
            continue
        shutil.copy2(str(src), str(td / fn))
        print(f"  staged: {fn}  ({src.stat().st_size/1e6:.1f}MB)")

    meta = {
        "title": TITLE,
        "id": f"{USER}/{SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    version_notes = "R1 best_ns22=0.9207"
    uploaded = False

    # 事前に存在チェック (Kaggle API は非存在時に 403 を返すクセがある)
    try:
        api.dataset_view(f"{USER}/{SLUG}")
        dataset_exists = True
        print(f"\nDataset {USER}/{SLUG} exists → create new VERSION")
    except Exception:
        dataset_exists = False
        print(f"\nDataset {USER}/{SLUG} not found → CREATE new dataset")

    if dataset_exists:
        try:
            api.dataset_create_version(folder=str(td),
                                        version_notes=version_notes,
                                        dir_mode="zip", quiet=False)
            print(f"\nOK Uploaded (new version) — {version_notes}")
            uploaded = True
        except Exception as e:
            print(f"  dataset_create_version error: {str(e)[:400]}")
    else:
        try:
            api.dataset_create_new(folder=str(td), public=False, dir_mode="zip", quiet=False)
            print(f"\nOK Created (first time)")
            uploaded = True
        except Exception as e:
            print(f"  dataset_create_new error: {str(e)[:400]}")

    assert uploaded, "Upload failed — check logs above"
    print(f"\nDataset URL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
```

## このあと

ローカルで:
```powershell
$env:PYTHONUTF8=1; python experiment/exp016/notebook/_push_nb_infer.py
```
→ Kaggle 上で `birdclef2026-exp016-infer` notebook が作成される。

Kaggle Web UI から:
- exp016-infer notebook を開く → 「Run All」または「Save Version」
- もしくは Submit to competition で提出 (1日5枠の1つを消費)
