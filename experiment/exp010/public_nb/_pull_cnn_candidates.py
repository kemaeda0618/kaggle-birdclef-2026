"""Pull metadata of CNN-related 0.93+ candidates."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

candidates = [
    "mtoshidesu/0-937-birdclef-2026-true-end-to-end-pipeline",
    "mtoshidesu/0-934-birdclef-2026-clean-visual-pipeline",
    "needless090/birdclef-2026-iter-pseudo-perch-sed-lb-0-935-ft",
    "needless090/birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s",
    "yuyajk/birdclef-2026-0-943-onnx-perch-sequence-sed-submit",
    "mattiaangeli/birdclef-2026-0-943-better-blend",
    "kojimar/0-949-lb-birdclef-2026-prior-axis-rank-fusion",
    "ttyn4519/perch-task236-task233-snowflake-sed-w003",
]
base = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi")
for slug in candidates:
    short = slug.replace("/", "_")
    out = base / short
    out.mkdir(parents=True, exist_ok=True)
    try:
        api.kernels_pull(slug, path=str(out), metadata=True)
        # Read metadata
        meta_file = out / "kernel-metadata.json"
        if meta_file.exists():
            m = json.loads(meta_file.read_text(encoding="utf-8"))
            print(f"\n=== {slug} ===")
            print(f"  title:           {m.get('title','')[:80]}")
            print(f"  dataset_sources: {m.get('dataset_sources', [])}")
            print(f"  kernel_sources:  {m.get('kernel_sources', [])}")
            print(f"  model_sources:   {m.get('model_sources', [])}")
        else:
            print(f"  no meta")
    except Exception as e:
        print(f"FAIL {slug}: {type(e).__name__}: {str(e)[:120]}")
