"""Verify infer NB attached inputs after v2 push.

[[feedback_kaggle_v2_input_drop]]: SDK kernels_push v2+ で input sources が drop されるバグの確認。
kernels_pull で metadata を取得して dataset_sources / competition_sources を表示する。
"""
import json, os, tempfile
from pathlib import Path

if not os.environ.get("KAGGLE_API_TOKEN"):
    key = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

SLUG = "maekeso/birdclef2026-exp017-infer"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    try:
        api.kernels_pull(SLUG, str(td), metadata=True)
        print("Pulled metadata to:", td)
        for f in sorted(td.iterdir()):
            print(f"  {f.name}  {f.stat().st_size}B")
            if f.suffix == ".json":
                print("\n--- " + f.name + " ---")
                print(f.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"kernels_pull error: {type(e).__name__}: {str(e)[:400]}")
