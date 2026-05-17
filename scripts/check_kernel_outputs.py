"""mel cache 系 NB の output 状況を確認するスクリプト。

各 NB の最新 version の output ファイル一覧を取得し、Kaggle Dataset 化されているかを把握する。
"""
import os
import sys

assert os.environ.get("KAGGLE_API_TOKEN"), "Set KAGGLE_API_TOKEN env var (KGAT_...)"

from kaggle.api.kaggle_api_extended import KaggleApi

USER = "maekeso"

DEFAULT_SLUGS = [
    "birdclef2026-mel-cache-bc2021-256",
    "birdclef2026-mel-cache-bc2022-256",
    "birdclef2026-mel-cache-bc2023-256",
    "birdclef2026-mel-cache-bc2024-256-v2",
    "birdclef2026-mel-cache-bc2025-256-v2",
    "birdclef2026-mel-cache-anuraset-256",
    "birdclef2026-mel-cache-inat-256",
]

slugs = sys.argv[1:] or DEFAULT_SLUGS

api = KaggleApi()
api.authenticate()

for slug in slugs:
    print(f"\n=== {slug} ===")
    try:
        files = api.kernels_output(USER, slug, path=None, force=False, quiet=True, list_files_only=True)
        if isinstance(files, list):
            print(f"  files: {len(files)}")
            for f in files[:10]:
                name = getattr(f, "name", str(f))
                size = getattr(f, "size", "?")
                print(f"    {name} ({size})")
        else:
            print(f"  result: {files}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:200]}")

# また、対応する Dataset の存在も確認
print("\n=== Datasets check ===")
dataset_candidates = [s.replace("birdclef2026-", "birdclef2026-ds-") for s in slugs]
for slug in slugs + dataset_candidates:
    try:
        meta = api.dataset_metadata(f"{USER}/{slug}", path=None)
        print(f"  ✓ {slug} : dataset exists")
    except Exception as e:
        msg = str(e)[:80]
        if "404" in msg or "Not found" in msg.lower():
            pass  # silent
        else:
            print(f"  ? {slug} : {msg}")
