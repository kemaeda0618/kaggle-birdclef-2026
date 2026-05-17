"""Kaggle kernel の最新 version の実行ステータスを確認するスクリプト。

Usage:
    PYTHONUTF8=1 KAGGLE_API_TOKEN=KGAT_xxxx python scripts/check_kernel_status.py [slug...]

引数なしでデフォルトの mel cache 系 NB をまとめて確認。
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

print(f"{'slug':<55} {'status':<15} {'failure':<60}")
print("-" * 130)
for slug in slugs:
    try:
        st = api.kernels_status(f"{USER}/{slug}")
        status = getattr(st, "status", "?")
        msg = getattr(st, "failure_message", "") or ""
    except Exception as e:
        status = "ERROR"
        msg = str(e)[:60]
    print(f"{slug:<55} {status:<15} {msg[:60]}")
