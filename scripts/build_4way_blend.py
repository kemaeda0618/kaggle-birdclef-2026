"""4-way post-process blend of existing submission CSVs.

Inputs (downloaded from Kaggle submission CSVs):
- exp019 blend (LB 0.947) — 主軸
- exp028 mirror (LB 0.919) — Sonotype mirror diversity
- exp029 l1 (LB 0.923) — l1 backbone diversity
- exp020 R2 5fold (LB 0.915) — 5-fold ensemble diversity

Output:
- experiment/exp030/submission/submission_4way_diversity.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCES = [
    # (kernel_slug, weight, label)
    ("maekeso/birdclef2026-exp019-blend-w35-40-25",      0.85, "exp019"),
    ("maekeso/birdclef2026-exp028-protossm-mirror",       0.05, "exp028"),
    ("maekeso/birdclef2026-exp029-r3-single-l1-infer",    0.05, "exp029"),
    ("maekeso/birdclef2026-exp020-r2-5fold-infer-onnx",   0.05, "exp020r2"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "experiment" / "exp030" / "submission" / "submission_4way_diversity.csv"))
    ap.add_argument("--download", action="store_true", help="DL submission CSVs from Kaggle NB outputs")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = ROOT / "experiment" / "exp030" / "_csv_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Setup Kaggle API if downloading
    if args.download:
        os.environ.setdefault("PYTHONUTF8", "1")
        os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()

    weights_sum = sum(w for _, w, _ in DEFAULT_SOURCES)
    assert abs(weights_sum - 1.0) < 1e-6, f"Weights sum != 1.0: {weights_sum}"

    print("Sources:")
    for slug, w, lbl in DEFAULT_SOURCES:
        print(f"  [{w:.2f}] {lbl:<10s}  {slug}")

    # DL each kernel's submission.csv
    csv_paths = []
    for slug, w, lbl in DEFAULT_SOURCES:
        local = cache_dir / f"{lbl}_submission.csv"
        if not local.exists() and args.download:
            print(f"\nDownloading {slug} -> {local}...")
            d = cache_dir / f"_{lbl}_dump"
            d.mkdir(exist_ok=True)
            api.kernels_output(slug, path=str(d), quiet=False, force=True)
            sub_candidates = list(d.glob("submission*.csv"))
            if not sub_candidates:
                raise FileNotFoundError(f"No submission*.csv in {slug} output")
            sub_candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
            sub = sub_candidates[0]
            sub.rename(local)
            print(f"  -> {local} ({local.stat().st_size/1e6:.1f} MB)")
        csv_paths.append((local, w, lbl))

    # Verify all exist
    for p, w, lbl in csv_paths:
        if not p.exists():
            print(f"\n!! Missing {lbl}: {p}")
            print("   Run with --download or place CSV manually")
            sys.exit(1)

    # Load + align by row_id
    print("\nLoading CSVs...")
    dfs = []
    for p, w, lbl in csv_paths:
        df = pd.read_csv(p)
        print(f"  {lbl:<10s}  rows={len(df):>7d}  cols={df.shape[1]}  weight={w}")
        dfs.append((df, w, lbl))

    # Verify schema agreement
    base_cols = dfs[0][0].columns.tolist()
    base_rowids = set(dfs[0][0]["row_id"])
    for df, w, lbl in dfs[1:]:
        assert df.columns.tolist() == base_cols, f"{lbl} columns mismatch"
        assert set(df["row_id"]) == base_rowids, f"{lbl} row_id mismatch"

    # Sort all by row_id same way
    species_cols = [c for c in base_cols if c != "row_id"]
    base_df = dfs[0][0].sort_values("row_id").reset_index(drop=True)
    result = base_df.copy()
    result[species_cols] = 0.0

    for df, w, lbl in dfs:
        df_sorted = df.sort_values("row_id").reset_index(drop=True)
        assert (df_sorted["row_id"].values == base_df["row_id"].values).all()
        result[species_cols] += w * df_sorted[species_cols].values

    # Sanity
    print(f"\nResult: {len(result)} rows x {len(species_cols)} species")
    print(f"  min={result[species_cols].values.min():.5f}  max={result[species_cols].values.max():.5f}")
    print(f"  mean={result[species_cols].values.mean():.5f}")

    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
