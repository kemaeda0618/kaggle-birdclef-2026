"""Fetch exp085 kernel output (script 322636523) and audit for data processing mistakes."""
import json, os, tempfile, re
from pathlib import Path

if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp085-4model-blend"
with tempfile.TemporaryDirectory() as td:
    api.kernels_output(slug, path=td, quiet=False)
    td_path = Path(td)
    print(f"\n=== Files ===")
    for f in sorted(td_path.rglob("*")):
        if f.is_file():
            print(f"  {f.name}  {f.stat().st_size/1024:.1f}KB")

    # Read log
    log_files = list(td_path.rglob("*.log"))
    if not log_files:
        print("No log file found")
        raise SystemExit(0)
    log = log_files[0]
    content = log.read_text(encoding="utf-8", errors="replace")

    # Extract all stdout/stderr lines
    lines = content.splitlines()
    print(f"\nTotal log lines: {len(lines)}")

    # Key checks
    print(f"\n=== Key markers ===")
    markers_to_find = [
        "ONNX Runtime installed",
        "TF 2.20 installed",
        "OpenVINO installed",
        "ONNX Runtime available",
        "OpenVINO available",
        "audio_cache built",
        "SED done:",
        "e17 done:",
        "e015 done:",
        "exp015 OV",
        "exp015 OV compiled",
        "exp015 R2 OV",
        "4-way rank blend",
        "Sonotype mirror",
        "blend shape:",
        "Submission:",
        "Mean pred:",
        "Max pred:",
    ]
    for m in markers_to_find:
        found = False
        for ln in lines:
            if m in ln:
                # Extract message after "data":"..."
                if '"data":"' in ln:
                    try:
                        idx = ln.index('"data":"')
                        snippet = ln[idx+8:].split('\\n')[0][:200]
                        print(f"  [OK] {m}: ... {snippet}")
                    except:
                        print(f"  [OK] {m}: {ln[:200]}")
                else:
                    print(f"  [OK] {m}: {ln[:200]}")
                found = True
                break
        if not found:
            print(f"  [MISS] {m}")

    # Look for errors / warnings
    print(f"\n=== Errors / Warnings ===")
    n_err, n_warn = 0, 0
    err_keys = ["Error", "ERROR", "Traceback", "FAILED", "AssertionError"]
    for ln in lines:
        for k in err_keys:
            if k in ln and "FileNotFoundError" not in ln:  # exclude wheel not found (already fixed)
                if '"data":"' in ln:
                    try:
                        idx = ln.index('"data":"')
                        snippet = ln[idx+8:].split('\\n')[0][:200]
                        # Skip benign warnings
                        if "SyntaxWarning" in snippet or "invalid escape" in snippet:
                            continue
                        n_err += 1
                        print(f"  ERR: {snippet}")
                    except:
                        n_err += 1
                        print(f"  ERR: {ln[:200]}")
                break
    print(f"  ({n_err} non-benign error lines)")

    # Final submission stats
    print(f"\n=== Last 50 lines ===")
    for ln in lines[-50:]:
        if '"data":"' in ln:
            try:
                idx = ln.index('"data":"')
                snippet = ln[idx+8:].split('\\n')[0][:250]
                # Skip empty
                if snippet.strip() and not snippet.startswith('  '):
                    print(f"  {snippet}")
            except:
                pass

    # Check submission.csv if downloaded
    sub_files = list(td_path.rglob("submission.csv"))
    if sub_files:
        import pandas as pd
        sub = pd.read_csv(sub_files[0])
        print(f"\n=== submission.csv ===")
        print(f"  Shape: {sub.shape}")
        print(f"  Columns: row_id + {len(sub.columns)-1} species")
        print(f"  row_id unique: {sub['row_id'].nunique()}")
        print(f"  row_id null: {sub['row_id'].isna().sum()}")
        # Prediction stats (skip row_id col)
        preds = sub.iloc[:, 1:].values
        print(f"  Pred min: {preds.min():.6f}")
        print(f"  Pred max: {preds.max():.6f}")
        print(f"  Pred mean: {preds.mean():.6f}")
        print(f"  All zeros rows: {(preds.sum(axis=1) == 0).sum()}")
        print(f"  Row stats: first 3 row_ids: {sub['row_id'].head(3).tolist()}")
        print(f"  Sample submission expected rows: check via sample_submission")
