"""Switch from kaggle CLI subprocess to Kaggle SDK direct call.

Issue: CLI subprocess may not honor KAGGLE_API_TOKEN env var (uses ~/.kaggle/kaggle.json
with Basic auth, gets 403 with KGAT_ token).

Fix: Use KaggleApi.dataset_download_files() directly. SDK reads env var correctly.
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp052\notebook\nb_stage1_xc_pretrain.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

NEW_DL_DATA = '''# ============================================================
# Cell 2: Download XC Datasets via SDK (NOT CLI subprocess)
# ============================================================
# Use kaggle SDK directly to avoid CLI subprocess env var issues.
# SDK respects KAGGLE_API_TOKEN env var for Bearer tokens (KGAT_).

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
print("OK Kaggle SDK authenticated")

DATA_ROOT = Path("/content/xc_data")
DATA_ROOT.mkdir(exist_ok=True)
part1_dir = DATA_ROOT / "part1"
part2_dir = DATA_ROOT / "part2"

def sdk_download(slug, target_dir, expected_subdir):
    target_dir.mkdir(exist_ok=True, parents=True)
    expected_meta = target_dir / expected_subdir / "_metadata.csv"
    if expected_meta.exists():
        print(f"  Already exists: {expected_meta}")
        return True
    print(f"\\nDownloading {slug} -> {target_dir} (this takes ~5-10 min for 17 GB)")
    try:
        api.dataset_download_files(slug, path=str(target_dir), unzip=True, quiet=False)
    except Exception as e:
        print(f"FAIL: {str(e)[:300]}")
        raise
    if expected_meta.exists():
        print(f"  OK metadata found: {expected_meta}")
        return True
    print(f"  WARN: expected metadata not found at {expected_meta}, listing dir:")
    for p in target_dir.rglob("*.csv"):
        print(f"    {p}")
    raise FileNotFoundError(f"Expected {expected_meta} after extract")

sdk_download("maekeso/birdclef2026-xc-api-dl-part1", part1_dir, "xc_api")
sdk_download("maekeso/birdclef2026-xc-api-dl-part2", part2_dir, "xc_api_part2")

# Verify
import pandas as pd
p1_meta = part1_dir / "xc_api" / "_metadata.csv"
p2_meta = part2_dir / "xc_api_part2" / "_metadata.csv"
p1_df = pd.read_csv(p1_meta)
p2_df = pd.read_csv(p2_meta)
print(f"\\n=== Datasets ready ===")
print(f"Part 1: {len(p1_df)} files, {p1_df['scientific_name'].nunique()} species")
print(f"Part 2: {len(p2_df)} files, {p2_df['scientific_name'].nunique()} species")

# Sample audio file check
sample_mp3 = next(part1_dir.rglob("*.mp3"), None)
if sample_mp3:
    print(f"\\nSample mp3: {sample_mp3} ({sample_mp3.stat().st_size/1e6:.1f} MB)")
'''

for c in nb["cells"]:
    if c.get("id") == "dl_data":
        c["source"] = NEW_DL_DATA.splitlines(keepends=True)
        print("[dl_data] OK switched to Kaggle SDK direct call")
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB}")
