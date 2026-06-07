"""Verify AnuraSet Dataset existence via multiple API methods."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Try multiple slug variants
SLUGS = [
    "maekeso/birdclef2026-exp047-anuraset-extracted",
    "maekeso/birdclef2026-exp047-anuraset-extract",
]
for slug in SLUGS:
    print(f"\n=== {slug} ===")
    try:
        files = api.dataset_list_files(slug)
        print(f"  files: {len(files.files)}")
        for f in files.files[:5]:
            print(f"    {f.name}")
    except Exception as e:
        print(f"  list_files err: {str(e)[:200]}")
    try:
        info = api.dataset_metadata(slug, "/tmp/anuraset_check")
        print(f"  metadata OK: {info}")
    except Exception as e:
        print(f"  metadata err: {str(e)[:200]}")

# List my datasets
print(f"\n=== My recent datasets ===")
try:
    datasets = api.dataset_list(user="maekeso", max_size=200)
    print(f"Total: {len(list(datasets))}")
    # Re-list (consumed)
    datasets = api.dataset_list(user="maekeso", max_size=200)
    for d in datasets:
        ref = getattr(d, 'ref', '?')
        if "anura" in ref.lower() or "exp047" in ref.lower():
            print(f"  ✓ {ref}")
except Exception as e:
    print(f"  list err: {str(e)[:200]}")

# Try kernels_output for AnuraSet NB to see actual upload details
print(f"\n=== AnuraSet NB upload log tail ===")
import re
try:
    out_dir = Path("/tmp/AnuraSet_log")
    log_files = list(out_dir.rglob("*.log"))
    if log_files:
        text = log_files[0].read_text(encoding="utf-8", errors="ignore")
        for line in text.split("\n")[-100:]:
            if any(k in line.lower() for k in ["upload", "error", "fail", "url", "dataset", "complete"]):
                m = re.search(r'"data":"([^"]+)"', line)
                if m:
                    line = m.group(1)
                line = line.strip().replace("\\n", "")
                if line and len(line) < 250:
                    print(f"  {line}")
except Exception as e:
    print(f"  err: {e}")
