import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

DS = [
    "birdclef2026-exp015-weights",
    "birdclef2026-exp017-weights",
    "birdclef2026-exp081-weights",
    "birdclef2026-exp082-weights",
    "birdclef2026-exp083-weights",
    "birdclef2026-exp084-weights",
]

for slug in DS:
    print(f"\n=== maekeso/{slug} ===")
    try:
        files = api.dataset_list_files(f"maekeso/{slug}")
        all_names = [f.name for f in files.files]
        # Categorize
        r1_files = [n for n in all_names if n.startswith("r1/")]
        r2_files = [n for n in all_names if n.startswith("r2/")]
        flat_files = [n for n in all_names if "/" not in n and "ckpt" in n.lower()]
        r2_prefix = [n for n in all_names if n.startswith("r2_")]
        r1_prefix = [n for n in all_names if n.startswith("r1_")]

        # R1 check
        r1_status = "❌"
        if r1_files:
            r1_status = f"✅ folder ({len(r1_files)} files)"
        elif flat_files:
            r1_status = f"✅ flat ({len(flat_files)} files)"

        # R2 check
        r2_status = "❌"
        if r2_files:
            r2_status = f"✅ folder ({len(r2_files)} files)"
        elif r2_prefix:
            r2_status = f"✅ flat prefix ({len(r2_prefix)} files)"

        print(f"  R1: {r1_status}")
        print(f"  R2: {r2_status}")
        # Sample listing
        for n in all_names[:8]:
            print(f"    {n}")
        if len(all_names) > 8:
            print(f"    ... ({len(all_names)-8} more)")
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {str(e)[:100]}")
