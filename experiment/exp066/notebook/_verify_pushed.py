"""Verify pushed Kaggle kernel has the fix."""
import json, os, sys, io, tempfile
from pathlib import Path
if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

with tempfile.TemporaryDirectory() as td:
    api.kernels_pull("maekeso/birdclef2026-exp066-train", path=td, quiet=True)
    nb_files = list(Path(td).glob("*.ipynb"))
    print(f"Files: {[f.name for f in nb_files]}")
    if not nb_files:
        print("ERROR: no ipynb found")
    else:
        nb = json.load(open(nb_files[0], encoding="utf-8"))
        # Search for fix markers
        for c in nb["cells"]:
            if c.get("cell_type") == "code":
                src = "".join(c["source"])
                if "_to_sec" in src or "HH:MM:SS" in src:
                    print(f"  ✓ FOUND fix in cell id={c.get('id', '?')}")
                if "WeightedRandomSampler" in src:
                    print(f"  ✓ FOUND WeightedRandomSampler in cell id={c.get('id', '?')}")
                if "filename, start, end, primary_label" in src or "annotation format" in src:
                    print(f"  ✓ FOUND new SC schema comment in cell id={c.get('id', '?')}")
        # Total cells
        print(f"\\nTotal cells: {len(nb['cells'])}")
