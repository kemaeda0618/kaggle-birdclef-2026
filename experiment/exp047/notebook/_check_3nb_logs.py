"""Get logs from 3 NB runs to check actual file counts."""
import json, os, tempfile
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NBS = [
    ("maekeso/birdclef2026-exp047-bc-extract", "BC"),
    ("maekeso/birdclef2026-exp047-inat-extract", "iNat"),
    ("maekeso/birdclef2026-exp047-anuraset-extract", "AnuraSet"),
]

import re
for nb_slug, name in NBS:
    out_dir = Path(f"/tmp/{name}_log")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        api.kernels_output(nb_slug, str(out_dir))
        log_files = list(out_dir.rglob("*.log"))
        if log_files:
            text = log_files[0].read_text(encoding="utf-8", errors="ignore")
            # Grep key lines
            print(f"\n=== {name} ===")
            for line in text.split("\n"):
                if any(k in line for k in ["Copied", "copied", "Total", "by source", "URL:", "Match to BC2026", "match species", "Storage", "GB", "matched", "files", "error", "ERROR"]):
                    # Clean JSON wrapper
                    if '"data":"' in line:
                        m = re.search(r'"data":"([^"]+)"', line)
                        if m:
                            line = m.group(1)
                    line = line.strip()
                    if line and len(line) < 300:
                        print(f"  {line}")
    except Exception as e:
        print(f"\n=== {name} ERR: {str(e)[:120]}")
