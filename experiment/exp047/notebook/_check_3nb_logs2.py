"""Get logs from 3 NB runs - simpler version."""
import json, os, re
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
for nb, name in NBS:
    out = Path(f"/tmp/{name}_log")
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n== {name} ==", flush=True)
    try:
        api.kernels_output(nb, str(out))
    except Exception as e:
        print(f"  DL err: {e}", flush=True); continue
    for f in out.rglob("*.log"):
        t = f.read_text(encoding="utf-8", errors="ignore")
        for line in t.split("\n"):
            if any(k in line for k in ["Copied","copied","Total","URL:","BC2026 match","species","Storage","iNat:","AnuraSet","Match","GB","by source","unique","Dataset","error","ERROR","RuntimeError"]):
                m = re.search(r'"data":"([^"]+)"', line)
                if m:
                    line = m.group(1)
                line = line.strip().replace("\\n", "").replace("\\u2713", "")
                if line and len(line) < 250:
                    print(line, flush=True)
        break
