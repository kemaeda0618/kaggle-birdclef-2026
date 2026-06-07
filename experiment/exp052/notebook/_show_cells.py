"""Show current state of install + dl_data cells."""
import json
from pathlib import Path
NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp052\notebook\nb_stage1_xc_pretrain.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

for cell_id in ["install", "dl_data"]:
    print(f"\n{'='*60}\n[{cell_id}]\n{'='*60}")
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            print("".join(c["source"]))
            break
