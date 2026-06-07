"""Inspect exp078 NB's Perch label mapping cells (v313_04, v313_05)."""
import json, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

nb = json.loads(Path("experiment/exp078/notebook/nb_blend_v313_swap.ipynb").read_text(encoding="utf-8"))
for cid in ["v313_04", "v313_05"]:
    for c in nb["cells"]:
        if c.get("id") == cid:
            src = "".join(c["source"])
            print(f"\n{'='*70}\n# CELL {cid}\n{'='*70}")
            # Find lines related to Perch label CSV loading
            in_load_section = False
            for ln in src.split("\n"):
                # Skip very long bodies, focus on path/file logic
                if any(k in ln for k in ["perch-meta", "perch_meta", "labels.csv", "ebird", "load_label",
                                          "logit_map", "perch_to_bc", "perch_to_idx", "open(", "read_csv",
                                          "pd.read", "Path(", "exists()", "logit", "TAXO", "mapped"]):
                    print(f"  {ln.rstrip()[:160]}")
            break
