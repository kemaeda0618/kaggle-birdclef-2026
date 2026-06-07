"""Fix framewise.max(dim=-1) → max(dim=1) in all inference NBs.

BirdSEDModel returns framewise as (B, time, num_class).
For frame_max (max over time), use dim=1, not dim=-1.
"""
import json
from pathlib import Path

NB_PATHS = [
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp081/notebook/nb_infer.ipynb",
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp081/notebook/nb_infer_r1.ipynb",
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp082/notebook/nb_infer.ipynb",
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/ensemble_e14_17/notebook/nb_infer_ensemble.ipynb",
]

for p in NB_PATHS:
    p = Path(p)
    if not p.exists():
        continue
    print(f"\n=== {p.parent.name}/{p.name} ===")
    nb = json.loads(p.read_text(encoding="utf-8"))
    changed = False
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        new_src = src.replace(
            "framewise.max(dim=-1).values",
            "framewise.max(dim=1).values"
        )
        if new_src != src:
            c["source"] = new_src.splitlines(keepends=True)
            print(f"  Fixed framewise.max in cell {c.get('id', '?')}")
            changed = True
    if changed:
        p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  Saved")
    else:
        print(f"  No changes")
print("Done")
