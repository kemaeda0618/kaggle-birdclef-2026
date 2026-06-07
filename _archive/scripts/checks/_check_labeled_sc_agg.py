import json
from pathlib import Path

nb = json.loads(Path("experiment/exp082/notebook/nb_train_r2.ipynb").read_text(encoding="utf-8"))
for c in nb["cells"]:
    if c.get("id") == "load_data":
        src = "".join(c.get("source", []))
        lines = src.splitlines()
        # Find LabeledSC building section (Y_SC construction)
        for i, ln in enumerate(lines):
            if "Y_SC" in ln or "sc_meta" in ln or "aggregat" in ln.lower() or "20s" in ln or "4x" in ln.lower() or "4×" in ln:
                # Print this line + a few before/after
                lo = max(0, i - 2)
                hi = min(len(lines), i + 8)
                print(f"--- Block @ L{i} ---")
                for j in range(lo, hi):
                    print(f"L{j:3d}: {lines[j]}")
                print()
        break
