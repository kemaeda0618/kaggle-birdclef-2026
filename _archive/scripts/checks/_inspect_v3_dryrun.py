"""Find exact escape form of dry-run fallback in V3 NB JSON."""
import json
from pathlib import Path
nb = json.loads(Path("experiment/exp091_analysis/notebook/nb_exp090_oof_v3.ipynb").read_text(encoding="utf-8"))
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if "dry-run" in src and "test_paths" in src:
        cid = c.get("id", "?")
        print(f"Cell id={cid}")
        # find the dry-run block
        for i, ln in enumerate(src.split("\n")):
            if "dry-run" in ln or "train_soundscapes" in ln and "glob" in ln:
                # print context
                lines = src.split("\n")
                start = max(0, i - 2)
                end = min(len(lines), i + 4)
                for j in range(start, end):
                    print(f"  L{j}: {repr(lines[j])}")
                print()
        break
