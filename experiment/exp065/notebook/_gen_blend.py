"""Generate exp065 blend NB: exp048 + exp065 R2 inline 4-way."""
import json
from pathlib import Path

SRC = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp061\notebook\nb_blend_inline.ipynb")
OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp065\notebook\nb_blend_inline.ipynb")

nb = json.load(open(SRC, encoding="utf-8"))

# Replace exp058 references with exp065
for c in nb["cells"]:
    src = "".join(c["source"])
    new_src = src
    new_src = new_src.replace("birdclef2026-exp058-effv2b0-combined", "birdclef2026-exp065-effv2b0-r2-sc")
    new_src = new_src.replace("effv2b0_combined_best.pth", "effv2b0_r2_sc_best.pth")
    new_src = new_src.replace("exp058 (effv2_b0 + XC pseudo combined)", "exp065 (effv2_b0 + BC + XC + SC pseudo)")
    new_src = new_src.replace("exp058 effv2_b0 + XC pseudo trained", "exp065 R2 SC")
    new_src = new_src.replace("E58", "E65")
    new_src = new_src.replace("exp058 effv2_b0 + XC pseudo", "exp065 R2 SC")
    new_src = new_src.replace("e58", "e65")
    new_src = new_src.replace("exp061 — Inline 4-way Blend (exp048 base + exp058 effv2_b0)",
                              "exp065-blend — Inline 4-way (exp048 + exp065 R2 effv2_b0)")
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)

json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Wrote {OUT}")
