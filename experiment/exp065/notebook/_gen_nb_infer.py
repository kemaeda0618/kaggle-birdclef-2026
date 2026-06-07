"""Generate exp065 standalone inference NB (Kaggle CPU).
Same architecture as exp058 infer, different ckpt slug.
"""
import json, shutil
from pathlib import Path

SRC = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp058\notebook\nb_infer.ipynb")
OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp065\notebook\nb_infer.ipynb")

nb = json.load(open(SRC, encoding="utf-8"))

# Patch: change ckpt path/slug + filename
for c in nb["cells"]:
    src = "".join(c["source"])
    new_src = src
    # ckpt slug
    new_src = new_src.replace("birdclef2026-exp058-effv2b0-combined", "birdclef2026-exp065-effv2b0-r2-sc")
    # ckpt filename
    new_src = new_src.replace("effv2b0_combined_best.pth", "effv2b0_r2_sc_best.pth")
    # hdr text
    new_src = new_src.replace("exp058 Standalone Inference", "exp065 (R2 SC) Standalone Inference")
    new_src = new_src.replace("Training: BC2026 train_audio (hard) + XC pseudo",
                              "Training: BC2026 (hard) + XC pseudo + SC pseudo (NEW), R1 warm-start")
    new_src = new_src.replace("Best ckpt: ep 11, val_macro 0.9678, val_ns22 0.8341",
                              "Best ckpt from R2 SC training (see exp065 log)")
    new_src = new_src.replace("Expected LB**: 0.91-0.94", "Expected LB**: 0.85-0.92")
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)

json.dump(nb, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Wrote {OUT}")
