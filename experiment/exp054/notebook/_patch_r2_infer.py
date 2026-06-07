"""Patch R2 inference NB: change ckpt path/slug from R1 to R2."""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp054\notebook\nb_r2_infer.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])
    # Change ckpt path
    src = src.replace(
        "/kaggle/input/birdclef2026-exp054-stage2-effv2s/stage2_v2_best.pth",
        "/kaggle/input/birdclef2026-exp054-stage2-r2-effv2s/stage2_v2_r2_best.pth"
    )
    src = src.replace(
        "/kaggle/input/datasets/maekeso/birdclef2026-exp054-stage2-effv2s/stage2_v2_best.pth",
        "/kaggle/input/datasets/maekeso/birdclef2026-exp054-stage2-r2-effv2s/stage2_v2_r2_best.pth"
    )
    src = src.replace("Stage 2 v2 ckpt loaded", "Stage 2 v2 R2 ckpt loaded")
    c["source"] = src.splitlines(keepends=True)

# Update hdr
for c in nb["cells"]:
    if c.get("id") == "hdr":
        src = "".join(c["source"])
        # Replace top-level
        new_src = src.replace(
            "exp054 Stage 2 v2 Inference",
            "exp054 R2 Inference"
        ).replace(
            "Stage 2 v2 ckpt",
            "R2 ckpt"
        ).replace(
            "maekeso/birdclef2026-exp054-stage2-effv2s",
            "maekeso/birdclef2026-exp054-stage2-r2-effv2s"
        )
        c["source"] = new_src.splitlines(keepends=True)
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB}")
