"""Patch exp050 to single GPU (no DataParallel) for safety fallback."""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp050\notebook\nb_train_convnext_r1.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

for c in nb["cells"]:
    if c.get("id") == "model_setup":
        src = "".join(c["source"])
        old = """if torch.cuda.device_count() > 1:  # DataParallel OK with NUM_WORKERS=0
    print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)"""
        new = """# DataParallel disabled (safety fallback - single GPU mode)
# Uncomment below to re-enable if diag NB shows DataParallel works
# if torch.cuda.device_count() > 1:
#     print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
#     model = nn.DataParallel(model)
print(f"Running on single GPU (DataParallel disabled for stability)")"""
        if old in src:
            src = src.replace(old, new)
            c["source"] = src.splitlines(keepends=True)
            print("[model_setup] DataParallel disabled (single GPU)")
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB}")
