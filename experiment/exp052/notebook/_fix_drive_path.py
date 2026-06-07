"""Fix Drive paths in Stage 1 NB:
Wrong: /content/drive/MyDrive/birdclef2026/...
Right: /content/drive/MyDrive/kaggle/birdclef2026/...  (per user_colab_drive_folder.md)
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp052\notebook\nb_stage1_xc_pretrain.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

n_changes = 0
for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])
    new_src = src.replace(
        "/content/drive/MyDrive/birdclef2026/",
        "/content/drive/MyDrive/kaggle/birdclef2026/"
    )
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)
        diff = src.count("/content/drive/MyDrive/birdclef2026/")
        n_changes += diff
        print(f"  [{c.get('id')}] {diff} path fixes")

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nTotal path fixes: {n_changes}")

# Verify
print("\n=== Verify all Drive paths ===")
for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])
    for line in src.split("\n"):
        if "drive/MyDrive" in line:
            print(f"  [{c.get('id')}] {line.strip()[:150]}")
