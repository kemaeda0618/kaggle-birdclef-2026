import json
from pathlib import Path

OLD = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\mtoshidesu_v313904345\0-932-v313904345.ipynb")
LATEST = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\mtoshidesu_0-932-test-mod-version-4\0-932-test-mod-version-4.ipynb")

nb_old = json.loads(OLD.read_text(encoding="utf-8"))
nb_latest = json.loads(LATEST.read_text(encoding="utf-8"))

print(f"v313904345 (0.932 LB candidate): {len(nb_old['cells'])} cells")
print(f"latest:                         {len(nb_latest['cells'])} cells")

print(f"\n=== v313904345 cells ===")
for i, c in enumerate(nb_old["cells"]):
    src = "".join(c.get("source", []))
    first = src.split("\n")[0][:120] if src else "(empty)"
    print(f"  [{i:02d}] {c['cell_type']:8s} {len(src):5d}c | {first}")

print(f"\n=== latest cells ===")
for i, c in enumerate(nb_latest["cells"]):
    src = "".join(c.get("source", []))
    first = src.split("\n")[0][:120] if src else "(empty)"
    print(f"  [{i:02d}] {c['cell_type']:8s} {len(src):5d}c | {first}")
