"""Fix Kaggle competition / dataset paths in extract NB (try both /competitions/<slug> and /<slug>)."""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_extract_pretrain.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

# Patch extract_bc2021 cell
for c in nb["cells"]:
    if c.get("id") == "extract_bc2021":
        src = "".join(c["source"])
        # Replace single path with multi-candidate
        old = 'BC2021_ROOT = Path("/kaggle/input/birdclef-2021")'
        new = '''BC2021_ROOT = next(
    (p for p in [
        Path("/kaggle/input/competitions/birdclef-2021"),
        Path("/kaggle/input/birdclef-2021"),
    ] if p.exists()),
    Path("/kaggle/input/competitions/birdclef-2021"),  # placeholder if missing
)'''
        if old in src:
            src = src.replace(old, new)
            c["source"] = src.splitlines(keepends=True)
            print("[extract_bc2021] OK path fix")

# Patch extract_bc2022_2025 cell
for c in nb["cells"]:
    if c.get("id") == "extract_bc2022_2025":
        src = "".join(c["source"])
        old = '    root = Path(f"/kaggle/input/birdclef-{year}")'
        new = '''    root_candidates = [
        Path(f"/kaggle/input/competitions/birdclef-{year}"),
        Path(f"/kaggle/input/birdclef-{year}"),
    ]
    root = next((p for p in root_candidates if p.exists()), None)
    if root is None:
        root = root_candidates[0]'''
        if old in src:
            src = src.replace(old, new)
            c["source"] = src.splitlines(keepends=True)
            print("[extract_bc2022_2025] OK path fix")

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nSaved {NB}")

# Verify
print("\n=== Verify ===")
for c in nb["cells"]:
    if c.get("id") in ("extract_bc2021", "extract_bc2022_2025"):
        src = "".join(c["source"])
        for line in src.split("\n"):
            if "competitions/birdclef" in line or "input/birdclef" in line:
                print(f"  [{c.get('id')}] {line.strip()[:120]}")
