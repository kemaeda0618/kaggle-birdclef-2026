"""Fix exp056: LABEL_TO_CLASS not defined. Load taxonomy inline."""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp056\notebook\nb_blend_aves_stage2a.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

OLD = '# === Aves-only Stage 2A blend (exp056) ===\nAVES_MASK = np.array([LABEL_TO_CLASS.get(lbl) == "Aves" for lbl in PRIMARY_LABELS], dtype=bool)'

NEW = """# === Aves-only Stage 2A blend (exp056) ===
# Load taxonomy inline to build LABEL_TO_CLASS map (avoid NameError)
_TAX_PATHS = [
    Path("/kaggle/input/competitions/birdclef-2026/taxonomy.csv"),
    Path("/kaggle/input/birdclef-2026/taxonomy.csv"),
]
_tax_path = next((p for p in _TAX_PATHS if p.exists()), None)
assert _tax_path is not None, f"taxonomy.csv not found in {_TAX_PATHS}"
_tax = pd.read_csv(_tax_path)
_LABEL_TO_CLASS = dict(zip(_tax["primary_label"], _tax["class_name"]))
AVES_MASK = np.array([_LABEL_TO_CLASS.get(lbl) == "Aves" for lbl in PRIMARY_LABELS], dtype=bool)"""

for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])
    if OLD in src:
        src = src.replace(OLD, NEW)
        c["source"] = src.splitlines(keepends=True)
        print("[exp056] OK LABEL_TO_CLASS inline load added")
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB}")
