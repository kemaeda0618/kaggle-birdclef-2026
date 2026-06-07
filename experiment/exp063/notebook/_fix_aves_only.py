"""Fix LABEL_TO_CLASS undefined in exp063 only (don't push, just save)."""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp063\notebook\nb_blend_aves_exp020.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

OLD = '''AVES_MASK = np.array([LABEL_TO_CLASS.get(lbl) == "Aves" for lbl in PRIMARY_LABELS], dtype=bool)
print(f"Aves species: {AVES_MASK.sum()} (expected 162)")'''

NEW = '''_taxo_e63 = pd.read_csv(BASE / "taxonomy.csv")
_label_to_class = dict(zip(_taxo_e63["primary_label"], _taxo_e63["class_name"]))
AVES_MASK = np.array([_label_to_class.get(lbl) == "Aves" for lbl in PRIMARY_LABELS], dtype=bool)
print(f"Aves species: {AVES_MASK.sum()} (expected 162)")'''

for c in nb["cells"]:
    if c.get("id") == "blend":
        src = "".join(c["source"])
        if OLD in src:
            src = src.replace(OLD, NEW)
            c["source"] = src.splitlines(keepends=True)
            print("  patched")
        else:
            print("  OLD not found, listing LABEL_TO_CLASS lines:")
            for line in src.split("\n"):
                if "LABEL_TO_CLASS" in line: print(f"    {line}")
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB.name}")
