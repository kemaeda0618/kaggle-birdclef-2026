"""Fix LABEL_TO_CLASS undefined: inline-define before Aves mask in exp062/exp063."""
import json
from pathlib import Path

OLD = '''# Aves mask
AVES_MASK = np.array([LABEL_TO_CLASS.get(lbl) == "Aves" for lbl in PRIMARY_LABELS], dtype=bool)
print(f"Aves species: {AVES_MASK.sum()} (expected 162)")'''

NEW = '''# Aves mask (inline-define LABEL_TO_CLASS from taxonomy.csv)
_taxo = pd.read_csv(BASE / "taxonomy.csv")
_label_to_class = dict(zip(_taxo["primary_label"], _taxo["class_name"]))
AVES_MASK = np.array([_label_to_class.get(lbl) == "Aves" for lbl in PRIMARY_LABELS], dtype=bool)
print(f"Aves species: {AVES_MASK.sum()} (expected 162)")'''

# exp062 also has same issue with different word
OLD_062 = '''# Aves mask
AVES_MASK = np.array([LABEL_TO_CLASS.get(lbl) == "Aves" for lbl in PRIMARY_LABELS], dtype=bool)
print(f"Aves species count: {AVES_MASK.sum()} (expected 162)")'''

NEW_062 = '''# Aves mask (inline-define LABEL_TO_CLASS from taxonomy.csv)
_taxo = pd.read_csv(BASE / "taxonomy.csv")
_label_to_class = dict(zip(_taxo["primary_label"], _taxo["class_name"]))
AVES_MASK = np.array([_label_to_class.get(lbl) == "Aves" for lbl in PRIMARY_LABELS], dtype=bool)
print(f"Aves species count: {AVES_MASK.sum()} (expected 162)")'''

NBS = [
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp062\notebook\nb_blend_aves_exp032.ipynb"),
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp063\notebook\nb_blend_aves_exp020.ipynb"),
]

for NB in NBS:
    print(f"\n=== {NB.name} ===")
    nb = json.load(open(NB, encoding="utf-8"))
    patched = False
    for c in nb["cells"]:
        if c.get("id") == "blend":
            src = "".join(c["source"])
            replaced = False
            if OLD in src:
                src = src.replace(OLD, NEW); replaced = True
            elif OLD_062 in src:
                src = src.replace(OLD_062, NEW_062); replaced = True
            if replaced:
                c["source"] = src.splitlines(keepends=True)
                patched = True
                print("  blend cell patched")
            # Also replace LABEL_TO_CLASS anywhere else in cell (safety)
            if "LABEL_TO_CLASS" in "".join(c["source"]):
                src2 = "".join(c["source"]).replace("LABEL_TO_CLASS", "_label_to_class")
                c["source"] = src2.splitlines(keepends=True)
                print("  additional LABEL_TO_CLASS refs replaced")
            break
    if patched:
        json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  Saved")
    else:
        print(f"  No patch needed")
