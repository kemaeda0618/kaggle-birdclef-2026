"""Check exp063 blend cell for _label_to_class definition."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp063\notebook\nb_blend_aves_exp020.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    if c.get("id") == "blend":
        src = "".join(c["source"])
        # Check if _label_to_class is referenced but not defined
        has_def = "_label_to_class = dict" in src
        has_ref = "_label_to_class.get" in src or "_label_to_class[" in src
        print(f"  has definition: {has_def}")
        print(f"  has reference:  {has_ref}")
        if has_ref and not has_def:
            print("  ★ FIX NEEDED ★")
        # Show context around _label_to_class
        for i, line in enumerate(src.split("\n")):
            if "_label_to_class" in line or "LABEL_TO_CLASS" in line or "AVES_MASK" in line:
                print(f"  L{i}: {line}")
