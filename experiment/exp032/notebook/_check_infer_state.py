"""Check infer NB current state and Kaggle dataset status."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_infer.ipynb", encoding="utf-8"))
print(f"cells: {len(nb['cells'])}")
# Show hdr cell and ckpt-related cells
for c in nb["cells"]:
    cid = c.get("id", "?")
    src = "".join(c.get("source", []))
    if c.get("cell_type") == "markdown":
        print(f"\n=== md {cid} ===")
        print(src[:500])
    if "ckpt" in src.lower() or "weight" in src.lower() or "r1" in src.lower() or "r2" in src.lower():
        # Find the relevant line
        for line in src.split("\n"):
            if "ckpt" in line.lower() or "weights" in line.lower() or "/exp032/" in line.lower():
                print(f"  [{cid}] {line.strip()[:120]}")
                if line.strip().startswith("CKPT"):
                    break
