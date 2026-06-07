"""Show CKPT_PRIORITY definition in nb_infer.ipynb."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_infer.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if "CKPT_PRIORITY" in src or "r2_ckpt" in src.lower():
        cid = c.get("id", "?")
        print(f"=== cell {cid} ===")
        # Print only the CKPT_PRIORITY block + surrounding load logic
        lines = src.split("\n")
        in_block = False
        for i, line in enumerate(lines):
            if "CKPT_PRIORITY" in line:
                in_block = True
            if in_block:
                print(f"  {line}")
                if line.strip().endswith("]") and "CKPT_PRIORITY" not in line:
                    # End of list, but continue a bit more to show load
                    for j in range(i+1, min(i+20, len(lines))):
                        print(f"  {lines[j]}")
                    in_block = False
                    break
