"""Find all print statements with potential issues (hardcoded exp numbers, leftover refs)."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp046\notebook\nb_train_r3_l1_gamma12.ipynb", encoding="utf-8"))

ISSUES = [
    "exp029",  # leftover (only allowed in baseline references)
    "exp045",  # leftover (chunk filter)
    "filter",  # may indicate old chunk filter references
    "CHUNK_FILTER",
    "primary_label_min",
]

for c in nb["cells"]:
    cid = c.get("id", "?")
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])
    for i, line in enumerate(src.split("\n"), 1):
        if "print" not in line:
            continue
        ls = line.strip()
        for issue in ISSUES:
            if issue in ls:
                print(f"  [{cid}:L{i}] {ls[:150]}")
                break

# Also show summary cell content
print("\n=== summary cell ===")
for c in nb["cells"]:
    if c.get("id") == "summary":
        print("".join(c["source"]))
        break
