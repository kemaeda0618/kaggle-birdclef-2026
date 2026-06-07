import json, sys
nb = json.loads(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\my_nb4_latest\birdclef2026-exp010-nb4-blend-protossm-mlp.ipynb", encoding="utf-8").read())
ids = [int(x) for x in sys.argv[1:]]
for i in ids:
    src = "".join(nb["cells"][i].get("source", []))
    print(f"\n=== CELL {i} ({len(src)}c) ===\n{src}\n")
