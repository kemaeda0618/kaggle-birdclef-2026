"""Find where v2 reference remains."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_xc_api_dl.ipynb", encoding="utf-8"))

for c in nb["cells"]:
    if c.get("id") == "query_xc":
        src = "".join(c["source"])
        for i, line in enumerate(src.split("\n"), 1):
            if "api/2" in line or "api/3" in line:
                print(f"  L{i}: {line.strip()[:200]}")
