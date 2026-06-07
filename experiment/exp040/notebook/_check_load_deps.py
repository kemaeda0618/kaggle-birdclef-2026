"""Check whether load cell has the lab_* variables that exp040 train cell needs."""
import json
v11 = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\notebook\nb_blend_tucker_v11_g26.ipynb", encoding="utf-8"))
for c in v11["cells"]:
    if c.get("id") == "load":
        src = "".join(c["source"])
        print("=== v11 load cell ===")
        # Print key variable definitions
        for line in src.splitlines():
            ls = line.strip()
            if ("lab_emb_files" in ls or "lab_scores_files" in ls or "lab_site_ids" in ls or
                "lab_hours" in ls or "lab_labels_files" in ls or "lab_prior_files" in ls):
                print(f"  {line}")
        break
