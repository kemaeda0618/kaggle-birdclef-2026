"""Fix R2 NB pseudo_infer N_WINDOWS = 12 -> 3."""
import json
nb_path = r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp082/notebook/nb_train_r2.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if "N_WINDOWS    = 12" in src:
        src = src.replace("N_WINDOWS    = 12", "N_WINDOWS    = 3    # exp082: 60s / 20s = 3")
        c["source"] = src.splitlines(keepends=True)
        print(f"Patched cell id={c.get('id', '?')}")
        break
json.dump(nb, open(nb_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("Saved")
