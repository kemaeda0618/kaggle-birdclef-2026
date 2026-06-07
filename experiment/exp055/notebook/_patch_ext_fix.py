"""Fix the ext_name NameError in the already-patched cell."""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp055\notebook\nb_blend_a3.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

OLD = """        # A3 dedup: extend with pseudo author/label_str (each external entry unique pseudo-author)
        _ext_authors = np.array([f"ext_{ext_name}_{i}" for i in range(len(_ext_lbl))])
        _ext_label_str = np.array([f"ext_{i}" for i in range(len(_ext_lbl))])
        ta_pool_authors = np.concatenate([ta_pool_authors, _ext_authors])
        ta_pool_label_str = np.concatenate([ta_pool_label_str, _ext_label_str])"""

NEW = """        # A3 dedup: extend with unique pseudo author/label_str per external entry
        # (each ext entry unique → dedup never removes them)
        _n_ext = len(_ext_lbl)
        _ext_authors = np.array([f"ext_{i}" for i in range(_n_ext)])
        _ext_label_str = np.array([f"ext_lbl_{i}" for i in range(_n_ext)])
        ta_pool_authors = np.concatenate([ta_pool_authors, _ext_authors])
        ta_pool_label_str = np.concatenate([ta_pool_label_str, _ext_label_str])"""

n = 0
for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])
    if OLD in src:
        src = src.replace(OLD, NEW)
        c["source"] = src.splitlines(keepends=True)
        n += 1
        print("[ext fix] OK ext_name NameError fixed")
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Total changes: {n}")
