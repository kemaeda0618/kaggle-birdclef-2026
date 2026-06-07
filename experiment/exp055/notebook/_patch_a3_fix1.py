"""Fix 1 retry: TA pool loading - preserve author + label_str.

Actual code has different structure than assumed. Use targeted line-level patch.
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp055\notebook\nb_blend_a3.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

# Insert A3 author preservation BEFORE the "del _ta_emb_raw, _ta_norm, _ta_meta" line
INSERT_BLOCK = """
    # A3: preserve author + label_str for retrieval dedup (BEFORE del _ta_meta)
    ta_pool_label_str = _ta_meta["primary_label"].astype(str).values
    if "author" in _ta_meta.columns:
        ta_pool_authors = _ta_meta["author"].astype(str).values
    else:
        try:
            _bc_train = pd.read_csv(BC2026_TRAIN_CSV)
            _fn_to_author = dict(zip(_bc_train["filename"], _bc_train["author"]))
            if "filename" in _ta_meta.columns:
                ta_pool_authors = np.array(
                    [_fn_to_author.get(fn, f"unk_{i}") for i, fn in enumerate(_ta_meta["filename"].values)]
                )
            else:
                ta_pool_authors = np.array([f"unk_{i}" for i in range(len(_ta_meta))])
        except Exception as _e:
            print(f"  A3 fallback: {_e}, using unk authors")
            ta_pool_authors = np.array([f"unk_{i}" for i in range(len(_ta_meta))])
    print(f"  A3 dedup: ta_pool_authors unique={len(set(ta_pool_authors))}, "
          f"ta_pool_label_str unique={len(set(ta_pool_label_str))}")
"""

n_changes = 0
for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])

    # Find target: "    del _ta_emb_raw, _ta_norm, _ta_meta"
    target = "    del _ta_emb_raw, _ta_norm, _ta_meta"
    has_def = "ta_pool_label_str = _ta_meta" in src  # actual DEFINITION
    if target in src and not has_def:
        # Insert before the del line
        src = src.replace(target, INSERT_BLOCK + target)
        c["source"] = src.splitlines(keepends=True)
        n_changes += 1
        print("[Fix 1 retry] OK author + label_str preservation inserted before del")
        break

# Also need to add BC2026_TRAIN_CSV constant
for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])
    if "BC2026_TRAIN_CSV" in src:
        # Already has it (from earlier patch)
        break
    if "EMB_DIR" in src and "SC_LABELS_CSV" in src:
        # Add near SC_LABELS_CSV
        src = src.replace(
            "SC_LABELS_CSV",
            "BC2026_TRAIN_CSV = '/kaggle/input/competitions/birdclef-2026/train.csv' if Path('/kaggle/input/competitions/birdclef-2026/train.csv').exists() else '/kaggle/input/birdclef-2026/train.csv'\nSC_LABELS_CSV",
            1
        )
        c["source"] = src.splitlines(keepends=True)
        n_changes += 1
        print("[BC2026_TRAIN_CSV] OK added")
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nTotal changes: {n_changes}")
print(f"Saved {NB}")

# Verify
print("\n=== Verify ===")
nb = json.load(open(NB, encoding="utf-8"))
for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])
    if "ta_pool_authors" in src and "del _ta" in src:
        # Check the order
        idx_def = src.index("ta_pool_authors = ")
        idx_del = src.index("del _ta_emb_raw")
        if idx_def < idx_del:
            print("  ✓ ta_pool_authors defined BEFORE del _ta_meta")
        else:
            print("  ✗ ORDER WRONG: ta_pool_authors AFTER del")
    if "BC2026_TRAIN_CSV" in src:
        print("  ✓ BC2026_TRAIN_CSV constant present")
