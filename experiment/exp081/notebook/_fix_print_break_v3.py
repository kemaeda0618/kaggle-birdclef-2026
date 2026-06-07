"""Fix v3: close first print + remove orphan line."""
import json
from pathlib import Path

NB_PATHS = [
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp081/notebook/nb_train_r1.ipynb",
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp081/notebook/nb_train_r2.ipynb",
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp082/notebook/nb_train_r1.ipynb",
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp082/notebook/nb_train_r2.ipynb",
]

orphan_line = '          f"{int((Y_SC.sum(axis=0) > 0).sum())} species")\n'


def find_cell(cells, cid):
    for c in cells:
        if c.get("id") == cid:
            return c
    return None


def fix(nb_path):
    print(f"\n=== Fixing {Path(nb_path).name} ===")
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    load_data = find_cell(nb["cells"], "load_data")
    if load_data is None:
        return False

    src = "".join(load_data["source"])

    # Check: broken pattern is `positives, "\n\n    # =====`
    broken_marker = 'positives, "\n\n    # ====='
    if broken_marker not in src:
        print("  not broken (already fixed?)")
        return False

    # Step 1: Close the first print by inserting species line after `positives, "\n`
    # Pattern: 'positives, "\n\n    # ====='  →  'positives, "\n          f"...species")\n\n    # ====='
    fixed = src.replace(
        broken_marker,
        f'positives, "\n{orphan_line}\n    # =====',
        1,
    )

    # Step 2: Remove the orphan line (the LAST occurrence; first one is the one we just added)
    last_idx = fixed.rfind(orphan_line)
    if last_idx < 0:
        print("  WARN: orphan not found")
        return False
    fixed = fixed[:last_idx] + fixed[last_idx + len(orphan_line):]

    load_data["source"] = fixed.splitlines(keepends=True)
    Path(nb_path).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  Fixed.")
    return True


for p in NB_PATHS:
    if Path(p).exists():
        fix(p)
    else:
        print(f"\nMissing: {p}")

print("\nDone.")
