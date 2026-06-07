"""Fix broken multi-line print statement in load_data cell.

The aggregation code was inserted BETWEEN the two lines of:
    print(f"Soundscape labels: ..."
          f"... species")

This breaks Python syntax. Move the aggregation AFTER the entire print.
"""
import json, re
from pathlib import Path

NB_PATHS = [
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp081/notebook/nb_train_r1.ipynb",
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp081/notebook/nb_train_r2.ipynb",
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp082/notebook/nb_train_r1.ipynb",
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp082/notebook/nb_train_r2.ipynb",
]


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
        print("  no load_data cell")
        return

    src = "".join(load_data["source"])

    # Step 1: Find broken pattern
    # The inserted aggregation block starts with "    # ============================================================\n    # ★ exp081/exp082: Aggregate labeled SS"
    # Use regex to find it.
    pat = re.compile(
        r'(positives, "\n)'              # End of first print line
        r'(\n    # ============================================================\n    # ★ exp08[12]: Aggregate labeled SS.*?(?=\n    sc_files = sc_meta))',
        re.DOTALL,
    )

    m = pat.search(src)
    if not m:
        print("  no broken pattern found, skipping (may already be fixed)")
        return

    # Step 2: Extract aggregation code
    print(f"  Found broken pattern at position {m.start()}")
    aggregation_block = m.group(2)

    # Step 3: Remove from current position (between print lines)
    src_clean = src[:m.start(2)] + src[m.end(2):]

    # Now src_clean has the print restored but no aggregation. Need to insert aggregation after the full print.

    # Step 4: Find the END of the print statement
    # Looking for the closing pattern: `species")`
    # After: `print(f"Soundscape labels: ... positives, "\n          f"...species")\n`
    end_pat = re.compile(
        r'(print\(f"Soundscape labels:.*?species"\)\n)',
        re.DOTALL,
    )
    m2 = end_pat.search(src_clean)
    if not m2:
        print("  WARN: complete print pattern not found in cleaned source")
        return

    insert_pos = m2.end()
    new_src = src_clean[:insert_pos] + aggregation_block + "\n" + src_clean[insert_pos:]

    load_data["source"] = new_src.splitlines(keepends=True)
    Path(nb_path).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  Fixed: aggregation block moved after complete print")


for p in NB_PATHS:
    if Path(p).exists():
        fix(p)
    else:
        print(f"\nMissing: {p}")

print("\nDone.")
