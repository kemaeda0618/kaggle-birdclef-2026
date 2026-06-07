"""Fix broken multi-line print v2.

Current broken state:
    print(f"Soundscape labels: ... positives, "    <- line 1 (open)
    [BLANK]
    # === aggregation === ...
    print(f"  [exp08X] After ...")
          f"{int(...)} species")                    <- ORPHAN: line 2 of original print

Goal: reconstruct as
    print(f"Soundscape labels: ... positives, "
          f"{int(...)} species")                    <- closed properly
    [BLANK]
    # === aggregation ===
    ...
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
        return False

    src = "".join(load_data["source"])

    # Check if broken pattern exists: print line 1 without closing,
    # followed by blank line, then aggregation, then orphan line 2
    # Detection: "positives, \"\n\n    # ====" pattern
    if 'positives, "\n\n    # =====' not in src:
        print("  no broken pattern found (already fixed?)")
        return False

    # Reconstruct:
    # The structure is:
    #   print(f"... positives, "\n   <- need to close this with the species line
    #   \n                            <- blank line
    #   [aggregation block, which contains its own print]
    #   \n                            <- blank line
    #   ORPHAN: '          f"{int((Y_SC.sum(axis=0) > 0).sum())} species")\n'
    #   \n                            <- blank line
    #   '    sc_files = sc_meta...

    # The orphan line is exactly: '          f"{int((Y_SC.sum(axis=0) > 0).sum())} species")\n'
    orphan_line = '          f"{int((Y_SC.sum(axis=0) > 0).sum())} species")\n'

    # Find the FIRST occurrence (this should be inside my inserted aggregation print, valid)
    # Find the SECOND occurrence (this is the orphan)
    occurrences = []
    pos = 0
    while True:
        idx = src.find(orphan_line, pos)
        if idx < 0:
            break
        occurrences.append(idx)
        pos = idx + len(orphan_line)
    print(f"  orphan line occurrences: {len(occurrences)}")

    if len(occurrences) < 2:
        print("  WARN: < 2 occurrences, can't identify orphan reliably")
        return False

    # Strategy:
    # 1. Add the species line right after first print line (close original print)
    # 2. Remove the LAST orphan occurrence
    line1_end_marker = 'positives, "\n\n    # ====='
    # Insert the species line after positives, "\n (closing original print)
    new_src = src.replace(
        'positives, "\n\n    # =====',
        'positives, "\n          f"{int((Y_SC.sum(axis=0) > 0).sum())} species")\n\n    # =====',
        1,  # only first occurrence
    )

    # Now we have 3 occurrences total (added one).
    # Remove the LAST occurrence (the original orphan).
    # Find last occurrence in new_src
    last_idx = new_src.rfind(orphan_line)
    if last_idx < 0:
        print("  WARN: orphan line not found in new_src")
        return False
    new_src = new_src[:last_idx] + new_src[last_idx + len(orphan_line):]

    load_data["source"] = new_src.splitlines(keepends=True)
    Path(nb_path).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  Fixed: print closed properly, orphan removed")
    return True


for p in NB_PATHS:
    if Path(p).exists():
        fix(p)
    else:
        print(f"\nMissing: {p}")

print("\nDone.")
