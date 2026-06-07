"""Fix exp058 infer NB to handle empty test_soundscapes in dev mode."""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp058\notebook\nb_infer.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

OLD = """probs_all = np.stack(probs_all).astype(np.float32)
print(f"\\ninference done: {probs_all.shape}, {(time.time()-t0)/60:.1f}min")"""

NEW = """if len(probs_all) > 0:
    probs_all = np.stack(probs_all).astype(np.float32)
    print(f"\\ninference done: {probs_all.shape}, {(time.time()-t0)/60:.1f}min")
else:
    # dev mode: test_soundscapes empty → create dummy with sample_sub row_ids
    print(f"\\nNo test files (dev mode). Submission will be filled with zeros aligned to sample_submission.")
    probs_all = np.zeros((0, N_WINDOWS, N_CLASSES), dtype=np.float32)
"""

OLD_SUB = """# Build submission
flat_pred = probs_all.reshape(-1, N_CLASSES).astype(np.float32)
submission = pd.DataFrame(flat_pred, columns=PRIMARY_LABELS)
submission.insert(0, "row_id", all_row_ids)"""

NEW_SUB = """# Build submission
if len(all_row_ids) > 0:
    flat_pred = probs_all.reshape(-1, N_CLASSES).astype(np.float32)
    submission = pd.DataFrame(flat_pred, columns=PRIMARY_LABELS)
    submission.insert(0, "row_id", all_row_ids)
else:
    # empty test → start with sample_sub structure (zeros)
    submission = sample_sub.copy()
    submission[PRIMARY_LABELS] = 0.0
    print("  empty test, filled zeros from sample_submission")"""

for c in nb["cells"]:
    if c.get("id") == "inference":
        src = "".join(c["source"])
        if OLD in src:
            src = src.replace(OLD, NEW)
            c["source"] = src.splitlines(keepends=True)
            print("  inference: empty handling added")
    if c.get("id") == "submission":
        src = "".join(c["source"])
        if OLD_SUB in src:
            src = src.replace(OLD_SUB, NEW_SUB)
            c["source"] = src.splitlines(keepends=True)
            print("  submission: empty handling added")

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB.name}")
