"""Fix submission block in exp060/exp062/exp063 blend cells.

Issue: blend cells use `row_ids` but the variable is `all_row_ids`.
Also missing the row_id alignment with sample_submission.csv.

Replace the submission-building block at end of blend cell with the original exp048 logic.
"""
import json
from pathlib import Path

CORRECT_BLOCK = """# === Build submission (matches exp048 logic with row_id alignment) ===
flat_pred = calibrated.reshape(-1, n_classes).astype(np.float32)

submission = pd.DataFrame(flat_pred, columns=PRIMARY_LABELS)
submission.insert(0, "row_id", all_row_ids)

sample_sub = pd.read_csv(BASE / "sample_submission.csv")
expected_ids = set(sample_sub["row_id"])
our_ids = set(submission["row_id"])
missing = expected_ids - our_ids
if missing:
    print(f"WARNING: {len(missing)} missing row_ids - filling zeros")
    missing_df = pd.DataFrame({"row_id": list(missing)})
    for sp in PRIMARY_LABELS:
        missing_df[sp] = 0.0
    submission = pd.concat([submission, missing_df], ignore_index=True)
extra = our_ids - expected_ids
if extra:
    submission = submission[submission["row_id"].isin(expected_ids)]
submission = submission.set_index("row_id").loc[sample_sub["row_id"]].reset_index()
submission.to_csv("submission.csv", index=False)

print(f"Submission: {submission.shape}")
print(f"Mean pred: {submission[PRIMARY_LABELS].values.mean():.6f}")
print(f"Max pred:  {submission[PRIMARY_LABELS].values.max():.6f}")
print(submission.head())
"""

NBS = [
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp060\notebook\nb_blend_inline.ipynb"),
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp062\notebook\nb_blend_aves_exp032.ipynb"),
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp063\notebook\nb_blend_aves_exp020.ipynb"),
]

for NB in NBS:
    print(f"\n=== {NB.name} ===")
    nb = json.load(open(NB, encoding="utf-8"))
    for c in nb["cells"]:
        if c.get("id") == "blend":
            src = "".join(c["source"])
            # Find marker - the start of our broken submission block
            # Look for "# === Build submission" or "flat_pred = calibrated.reshape"
            markers = [
                "# (5) Build submission",
                "# === Build submission ===",
                "flat_pred = calibrated.reshape(-1, n_classes)",
            ]
            cut_idx = -1
            for m in markers:
                idx = src.find(m)
                if idx >= 0:
                    cut_idx = idx
                    print(f"  found marker: {m[:50]}")
                    break
            if cut_idx >= 0:
                new_src = src[:cut_idx] + CORRECT_BLOCK
                c["source"] = new_src.splitlines(keepends=True)
                print(f"  patched, new length: {len(new_src.splitlines())} lines")
                break
            else:
                print(f"  marker not found, skipping")
    json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  Saved {NB.name}")
