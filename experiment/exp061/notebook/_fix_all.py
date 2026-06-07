"""Fix exp061: EMB_DIR private path + row_ids → all_row_ids."""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp061\notebook\nb_blend_inline.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

# Fix 1: EMB_DIR path (private kernel_source fallback)
OLD_EMB = 'EMB_DIR = Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp010-nb1-embedding")'
NEW_EMB = '''_emb_candidates = [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp010-nb1-embedding"),
    Path("/kaggle/input/maekeso/birdclef2026-exp010-nb1-embedding"),  # private kernel_source
    Path("/kaggle/input/birdclef2026-exp010-nb1-embedding"),
]
EMB_DIR = next((p for p in _emb_candidates if p.exists()), _emb_candidates[0])
if not EMB_DIR.exists():
    for _hit in Path("/kaggle/input").rglob("soundscape_embeddings.npz"):
        EMB_DIR = _hit.parent; break'''

# Fix 2: blend submission block (row_ids → all_row_ids + alignment)
CORRECT_BLEND_END = """# === Build submission (matches exp048 logic with row_id alignment) ===
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

# Apply fixes
for c in nb["cells"]:
    if c.get("id") == "config":
        src = "".join(c["source"])
        if OLD_EMB in src:
            src = src.replace(OLD_EMB, NEW_EMB)
            c["source"] = src.splitlines(keepends=True)
            print("  config: EMB_DIR fallback added")
    if c.get("id") == "blend":
        src = "".join(c["source"])
        # Find marker - same logic as before
        markers = [
            "# === Build submission ===",
            "# (5) Build submission",
            "flat_pred = calibrated.reshape(-1, n_classes)",
        ]
        cut_idx = -1
        for m in markers:
            idx = src.find(m)
            if idx >= 0:
                cut_idx = idx
                print(f"  blend: marker found '{m[:40]}'")
                break
        if cut_idx >= 0:
            new_src = src[:cut_idx] + CORRECT_BLEND_END
            c["source"] = new_src.splitlines(keepends=True)
            print(f"  blend: submission block fixed")

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB.name}")
