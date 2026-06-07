"""Build exp096 NB by adding temporal block completion PP to exp090 blend cell.

Source: hengck23 trick1 (BC2026 discussion #684148)
  "labels come in longer continuous blocks (10-40s). 3+ consecutive 5s segments
   with same label = block."

Implementation:
  - Per soundscape file (parsed from row_id "<filename>_<end_time>"), sort by end_time.
  - For each species column, detect runs where p > THRESH.
  - If run length >= MIN_BLOCK_LEN: apply block_max with mix α=BLOCK_ALPHA (dip-fill).

Defensive: skip on row_id parse failure, missing files, etc.

Inserts new PP cell AFTER tax_smoothing application, BEFORE submission.to_csv save.
Header updated to reflect new axis.
"""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_exp096_block_pp.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# === Find blend cell ===
blend_cell = None
for c in nb["cells"]:
    if c.get("id") == "blend":
        blend_cell = c
        break
assert blend_cell is not None, "blend cell not found"

blend_src = "".join(blend_cell["source"])
print(f"blend cell original length: {len(blend_src)}")

# === New PP function + application ===
# Insert AFTER tax_smoothing application, BEFORE submission.to_csv
ANCHOR_OLD = """# Apply tax smoothing
submission = f_tax_smoothing(submission, genus_alpha=0.15, class_alpha=0.05)

submission.to_csv("submission.csv", index=False)"""

NEW_BLOCK = '''# Apply tax smoothing
submission = f_tax_smoothing(submission, genus_alpha=0.15, class_alpha=0.05)

# === EXP096: Temporal block completion (hengck23 trick1, #684148) ===
# Per-species: detect runs of MIN_BLOCK_LEN+ consecutive 5s segments with p > THRESH,
# then dip-fill within block via block_max mixed at BLOCK_ALPHA.
# Defensive: skip on row_id parse failure.
BLOCK_THRESH    = 0.30   # min p to count as "active" segment
MIN_BLOCK_LEN   = 3      # min consecutive segments to form a block (= 15s window)
BLOCK_ALPHA     = 0.50   # 0.0=identity, 1.0=full block_max replacement

def f_temporal_block_completion(submission, label_cols,
                                 thresh=BLOCK_THRESH,
                                 min_block_len=MIN_BLOCK_LEN,
                                 alpha=BLOCK_ALPHA):
    """Per-file, per-species block detection + dip-fill.

    Mechanism: within a detected block, replace each p with
        (1 - alpha) * p + alpha * max(block_p)
    This lifts mid-block dips toward the block peak while preserving the peak.
    AUC-relevant because it changes the relative ordering of within-file segments
    against negative segments from OTHER files.

    Defensive: any parse/loop failure returns submission unchanged.
    """
    try:
        from collections import defaultdict
        row_ids = submission["row_id"].values
        # Parse row_id = "<filename>_<end_time_seconds>"
        meta = []
        for rid in row_ids:
            parts = str(rid).rsplit("_", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                print(f"[block_pp] row_id parse fail: {rid!r} - skip whole PP")
                return submission
            meta.append((parts[0], int(parts[1])))

        file_to_idx = defaultdict(list)
        for i, (fn, t) in enumerate(meta):
            file_to_idx[fn].append((t, i))

        probs = submission[label_cols].values.astype(np.float32).copy()
        modified_blocks = 0
        n_lifted_points = 0
        for fn, lst in file_to_idx.items():
            lst.sort()
            row_idx = [i for _, i in lst]
            n_seg = len(row_idx)
            if n_seg < min_block_len:
                continue
            seg_probs = probs[row_idx, :]  # (n_seg, n_classes)

            for c in range(seg_probs.shape[1]):
                col = seg_probs[:, c]
                active = col > thresh
                k = 0
                while k < n_seg:
                    if active[k]:
                        run_end = k
                        while run_end < n_seg and active[run_end]:
                            run_end += 1
                        run_len = run_end - k
                        if run_len >= min_block_len:
                            bmax = col[k:run_end].max()
                            new_vals = (1.0 - alpha) * col[k:run_end] + alpha * bmax
                            lifted = (new_vals > col[k:run_end]).sum()
                            n_lifted_points += int(lifted)
                            seg_probs[k:run_end, c] = new_vals.astype(np.float32)
                            modified_blocks += 1
                        k = run_end
                    else:
                        k += 1

            probs[row_idx, :] = seg_probs

        submission[label_cols] = probs
        print(f"[block_pp] thresh={thresh}, min_len={min_block_len}, alpha={alpha}")
        print(f"[block_pp] blocks_lifted={modified_blocks}, points_lifted={n_lifted_points}")
        print(f"[block_pp] mean post: {submission[label_cols].values.mean():.6f}")
        return submission
    except Exception as e:
        print(f"[block_pp] error: {e} - skip")
        return submission

submission = f_temporal_block_completion(submission, label_cols=PRIMARY_LABELS)
# === / EXP096 ===

submission.to_csv("submission.csv", index=False)'''

if ANCHOR_OLD not in blend_src:
    raise RuntimeError("Anchor not found in blend cell")

new_blend_src = blend_src.replace(ANCHOR_OLD, NEW_BLOCK)
blend_cell["source"] = new_blend_src.splitlines(keepends=True)
print(f"blend cell new length: {len(new_blend_src)}")
print(f"added: +{len(new_blend_src) - len(blend_src)} chars")

# === Update header cell ===
hdr_cell = None
for c in nb["cells"]:
    if c.get("id") == "hdr":
        hdr_cell = c
        break
assert hdr_cell is not None

new_hdr = '''# exp096 — exp090 + Temporal block completion PP (hengck23 trick1)

## 変更点 (vs exp090)
- exp090 stack (NB4 v7 0.35 / Tucker 0.40 / exp029 R3 0.25 + lambda_prior 0.65 + rank_aware 0.6 + TAX_SMOOTHING) を維持
- **追加: Temporal block completion PP** (Tax smoothing 直後、to_csv 直前)
  - per-file, per-species で `p > 0.30` の 3+ consecutive 5s 検出 → block 内 dip を `0.5 * p + 0.5 * block_max` で持ち上げ
  - 出典: hengck23 BC2026 discussion #684148 trick1 (chiranjithdharma 0.910 公開 NB が exploit)

## Mechanism
- Per-file の within-file rank が変わる (multi-row change)、AUC は per-class column の global rank で評価されるので AUC に影響あり (NOT monotonic transform)
- 仮定: 連続 active segment = 同種が continuously calling、間の dip は momentary quiet → block_max で持ち上げる

## Expected
- LB **0.951-0.954** (exp090 の 0.951 から +0.000-0.003)
- 改善確率: **45%**
- drag 確率: **20%** (block 内に真の negative 段間がある場合)
- defensive: row_id parse fail なら identity 返す

## Hyperparameters
- BLOCK_THRESH    = 0.30
- MIN_BLOCK_LEN   = 3 (= 15s window)
- BLOCK_ALPHA     = 0.50 (中庸; 1.0=full block_max、0.0=identity)

## Inputs (exp090 と完全同一)
- `birdclef-2026` (competition)
- `jaejohn/perch-meta`
- `rishikeshjani/perch-onnx-for-birdclef-2026`
- `tuckerarrants/bc2026-distilled-sed-public`
- `maekeso/birdclef2026-exp029-l1-single`
- kernel_sources: `ashok205/tf-wheels`
- model_sources: `google/bird-vocalization-classifier/.../perch_v2_cpu/1`
'''
hdr_cell["source"] = new_hdr.splitlines(keepends=True)

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n[OK] Saved: {NB_PATH}")
print(f"Cell count: {len(nb['cells'])}")
