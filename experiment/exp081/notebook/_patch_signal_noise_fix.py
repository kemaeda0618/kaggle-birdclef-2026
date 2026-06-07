"""Patch exp081 R1+R2 NB to fix signal noise from 5s → 10s window:

Fix 1: LabeledSCDS aggregation
  - sc_meta 5s start_secs → 10s windows (pair adjacent 5s chunks)
  - Y_SC aggregated via max() of 2 consecutive 5s labels
  - Affects both training and val eval (val_sc_df derived from sc_meta)

Fix 2: Val eval consistency
  - Auto-fixed via Fix 1 (val uses same sc_meta)

Fix 3: Perch distill alignment
  - For 10s wav (320000 samples): compute 2× Perch embeddings (first/second 5s) → average
  - Model's distill_emb aligned with averaged Perch embedding
  - Cost: 2x Perch inference (~+10-20% training time)
"""
import json
from pathlib import Path

NB_PATHS = [
    Path(__file__).with_name("nb_train_r1.ipynb"),
    Path(__file__).with_name("nb_train_r2.ipynb"),
]


def find_cell(cells, cid):
    for c in cells:
        if c.get("id") == cid:
            return c
    return None


def patch_load_data_labeled_sc(load_data_cell):
    """Add Y_SC + sc_meta aggregation 5s → 10s in load_data cell."""
    src = "".join(load_data_cell["source"])
    if "★ exp081: Aggregate labeled SS 5s" in src:
        print("  [load_data] labeled SS aggregation already added, skipping")
        return False

    # Insert aggregation AFTER Y_SC is built. Find the print line.
    # Look for `print(f"Soundscape labels: {len(sc_meta)} windows, ...")`
    ANCHOR = 'print(f"Soundscape labels: {len(sc_meta)} windows'
    if ANCHOR not in src:
        print("  [load_data] WARN: anchor not found, skipping labeled SS fix")
        return False

    # Find the end of the labeled SS section (before non_s22_mask logic)
    INSERT_AFTER_LINE = ANCHOR
    aggregation_code = '''
    # ============================================================
    # ★ exp081: Aggregate labeled SS 5s -> 10s windows (max() merge)
    # ============================================================
    # Pair adjacent 5s chunks (start_sec, start_sec+5) -> 10s window at start_sec
    # Result: 6 non-overlapping 10s windows per file (vs 12 5s windows)
    print("[exp081] Aggregating labeled SS 5s -> 10s windows...")
    sc_meta_orig = sc_meta.copy()
    sc_meta_orig["_orig_idx"] = np.arange(len(sc_meta_orig))
    Y_SC_orig = Y_SC.copy()
    new_sc_rows = []
    new_Y_SC_list = []
    for fname, group in sc_meta_orig.groupby("filename"):
        group_sorted = group.sort_values("start_sec").reset_index(drop=True)
        # Non-overlapping 10s windows: pairs (0,1), (2,3), ...
        for k in range(0, len(group_sorted) - 1, 2):
            idx_a = int(group_sorted.iloc[k]["_orig_idx"])
            idx_b = int(group_sorted.iloc[k+1]["_orig_idx"])
            start_sec_10s = float(group_sorted.iloc[k]["start_sec"])
            y_agg = np.maximum(Y_SC_orig[idx_a], Y_SC_orig[idx_b])
            site_v = group_sorted.iloc[k].get("site", "UNK")
            new_sc_rows.append({"filename": fname, "start_sec": start_sec_10s, "site": site_v})
            new_Y_SC_list.append(y_agg)
    sc_meta = pd.DataFrame(new_sc_rows).reset_index(drop=True)
    Y_SC = np.array(new_Y_SC_list, dtype=np.float32)
    print(f"  [exp081] After labeled SS 10s aggregation: {len(sc_meta)} windows, "
          f"{int(Y_SC.sum())} positives, {int((Y_SC.sum(axis=0) > 0).sum())} species")
'''
    # Insert RIGHT AFTER the anchor line (which ends with )
    idx = src.find(ANCHOR)
    # find end of that line
    end_idx = src.find("\n", idx)
    if end_idx == -1:
        print("  [load_data] WARN: anchor line end not found")
        return False
    # Insert aggregation after the anchor line
    new_src = src[:end_idx+1] + aggregation_code + src[end_idx+1:]
    load_data_cell["source"] = new_src.splitlines(keepends=True)
    print("  [load_data] Added labeled SS 5s -> 10s aggregation")
    return True


def patch_train_loop_perch_distill(train_cell):
    """Replace center-5s Perch with 2x 5s averaged for 10s alignment."""
    src = "".join(train_cell["source"])
    if "★ exp081 Perch distill: 2x 5s averaged" in src:
        print("  [train_loop] Perch distill already patched, skipping")
        return False

    OLD_PERCH = '''            if USE_PERCH_DISTILL and perch_teacher is not None:
                with torch.no_grad():
                    wav_5s = wav.squeeze(1)
                    N = wav_5s.shape[1]
                    if N > 160000:
                        start_off = (N - 160000) // 2
                        wav_5s = wav_5s[:, start_off:start_off + 160000]
                    elif N < 160000:
                        wav_5s = F.pad(wav_5s, (0, 160000 - N))
                    perch_emb = perch_teacher.embed(wav_5s).to(device)
                distill_loss = F.mse_loss(distill_emb, perch_emb)
                loss = cls_loss + ALPHA_DISTILL * distill_loss'''

    NEW_PERCH = '''            if USE_PERCH_DISTILL and perch_teacher is not None:
                # ★ exp081 Perch distill: 2x 5s averaged for 10s alignment
                with torch.no_grad():
                    wav_full = wav.squeeze(1)
                    N = wav_full.shape[1]
                    if N == 160000:
                        # 5s legacy path
                        perch_emb = perch_teacher.embed(wav_full).to(device)
                    elif N == 320000:
                        # 10s: 2x 5s Perch -> average (better alignment than center-5s)
                        wav_a = wav_full[:, :160000]
                        wav_b = wav_full[:, 160000:]
                        emb_a = perch_teacher.embed(wav_a).to(device)
                        emb_b = perch_teacher.embed(wav_b).to(device)
                        perch_emb = (emb_a + emb_b) * 0.5
                    elif N > 160000:
                        start_off = (N - 160000) // 2
                        perch_emb = perch_teacher.embed(wav_full[:, start_off:start_off + 160000]).to(device)
                    else:
                        wav_pad = F.pad(wav_full, (0, 160000 - N))
                        perch_emb = perch_teacher.embed(wav_pad).to(device)
                distill_loss = F.mse_loss(distill_emb, perch_emb)
                loss = cls_loss + ALPHA_DISTILL * distill_loss'''

    if OLD_PERCH in src:
        new_src = src.replace(OLD_PERCH, NEW_PERCH)
        train_cell["source"] = new_src.splitlines(keepends=True)
        print("  [train_loop] Perch distill: center-5s -> 2x 5s averaged (10s alignment)")
        return True
    else:
        print("  [train_loop] WARN: Perch distill anchor not found")
        return False


for nb_path in NB_PATHS:
    if not nb_path.exists():
        print(f"\nSKIP: {nb_path.name} not found")
        continue
    print(f"\n=== Patching {nb_path.name} ===")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    load_data_cell = find_cell(nb["cells"], "load_data")
    train_cell = find_cell(nb["cells"], "train_loop")

    changed = False
    if load_data_cell is not None:
        c = patch_load_data_labeled_sc(load_data_cell)
        changed = changed or c
    if train_cell is not None:
        c = patch_train_loop_perch_distill(train_cell)
        changed = changed or c

    if changed:
        nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  Saved {nb_path.name}")
    else:
        print(f"  No changes for {nb_path.name}")

print("\nDone.")
