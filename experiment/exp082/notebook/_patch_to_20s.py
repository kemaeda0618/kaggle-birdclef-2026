"""Patch exp082 R1+R2 NB: exp081 base (10s) → 20s window.

Changes:
1. CFG: TRAIN_DURATION=20, VAL_DURATION=20, Mel spec → Babych (224/4096/1252)
2. pseudo_infer: N_WINDOWS = 3 (60s / 20s = 3)
3. Labeled SS aggregation: 4× 5s consecutive → 1× 20s (was 2× 5s → 10s)
4. Perch distill: 4× 5s averaged (was 2× 5s → 10s)
5. Paths: exp081 → exp082, Kaggle slug exp082
6. Hdr: update title

Mel spec choice (Babych spec 採用):
  - n_mels=224, n_fft=4096, hop=1252
  - 20s × 32000 / 1252 = 511 frames (manageable size)
  - Babych M3'/M7 で proven (但し Perch distill なしで失敗 → exp082 は Perch あり)

Pseudo source:
  - R2 NB は exp082 R1 pseudo (20s native、3 rows/file) を期待
  - auto-detect で 12 rows (5s) / 6 rows (10s) / 3 rows (20s) 全対応
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


def patch_source(cell, replacements):
    src = "".join(cell["source"])
    for old, new in replacements:
        if old not in src:
            print(f"  WARN: '{old[:60]}' not found in cell {cell.get('id')}")
            continue
        src = src.replace(old, new)
    cell["source"] = src.splitlines(keepends=True)


def patch_hdr(cell, is_r2):
    """Update hdr to exp082 20s."""
    role = "R2" if is_r2 else "R1"
    new_md = f"""# exp082 {role} — exp081 base + 20s window (Colab Pro G4)

**exp082 = exp081 (10s) を Window 20s に拡張**

## Changes from exp081 ({role})
- `TRAIN_DURATION` / `VAL_DURATION`: 10 → **20**
- Mel spec: Tucker (256/2048/512) → **Babych (224/4096/1252)** ★ 20s に最適
  - 20s × 32000 / 1252 = 511 frames (handles long context efficiently)
- `N_WINDOWS` in pseudo_infer: 6 → **3** (60s / 20s = 3 windows/file)
- LabeledSC aggregation: **4× 5s → 1× 20s** (max merge)
- Perch distill: **4× 5s averaged** (was 2× 5s for 10s)
- Output: `output/exp082/{role.lower()}/`, Kaggle slug `birdclef2026-exp082-weights`

## Why 20s
- exp081 (10s) は exp048 と diversity 軸として並行
- 20s は更に長 context、Aves/Amphibia long call を full capture
- Tucker SED + e17 (5s)、exp081 (10s)、exp082 (20s) の 3 軸 temporal diversity 構築

## Past 20s 失敗の正体
- M3' / M7 (LB 0.85) = **Babych init + non-Perch path** が失敗原因
- exp082 = **Perch distill + ImageNet init** で 20s は untested but expected to work

## 不変項目
- Backbone: eca_nfnet_l0 (24M)
- Init: ImageNet (no Babych init)
- LR 3e-4 (NFNet rule)
- Batch: 192 (G4 96GB、Babych mel で frame 511 = 10s 相当)
- N_TOTAL_EPOCHS 25 (R1) / 20 (R2)
- ALPHA_DISTILL 1.0、Perch v2 direct distill

## 想定時間 (G4 96GB)
- 学習: ~6-8h ({role})
- pseudo gen: ~30 min
- Perch 4× cost: +10-20% training time

## 期待
- val_ns22: 0.92-0.94 (long context lift 期待)
- LB single: 0.92-0.94
- Blend (exp048 + exp081 + exp082) +0.003-0.008
"""
    cell["source"] = new_md.splitlines(keepends=True)


def patch_config(cell):
    """20s window + Babych mel spec."""
    patch_source(cell, [
        ("TRAIN_DURATION = 10   # ★ exp081: 5 → 10",
         "TRAIN_DURATION = 20   # ★ exp082: 10 → 20"),
        ("VAL_DURATION   = 10   # ★ exp081: 5 → 10",
         "VAL_DURATION   = 20   # ★ exp082: 10 → 20"),
        # Mel spec: Tucker (256/2048/512) → Babych (224/4096/1252)
        ("N_FFT          = 2048",
         "N_FFT          = 4096   # ★ exp082 Babych spec for 20s"),
        ("HOP_LENGTH     = 512",
         "HOP_LENGTH     = 1252   # ★ exp082 Babych spec (20s × 32k / 1252 = 511 frames)"),
        ("N_MELS         = 256",
         "N_MELS         = 224    # ★ exp082 Babych spec"),
        ("FMIN           = 20",
         "FMIN           = 0      # ★ exp082 Babych spec"),
    ])


def patch_pseudo_infer(cell):
    """N_WINDOWS = 6 → 3 (20s × 3 = 60s)."""
    patch_source(cell, [
        ("N_WINDOWS = 6   # ★ exp081: 10s window で 6 chunks/file",
         "N_WINDOWS = 3   # ★ exp082: 20s window で 3 chunks/file"),
    ])


def patch_labeled_sc_aggregation(cell):
    """LabeledSC: 2x aggregation (10s) → 4x aggregation (20s)."""
    src = "".join(cell["source"])
    OLD = '''        # Non-overlapping 10s windows: pairs (0,1), (2,3), ...
        for k in range(0, len(group_sorted) - 1, 2):
            idx_a = int(group_sorted.iloc[k]["_orig_idx"])
            idx_b = int(group_sorted.iloc[k+1]["_orig_idx"])
            start_sec_10s = float(group_sorted.iloc[k]["start_sec"])
            y_agg = np.maximum(Y_SC_orig[idx_a], Y_SC_orig[idx_b])'''

    NEW = '''        # ★ exp082: Non-overlapping 20s windows: quads (0,1,2,3), (4,5,6,7), (8,9,10,11)
        for k in range(0, len(group_sorted) - 3, 4):
            idx_a = int(group_sorted.iloc[k]["_orig_idx"])
            idx_b = int(group_sorted.iloc[k+1]["_orig_idx"])
            idx_c = int(group_sorted.iloc[k+2]["_orig_idx"])
            idx_d = int(group_sorted.iloc[k+3]["_orig_idx"])
            start_sec_10s = float(group_sorted.iloc[k]["start_sec"])  # 20s window start
            y_agg = np.maximum.reduce([Y_SC_orig[idx_a], Y_SC_orig[idx_b],
                                       Y_SC_orig[idx_c], Y_SC_orig[idx_d]])'''

    if OLD in src:
        new_src = src.replace(OLD, NEW)
        cell["source"] = new_src.splitlines(keepends=True)
        print("  [load_data] LabeledSC 2x → 4x aggregation (20s)")
        return True
    print("  [load_data] WARN: LabeledSC aggregation anchor not found")
    return False


def patch_perch_distill(cell):
    """Perch distill: 2x 5s averaged → 4x 5s averaged."""
    src = "".join(cell["source"])
    OLD = '''            if USE_PERCH_DISTILL and perch_teacher is not None:
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

    NEW = '''            if USE_PERCH_DISTILL and perch_teacher is not None:
                # ★ exp082 Perch distill: 4x 5s averaged for 20s alignment
                with torch.no_grad():
                    wav_full = wav.squeeze(1)
                    N = wav_full.shape[1]
                    if N == 160000:
                        # 5s legacy
                        perch_emb = perch_teacher.embed(wav_full).to(device)
                    elif N == 320000:
                        # 10s: 2x 5s averaged
                        emb_a = perch_teacher.embed(wav_full[:, :160000]).to(device)
                        emb_b = perch_teacher.embed(wav_full[:, 160000:]).to(device)
                        perch_emb = (emb_a + emb_b) * 0.5
                    elif N == 640000:
                        # 20s: 4x 5s Perch -> averaged
                        emb_a = perch_teacher.embed(wav_full[:, :160000]).to(device)
                        emb_b = perch_teacher.embed(wav_full[:, 160000:320000]).to(device)
                        emb_c = perch_teacher.embed(wav_full[:, 320000:480000]).to(device)
                        emb_d = perch_teacher.embed(wav_full[:, 480000:]).to(device)
                        perch_emb = (emb_a + emb_b + emb_c + emb_d) * 0.25
                    elif N > 160000:
                        start_off = (N - 160000) // 2
                        perch_emb = perch_teacher.embed(wav_full[:, start_off:start_off + 160000]).to(device)
                    else:
                        wav_pad = F.pad(wav_full, (0, 160000 - N))
                        perch_emb = perch_teacher.embed(wav_pad).to(device)
                distill_loss = F.mse_loss(distill_emb, perch_emb)
                loss = cls_loss + ALPHA_DISTILL * distill_loss'''

    if OLD in src:
        new_src = src.replace(OLD, NEW)
        cell["source"] = new_src.splitlines(keepends=True)
        print("  [train_loop] Perch distill: 2x → 4x 5s averaged (20s)")
        return True
    print("  [train_loop] WARN: Perch distill anchor not found")
    return False


def patch_pseudo_verify_r2(cell):
    """R2 pseudo verify: handle 3 rows/file (20s native) too."""
    src = "".join(cell["source"])
    if "20s native" in src:
        print("  [load_data] pseudo verify for 20s already added, skipping")
        return False
    OLD = '''if _n_per_file.median() == 6:
    print("  [exp081] OK: 10s native pseudo (6 rows/file from exp081 R1)")
elif _n_per_file.median() == 12:'''
    NEW = '''if _n_per_file.median() == 3:
    print("  [exp082] OK: 20s native pseudo (3 rows/file from exp082 R1)")
elif _n_per_file.median() == 6:
    print("  [exp082] WARN: 10s pseudo detected (6 rows/file). Aggregating to 20s...")
    import numpy as _np_e82
    import pandas as _pd_e82
    _psd_full = pseudo_meta.copy()
    _psd_full["_orig_idx"] = _np_e82.arange(len(_psd_full))
    _psd_full = _psd_full.sort_values(["filename", "start_sec"]).reset_index(drop=True)
    new_meta_rows = []
    new_Y_20s = []
    for fname, group in _psd_full.groupby("filename"):
        group_sorted = group.reset_index(drop=True)
        for k in range(0, len(group_sorted) - 1, 2):
            idx_a = int(group_sorted.iloc[k]["_orig_idx"])
            idx_b = int(group_sorted.iloc[k+1]["_orig_idx"])
            start_sec_20s = float(group_sorted.iloc[k]["start_sec"])
            y_agg = _np_e82.maximum(Y_PSEUDO[idx_a], Y_PSEUDO[idx_b])
            new_meta_rows.append({"filename": fname, "start_sec": start_sec_20s})
            new_Y_20s.append(y_agg)
    pseudo_meta = _pd_e82.DataFrame(new_meta_rows).reset_index(drop=True)
    Y_PSEUDO = _np_e82.array(new_Y_20s, dtype=_np_e82.float32)
    print(f"  After 10s -> 20s aggregation: {len(pseudo_meta)} rows")
elif _n_per_file.median() == 12:'''
    if OLD in src:
        new_src = src.replace(OLD, NEW)
        # Also update the existing 12 case to aggregate 4x (was 2x)
        OLD12 = '''elif _n_per_file.median() == 12:
    print("  [exp081] WARN: 5s pseudo detected (12 rows/file). Aggregating to 10s...")
    import numpy as _np_e81
    import pandas as _pd_e81
    _psd_full = pseudo_meta.copy()
    _psd_full["_orig_idx"] = _np_e81.arange(len(_psd_full))
    _psd_full = _psd_full.sort_values(["filename", "start_sec"]).reset_index(drop=True)
    new_meta_rows = []
    new_Y_10s = []
    for fname, group in _psd_full.groupby("filename"):
        group_sorted = group.reset_index(drop=True)
        for k in range(0, len(group_sorted) - 1, 2):
            idx_a = int(group_sorted.iloc[k]["_orig_idx"])
            idx_b = int(group_sorted.iloc[k+1]["_orig_idx"])
            start_sec_10s = float(group_sorted.iloc[k]["start_sec"])
            y_agg = _np_e81.maximum(Y_PSEUDO[idx_a], Y_PSEUDO[idx_b])
            new_meta_rows.append({"filename": fname, "start_sec": start_sec_10s})
            new_Y_10s.append(y_agg)
    pseudo_meta = _pd_e81.DataFrame(new_meta_rows).reset_index(drop=True)
    Y_PSEUDO = _np_e81.array(new_Y_10s, dtype=_np_e81.float32)
    print(f"  After aggregation: {len(pseudo_meta)} rows, Y shape {Y_PSEUDO.shape}")'''
        NEW12 = '''elif _n_per_file.median() == 12:
    print("  [exp082] WARN: 5s pseudo detected (12 rows/file). Aggregating to 20s (4x)...")
    import numpy as _np_e82_b
    import pandas as _pd_e82_b
    _psd_full = pseudo_meta.copy()
    _psd_full["_orig_idx"] = _np_e82_b.arange(len(_psd_full))
    _psd_full = _psd_full.sort_values(["filename", "start_sec"]).reset_index(drop=True)
    new_meta_rows = []
    new_Y_20s = []
    for fname, group in _psd_full.groupby("filename"):
        group_sorted = group.reset_index(drop=True)
        # 12 5s rows -> 3 20s rows (4x aggregation)
        for k in range(0, len(group_sorted) - 3, 4):
            idx_a = int(group_sorted.iloc[k]["_orig_idx"])
            idx_b = int(group_sorted.iloc[k+1]["_orig_idx"])
            idx_c = int(group_sorted.iloc[k+2]["_orig_idx"])
            idx_d = int(group_sorted.iloc[k+3]["_orig_idx"])
            start_sec_20s = float(group_sorted.iloc[k]["start_sec"])
            y_agg = _np_e82_b.maximum.reduce([Y_PSEUDO[idx_a], Y_PSEUDO[idx_b],
                                              Y_PSEUDO[idx_c], Y_PSEUDO[idx_d]])
            new_meta_rows.append({"filename": fname, "start_sec": start_sec_20s})
            new_Y_20s.append(y_agg)
    pseudo_meta = _pd_e82_b.DataFrame(new_meta_rows).reset_index(drop=True)
    Y_PSEUDO = _np_e82_b.array(new_Y_20s, dtype=_np_e82_b.float32)
    print(f"  After 5s -> 20s aggregation: {len(pseudo_meta)} rows, Y shape {Y_PSEUDO.shape}")'''
        if OLD12 in new_src:
            new_src = new_src.replace(OLD12, NEW12)
        cell["source"] = new_src.splitlines(keepends=True)
        print("  [load_data] R2 pseudo verify: added 20s native + 5s→20s, 10s→20s aggregation")
        return True
    return False


for nb_path in NB_PATHS:
    if not nb_path.exists():
        print(f"\nSKIP: {nb_path.name} not found")
        continue
    is_r2 = "_r2" in nb_path.name
    print(f"\n=== Patching {nb_path.name} (is_r2={is_r2}) ===")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # 1. hdr
    hdr = find_cell(nb["cells"], "hdr")
    if hdr is not None:
        patch_hdr(hdr, is_r2)
        print("  [hdr] Updated for exp082 20s")

    # 2. setup
    setup = find_cell(nb["cells"], "setup")
    if setup is not None:
        patch_source(setup, [("exp081", "exp082")])
        print("  [setup] paths exp081 -> exp082")

    # 3. config
    config = find_cell(nb["cells"], "config")
    if config is not None:
        patch_config(config)
        print("  [config] Window + Babych mel spec")

    # 4. pseudo_infer (R1 only)
    pseudo_infer = find_cell(nb["cells"], "pseudo_infer")
    if pseudo_infer is not None:
        patch_pseudo_infer(pseudo_infer)
        print("  [pseudo_infer] N_WINDOWS 6 -> 3")

    # 5. load_data labeled SS aggregation
    load_data = find_cell(nb["cells"], "load_data")
    if load_data is not None:
        patch_labeled_sc_aggregation(load_data)
        # R2 only: patch pseudo verify
        if is_r2:
            patch_pseudo_verify_r2(load_data)

    # 6. train_loop Perch distill
    train_loop = find_cell(nb["cells"], "train_loop")
    if train_loop is not None:
        patch_perch_distill(train_loop)

    # 7. upload slug
    upload = find_cell(nb["cells"], "upload" if not is_r2 else "upload_state")
    if upload is None:
        upload = find_cell(nb["cells"], "upload_state")
    if upload is None:
        upload = find_cell(nb["cells"], "upload")
    if upload is not None:
        patch_source(upload, [
            ("birdclef2026-exp081-weights", "birdclef2026-exp082-weights"),
            ("exp081", "exp082"),
        ])
        print("  [upload] Slug exp081 -> exp082")

    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  Saved {nb_path.name}")

print("\nDone.")
