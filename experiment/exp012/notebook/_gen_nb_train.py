"""Generate exp012 nb_train.ipynb based on Tucker's bc2026-distilled-sed.

Strategy for v1 (this version):
  - Faithful reproduction of Tucker's pipeline cell-by-cell, with these changes:
    * MODE = "train" (we are training, not just inference)
    * DEBUG = False (we want to actually train)
    * FOLDS = [0]  -- single fold per Kaggle session (Kaggle 12hr cap)
    * EPOCHS = 25, BATCH = 16 (Tucker's Kaggle setting)
    * Add resumable mechanism: copy prev kernel /kaggle/working/ → start
    * Add per-epoch checkpoint save (atomic) so we can resume mid-fold
    * Time-budget guard: exit cleanly if < 30 min remain
    * Skip the inference cells at the end (we'll add separate nb_submit.ipynb later)

Inputs (Kaggle data sources):
  - competition: birdclef-2026
  - dataset:  tuckerarrants/birdclef-2026-waveform-cache
  - dataset:  tuckerarrants/perch-v2-no-dft-onnx

Outputs (in /kaggle/working):
  - fold0_best_ns22.pt
  - fold0_best_macro.pt
  - fold0_history.json
  - fold0_ep{N}.pt  (resume checkpoints, deleted after fold completes)
  - sed_distill_fold0.onnx (if fold completes)

Run: python _gen_nb_train.py  -> writes nb_train.ipynb
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

# Load Tucker reference notebook
ref_path = HERE / "_reference_tucker.ipynb"
ref_nb = json.loads(ref_path.read_text(encoding="utf-8"))

# Helper to make a code cell
def code_cell(source, cid):
    if isinstance(source, str):
        lines = source.split("\n")
        src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    else:
        src = source
    return {"cell_type": "code", "id": cid, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}

def md_cell(source, cid):
    if isinstance(source, str):
        lines = source.split("\n")
        src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    else:
        src = source
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}


# Walk through Tucker's cells, modify or replace as needed
# We rebuild the cell list with a new structure:
#   1. Header md (replaced)
#   2. Resume preamble (NEW)
#   3. S0 install (kept)
#   4. S1 imports + config (modified: MODE="train", DEBUG=False, FOLDS=[0])
#   5. S2 load data (kept)
#   6. (skip DEBUG block)
#   7. S3 model (kept)
#   8. S4 data pipeline (kept)
#   9. S5 training (modified: add resumable, time guard)
#   10. S6 fold loop + ONNX export (modified: resume + early-exit)
#   11. S7 OOF eval (kept)
#   12. (skip inference cells 19-26)

new_cells = []

# 1. Header
new_cells.append(md_cell(r"""# exp012 — Distilled SED Training (Tucker base, single-fold resumable)

Faithful reproduction of `tuckerarrants/bc2026-distilled-sed` for our training pipeline. Single fold per Kaggle session (12hr cap), resumable via kernel_source self-reference.

## Key changes from Tucker's NB
- `MODE = "train"`, `DEBUG = False`
- `FOLDS = [0]` — train only fold 0 in this notebook
- Resumable: copies prev kernel `/kaggle/working/` outputs into current `/kaggle/working/`
- Per-epoch atomic checkpoint save → resume mid-fold
- Time guard: exit cleanly if < 30 min remain (`SOFT_LIMIT_SEC = 11h30m`)
- Skips inference cells (we'll build a separate `nb_submit.ipynb`)

## Inputs
- `birdclef-2026` (competition)
- `tuckerarrants/birdclef-2026-waveform-cache` (waveform `.pt` int16 cache + metadata)
- `tuckerarrants/perch-v2-no-dft-onnx` (Perch ONNX teacher)
- (resume) `maekeso/birdclef2026-exp012-train-fold0` (self)

## Outputs
- `fold0_best_ns22.pt`, `fold0_best_macro.pt`
- `fold0_history.json`
- `sed_distill_fold0.onnx`
""", "hdr"))

# 2. Resume preamble (NEW)
new_cells.append(code_cell(r"""# ============================================================
# Resume preamble: copy prev kernel outputs into /kaggle/working
# ============================================================
import os, time, json, shutil
from pathlib import Path

WALL_START = time.time()
SOFT_LIMIT_SEC  = 11 * 3600 + 30 * 60   # 11h30m
EXIT_MARGIN_SEC = 30 * 60                # exit if < 30 min left

def time_left():
    return SOFT_LIMIT_SEC - (time.time() - WALL_START)

def time_low():
    return time_left() < EXIT_MARGIN_SEC

WORK_DIR = Path("/kaggle/working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Resume from previous version's output if mounted as kernel_source
PREV_KERNEL_CANDIDATES = [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp012-train-fold0"),
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-birdclef2026-exp012-train-fold0"),
]
PREV_KERNEL = next((p for p in PREV_KERNEL_CANDIDATES if p.exists()), None)
if PREV_KERNEL is not None:
    print(f"PREV_KERNEL: {PREV_KERNEL}")
    n = 0
    for src in PREV_KERNEL.rglob("*"):
        if not src.is_file(): continue
        rel = src.relative_to(PREV_KERNEL)
        if rel.name.startswith("__") or rel.suffix in (".log", ".html"):
            continue
        if "kernel-metadata.json" in str(rel):
            continue
        dst = WORK_DIR / rel
        if dst.exists(): continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst); n += 1
        except Exception as e:
            print(f"  copy err {rel}: {e}")
    print(f"  copied {n} files from prev kernel")
else:
    print("PREV_KERNEL not mounted (first run)")

print(f"WORK_DIR: {sorted(p.name for p in WORK_DIR.iterdir())[:30]}")
""", "resume"))

# Now walk Tucker's cells and pick what we need
ref_cells = ref_nb["cells"]

# helper: source as str
def cell_src(c):
    s = c["source"]
    return s if isinstance(s, str) else "".join(s)

# Cell 2: install onnxruntime wheel
c2 = ref_cells[2]
new_cells.append(code_cell(cell_src(c2), "install_ort"))

# Cell 4: imports + config — MODIFY
c4_src = cell_src(ref_cells[4])
# Replace: MODE="infer" -> MODE="train"; DEBUG=True branch; FOLDS=[0,1,2,3,4] -> [0]
import re
c4_src = c4_src.replace('MODE = "infer"', 'MODE = "train"')
# Whitespace-tolerant DEBUG branch replacement
c4_src = re.sub(
    r"if\s+MODE\s*==\s*['\"]train['\"][^\n]*\n\s+DEBUG\s*=\s*True[^\n]*\n\s*else\s*:\s*\n\s+DEBUG\s*=\s*False",
    "DEBUG = False  # exp012: full training, not debug",
    c4_src, count=1)
c4_src = c4_src.replace("FOLDS  = [0, 1, 2, 3, 4]", "FOLDS  = [0]  # exp012: single fold per Kaggle session")
# T4 single-GPU safety: 64 may OOM at 256 mels; use 32
c4_src = c4_src.replace("BATCH  = 16 if DEBUG else 64", "BATCH  = 32  # exp012: T4 16GB safety")
assert 'MODE = "train"' in c4_src, "MODE replace failed"
assert "FOLDS  = [0]" in c4_src and "[0, 1, 2, 3, 4]" not in c4_src, "FOLDS replace failed"
assert "DEBUG = False  # exp012" in c4_src and "DEBUG = True" not in c4_src, "DEBUG replace failed"
new_cells.append(code_cell(c4_src, "config"))

# Cell 5: optional install for train mode (kept)
new_cells.append(code_cell(cell_src(ref_cells[5]), "install_train_pkgs"))

# Cell 7: load data (kept)
new_cells.append(code_cell(cell_src(ref_cells[7]), "load_data"))

# Cell 8: DEBUG override — SKIP (DEBUG=False)
# (intentionally omitted)

# Cell 10: model definitions (kept)
new_cells.append(code_cell(cell_src(ref_cells[10]), "model_defs"))

# Cell 12: data pipeline (kept)
new_cells.append(code_cell(cell_src(ref_cells[12]), "data_pipeline"))

# Cell 14: training function — MODIFY for resumable + time guard
c14_src = cell_src(ref_cells[14])
# Insert resume + time-guard logic into train_fold:
# Strategy: replace the original train_fold definition with our resumable version
# Simpler: append an override block AFTER the original definition

# Original train_fold is defined here. We'll patch by appending an override.
new_cells.append(code_cell(c14_src, "train_fn_orig"))

# Patch cell: redefine train_fold with resume + time-guard
patch = r"""# ============================================================
# Patch: train_fold with resume + time-guard
# ============================================================
import sys

def train_fold(fold_k):
    vm = sc_cache_meta["fold"].values == fold_k
    Y_val = Y_SC[vm]
    ns22_val = non_s22_mask_sc[vm]
    val_sc_df = sc_cache_meta[vm].reset_index(drop=True)

    active = build_active_datasets(fold_k)
    names, datasets, sizes = zip(*active)
    mds = ConcatDataset(list(datasets))
    nst = max(100, int(sum(sizes) / BATCH))

    print(f"  Streams: {dict(zip(names, sizes))}  steps/ep: {nst}")

    m = make_model()
    mel_transform = MelSpecTransform().to(device)
    spec_augment = SpecAugment().to(device)
    perch_teacher = PerchTeacher(PERCH_ONNX_PATH,
                                  "cuda" if torch.cuda.is_available() else "cpu") \
                    if USE_PERCH_DISTILL else None

    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    scaler = GradScaler()
    warmup_steps = nst * WARMUP_EPOCHS
    total_steps  = nst * EPOCHS
    warmup_sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1/25, end_factor=1.0,
                                                      total_iters=warmup_steps)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps,
                                                               eta_min=1e-6)
    sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup_sched, cosine_sched],
                                                 milestones=[warmup_steps])

    history = {"ep": [], "train_loss": [], "cls_loss": [], "dist_loss": [],
               "macro": [], "ns22_macro": [],
               "ns22_Aves": [], "ns22_Amphibia": [], "ns22_Insecta": [], "ns22_Mammalia": [],
               "val_preds": []}
    best_ns22, best_state_ns22 = -1.0, None
    best_macro, best_state_macro = -1.0, None
    val_wavs = _load_val_waveforms(val_sc_df)

    # ---- Resume support ----
    HIST_PATH  = WORK_DIR / f"fold{fold_k}_history.json"
    NS22_PATH  = WORK_DIR / f"fold{fold_k}_best_ns22.pt"
    MACRO_PATH = WORK_DIR / f"fold{fold_k}_best_macro.pt"
    LAST_PATH  = WORK_DIR / f"fold{fold_k}_last.pt"
    start_ep = 0
    if HIST_PATH.exists():
        try:
            saved = json.loads(HIST_PATH.read_text())
            for k in history.keys():
                if k != "val_preds" and k in saved:
                    history[k] = list(saved[k])
            start_ep = (max(history["ep"]) + 1) if history["ep"] else 0
            best_ns22  = max(history["ns22_macro"]) if history["ns22_macro"] else -1.0
            best_macro = max(history["macro"])      if history["macro"]      else -1.0
            print(f"  RESUME from epoch {start_ep} (best_ns22={best_ns22:.4f}, best_macro={best_macro:.4f})")
        except Exception as e:
            print(f"  RESUME failed: {e}; restarting")
            start_ep = 0
    if LAST_PATH.exists() and start_ep > 0:
        try:
            ckpt = torch.load(str(LAST_PATH), map_location=device, weights_only=False)
            m.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            sch.load_state_dict(ckpt["sch"])
            scaler.load_state_dict(ckpt["scaler"])
            print(f"  loaded last.pt (epoch {ckpt.get('epoch')})")
        except Exception as e:
            print(f"  load last.pt failed: {e}; restarting")
            start_ep = 0
    if NS22_PATH.exists():
        try:
            best_state_ns22 = torch.load(str(NS22_PATH), map_location="cpu", weights_only=False)
        except Exception:
            pass
    if MACRO_PATH.exists():
        try:
            best_state_macro = torch.load(str(MACRO_PATH), map_location="cpu", weights_only=False)
        except Exception:
            pass

    if start_ep >= EPOCHS:
        print(f"  fold {fold_k} already finished {start_ep} epochs")
        return best_state_ns22, best_state_macro, history

    for ep in range(start_ep, EPOCHS):
        if time_low():
            print(f"  TIME LOW at start of ep{ep}, exiting cleanly")
            sys.exit(0)
        m.train()
        smp = MixSamp(list(sizes), list(names), SHARES, BATCH, nst, seed=42 + ep)
        tl = DataLoader(mds, batch_sampler=smp, collate_fn=collate_m,
                        num_workers=0, pin_memory=True)
        el, el_cls, el_dist, nb_count = 0.0, 0.0, 0.0, 0
        t0 = time.time()

        for wav, lb, wt, mk, sr in tl:
            wav, lb, wt, mk = wav.to(device), lb.to(device), wt.to(device), mk.to(device)
            sw = mk_sw(sr).to(device)

            with torch.no_grad():
                mel = mel_transform(wav)
                B = mel.size(0)
                for i in range(B):
                    mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
                mel = spec_augment(mel)

            with autocast():
                if USE_PERCH_DISTILL:
                    clip_logits, framewise, distill_emb = m(mel, return_framewise=True,
                                                            return_distill=True)
                else:
                    clip_logits, framewise = m(mel, return_framewise=True)

                frame_max_logits = framewise.max(dim=1).values

                bce_clip = F.binary_cross_entropy_with_logits(clip_logits, lb, reduction="none")
                bce_frame = F.binary_cross_entropy_with_logits(frame_max_logits, lb, reduction="none")
                bce = 0.5 * bce_clip + 0.5 * bce_frame
                ps = (bce * wt * mk).sum(1) / (mk.sum(1) + 1e-8)
                cls_loss = (ps * sw).mean()

                if USE_PERCH_DISTILL and perch_teacher is not None:
                    with torch.no_grad():
                        wav_5s = wav.squeeze(1)
                        N = wav_5s.shape[1]
                        if N > 160000:
                            start = (N - 160000) // 2
                            wav_5s = wav_5s[:, start:start + 160000]
                        elif N < 160000:
                            wav_5s = F.pad(wav_5s, (0, 160000 - N))
                        perch_emb = perch_teacher.embed(wav_5s).to(device)
                    distill_loss = F.mse_loss(distill_emb, perch_emb)
                    loss = cls_loss + ALPHA_DISTILL * distill_loss
                else:
                    distill_loss = torch.tensor(0.0)
                    loss = cls_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            el += loss.item(); el_cls += cls_loss.item()
            el_dist += distill_loss.item(); nb_count += 1

        # Validation
        val_preds_dict = _predict_from_waveforms(m, mel_transform, val_wavs)
        val_preds = val_preds_dict["blend"]
        r = full_eval(Y_val, val_preds, ns22_val, TAXON_MASKS)

        history["ep"].append(ep)
        history["train_loss"].append(round(el / nb_count, 5))
        history["cls_loss"].append(round(el_cls / nb_count, 5))
        history["dist_loss"].append(round(el_dist / nb_count, 5))
        history["macro"].append(r["macro_auc_all"])
        history["ns22_macro"].append(r["non_s22_macro"])
        for t in ["Aves", "Amphibia", "Insecta", "Mammalia"]:
            history[f"ns22_{t}"].append(r[f"non_s22_{t}"])

        tag_ns22 = ""; tag_macro = ""
        if r["non_s22_macro"] > best_ns22:
            best_ns22 = r["non_s22_macro"]
            best_state_ns22 = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            torch.save(best_state_ns22, str(NS22_PATH))
            tag_ns22 = " *ns22"
        if r["macro_auc_all"] > best_macro:
            best_macro = r["macro_auc_all"]
            best_state_macro = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            torch.save(best_state_macro, str(MACRO_PATH))
            tag_macro = " *macro"

        # Atomic save: history + last.pt
        tmp_hist = HIST_PATH.with_suffix(".tmp.json")
        tmp_hist.write_text(json.dumps({k: v for k, v in history.items() if k != "val_preds"}))
        os.replace(str(tmp_hist), str(HIST_PATH))

        tmp_last = LAST_PATH.with_suffix(".tmp.pt")
        torch.save({"epoch": ep, "model": m.state_dict(),
                    "opt": opt.state_dict(), "sch": sch.state_dict(),
                    "scaler": scaler.state_dict()}, str(tmp_last))
        os.replace(str(tmp_last), str(LAST_PATH))

        dist_str = f" dist={el_dist/nb_count:.4f}" if USE_PERCH_DISTILL else ""
        print(f"    Ep{ep:02d}: loss={el/nb_count:.4f} cls={el_cls/nb_count:.4f}{dist_str} "
              f"lr={opt.param_groups[0]['lr']:.1e} | "
              f"ns22: {r['non_s22_macro']:.4f} | "
              f"Av={r['non_s22_Aves']:.4f} Am={r['non_s22_Amphibia']:.4f} "
              f"In={r['non_s22_Insecta']:.4f} Ma={r['non_s22_Mammalia']:.4f} "
              f"[{time.time()-t0:.0f}s]{tag_ns22}{tag_macro}  time_left={time_left()/3600:.1f}h")

    # Cleanup last.pt at end of training (saves space)
    if LAST_PATH.exists():
        LAST_PATH.unlink()

    del perch_teacher, m, mel_transform, spec_augment
    torch.cuda.empty_cache(); gc.collect()
    return best_state_ns22, best_state_macro, history

print("OK Resumable training function ready")
"""
new_cells.append(code_cell(patch, "train_fn_resumable"))

# Cell 16: fold loop + ONNX export (kept; train_fold is now resumable)
new_cells.append(code_cell(cell_src(ref_cells[16]), "fold_loop"))

# Cell 18: OOF eval (kept)
new_cells.append(code_cell(cell_src(ref_cells[18]), "oof_eval"))

# Skip inference cells (19-26)
new_cells.append(md_cell(r"""## Done

Inference is in a separate `nb_submit.ipynb`. This notebook only trains and exports ONNX.
""", "footer"))

# Build final notebook
nb_out = {
    "cells": new_cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.12",
                          "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3", "nbconvert_exporter": "python",
                          "file_extension": ".py"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out_path = HERE / "nb_train.ipynb"
out_path.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote: {out_path}")
print(f"Cells: {len(new_cells)}")
print(f"Size: {out_path.stat().st_size/1024:.1f} KB")
