"""NaN loss fix patch:
1. Remove AMP (autocast + GradScaler) - use fp32 throughout
2. Add gradient clipping (max_norm=1.0)
3. Add NaN skip safety
4. Better mel normalize (clamp + standardize)
"""
import json
from pathlib import Path

NBS = [
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp049\notebook\nb_train_effv2s_r1.ipynb"),
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp050\notebook\nb_train_convnext_r1.ipynb"),
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp051\notebook\nb_train_swin_r1.ipynb"),
]

# New train_loop without AMP
NEW_TRAIN_LOOP = '''# ============================================================
# Cell 10: Training loop (fp32, no AMP, NaN-safe)
# ============================================================
import sys
def _p(msg):
    print(msg, flush=True)
    sys.stdout.flush()

# Smoke test (fp32 mode)
_p(f"[smoke] creating train_loader iter...")
import time
_st = time.time()
smoke_iter = iter(train_loader)
_p(f"[smoke] iter created in {time.time()-_st:.1f}s, fetching batch...")
_st = time.time()
test_wav, test_label = next(smoke_iter)
_p(f"[smoke] batch fetched in {time.time()-_st:.1f}s: wav={test_wav.shape} label_sum={test_label.sum().item():.0f}")

_p(f"[smoke] move to GPU + mel + forward + backward (fp32 mode)...")
_st = time.time()
test_wav_g = test_wav.to(DEVICE)
test_label_g = test_label.to(DEVICE)
test_mel = mel_extractor(test_wav_g)
_p(f"[smoke] mel: shape={test_mel.shape}, range=[{test_mel.min().item():.2f}, {test_mel.max().item():.2f}], mean={test_mel.mean().item():.2f}")
test_logit, _ = model(test_mel)
_p(f"[smoke] logit: shape={test_logit.shape}, range=[{test_logit.min().item():.2f}, {test_logit.max().item():.2f}]")
test_loss = loss_fn(test_logit, test_label_g)
_p(f"[smoke] loss={test_loss.item():.4f}")
test_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
torch.cuda.synchronize()
_p(f"[smoke] full step in {time.time()-_st:.1f}s")
del smoke_iter, test_wav, test_label, test_wav_g, test_label_g, test_mel, test_logit, test_loss
torch.cuda.empty_cache()
_p(f"\\n=== SMOKE TEST PASSED, starting main training ===\\n")

# Main training loop
history = {"train_loss": [], "val_ns22": [], "lr": [], "elapsed_min": []}
best_val = 0.0
start_t = time.time()
nan_skip_count = 0

for epoch in range(1, CFG.EPOCHS + 1):
    ep_start = time.time()
    _p(f"[ep{epoch}] start, model.train()...")
    model.train()
    train_losses = []
    _p(f"[ep{epoch}] iterating train_loader (steps/ep={len(train_loader)})...")
    pbar = tqdm(train_loader, desc=f"Ep {epoch}/{CFG.EPOCHS}", leave=False, file=sys.stdout)
    for batch_idx, (wav, label) in enumerate(pbar):
        if batch_idx == 0:
            _p(f"[ep{epoch}] first batch in {time.time()-ep_start:.1f}s")
        wav = wav.to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)
        mel = mel_extractor(wav)
        mel, label_mix = spec_mixup(mel, label)
        mel = spec_augment(mel)
        logit, _ = model(mel)
        loss = loss_fn(logit, label_mix)

        # NaN safety
        if torch.isnan(loss) or torch.isinf(loss):
            nan_skip_count += 1
            optimizer.zero_grad(set_to_none=True)
            if batch_idx % 50 == 0:
                _p(f"[ep{epoch}] NaN/Inf at step {batch_idx}, skipping (total skips={nan_skip_count})")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_losses.append(loss.item())
        if batch_idx % 50 == 0:
            _p(f"[ep{epoch}] step {batch_idx}/{len(train_loader)} loss={np.mean(train_losses[-50:]):.4f}")
            torch.cuda.empty_cache()
        del mel, logit, loss
        if "label_mix" in dir():
            del label_mix
    scheduler.step()
    torch.cuda.empty_cache()

    train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
    _p(f"[ep{epoch}] train done (loss={train_loss:.4f}), evaluating val...")
    val_ns22 = evaluate(model, mel_extractor, val_loader)
    lr_now = optimizer.param_groups[0]["lr"]
    elapsed_min = (time.time() - start_t) / 60
    ep_min = (time.time() - ep_start) / 60

    history["train_loss"].append(train_loss)
    history["val_ns22"].append(val_ns22)
    history["lr"].append(lr_now)
    history["elapsed_min"].append(elapsed_min)

    is_best = val_ns22 > best_val
    if is_best:
        best_val = val_ns22
        torch.save(
            {"epoch": epoch, "val_ns22": val_ns22,
             "state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            },
            CFG.BEST_CKPT,
        )

    json.dump(history, open(CFG.HIST_JSON, "w"), indent=2)
    _p(f"=== Ep {epoch}/{CFG.EPOCHS}: loss={train_loss:.4f} val_ns22={val_ns22:.4f}"
       f" {'BEST' if is_best else ''} lr={lr_now:.2e} ({ep_min:.1f}min, total {elapsed_min:.1f}min, nan_skips={nan_skip_count}) ===")

# Save final
torch.save(
    {"epoch": CFG.EPOCHS, "val_ns22": history["val_ns22"][-1],
     "state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
    },
    CFG.LAST_CKPT,
)
_p(f"\\nTraining DONE. Best val_ns22={best_val:.4f}, nan_skips={nan_skip_count}")
'''

# Also fix model_setup cell - remove GradScaler (no AMP)
NEW_MODEL_SETUP_BLOCK = '''optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
# NOTE: AMP removed (caused NaN loss in early iterations). Using fp32 throughout.
# scaler = GradScaler()  # disabled
loss_fn = nn.BCEWithLogitsLoss()'''

# Also fix mel normalize - safer range
NEW_MEL_FORWARD = '''    def forward(self, wav):
        # wav: (B, T) -> mel: (B, n_mels, T'), then dB, then normalize
        mel = self.mel(wav)
        mel = self.db(mel)
        # Better normalization: clamp to safe range, then standardize
        mel = torch.clamp(mel, min=-80.0, max=0.0)
        mel = (mel + 40.0) / 40.0  # range approx [-1, 1] but more stable
        return mel'''

for NB in NBS:
    if not NB.exists():
        print(f"[SKIP] {NB.name}")
        continue
    nb = json.load(open(NB, encoding="utf-8"))
    n_changes = 0

    # 1. Replace train_loop entirely
    for c in nb["cells"]:
        if c.get("id") == "train_loop":
            c["source"] = NEW_TRAIN_LOOP.splitlines(keepends=True)
            n_changes += 1
            break

    # 2. Fix model_setup: remove GradScaler
    for c in nb["cells"]:
        if c.get("id") == "model_setup":
            src = "".join(c["source"])
            # Remove scaler line
            if "scaler = GradScaler()" in src:
                # Replace whole block including comment
                src = src.replace(
                    """optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
scaler = GradScaler()
loss_fn = nn.BCEWithLogitsLoss()""",
                    NEW_MODEL_SETUP_BLOCK
                )
                n_changes += 1
            c["source"] = src.splitlines(keepends=True)
            break

    # 3. Fix MelExtractor.forward to use safer normalize
    for c in nb["cells"]:
        if c.get("id") == "dataset":
            src = "".join(c["source"])
            old = """    def forward(self, wav):
        # wav: (B, T) -> mel: (B, n_mels, T'), then dB
        mel = self.mel(wav)
        mel = self.db(mel)
        # Normalize to [-1, 1] approx
        mel = (mel + 80) / 80 * 2 - 1
        return mel"""
            if old in src:
                src = src.replace(old, NEW_MEL_FORWARD)
                n_changes += 1
            c["source"] = src.splitlines(keepends=True)
            break

    # 4. Fix eval_fn: remove autocast
    for c in nb["cells"]:
        if c.get("id") == "eval_fn":
            src = "".join(c["source"])
            if "with autocast():" in src:
                src = src.replace(
                    """        with autocast():
            mel = mel_ex(wav)
            logit, _ = model(mel)""",
                    """        mel = mel_ex(wav)
        logit, _ = model(mel)"""
                )
                n_changes += 1
            c["source"] = src.splitlines(keepends=True)
            break

    json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{NB.parent.parent.name}] {n_changes} changes (AMP removed, NaN safety, mel fix)")
