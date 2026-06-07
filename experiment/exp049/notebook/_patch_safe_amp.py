"""v7 patch: re-enable safe AMP + DataParallel + big batch.

Goal: 20 ep in <9h Kaggle session.
- AMP on with init_scale=2**10 (avoid initial overflow)
- DataParallel on (2 T4)
- Batch 128 (effective)
- NaN skip + grad clip after unscale (safety nets keep)
"""
import json
from pathlib import Path

NB_BATCH = [
    (Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp049\notebook\nb_train_effv2s_r1.ipynb"), 128),
    (Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp050\notebook\nb_train_convnext_r1.ipynb"), 192),  # ConvNeXt smaller
    (Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp051\notebook\nb_train_swin_r1.ipynb"), 96),  # Swin heavy
]

NEW_TRAIN_LOOP = '''# ============================================================
# Cell 10: Training loop (safe AMP, DP, NaN-safe, 20 ep target)
# ============================================================
import sys
from torch.amp import autocast, GradScaler
def _p(msg):
    print(msg, flush=True)
    sys.stdout.flush()

# Initialize GradScaler with smaller init_scale to avoid initial overflow
scaler = GradScaler("cuda", init_scale=2**10)

# Smoke test
_p(f"[smoke] creating train_loader iter...")
import time
_st = time.time()
smoke_iter = iter(train_loader)
_p(f"[smoke] iter created in {time.time()-_st:.1f}s, fetching batch...")
_st = time.time()
test_wav, test_label = next(smoke_iter)
_p(f"[smoke] batch fetched in {time.time()-_st:.1f}s: wav={test_wav.shape} label_sum={test_label.sum().item():.0f}")

_p(f"[smoke] forward + backward with safe AMP...")
_st = time.time()
test_wav_g = test_wav.to(DEVICE)
test_label_g = test_label.to(DEVICE)
test_mel = mel_extractor(test_wav_g)
_p(f"[smoke] mel: shape={test_mel.shape}, range=[{test_mel.min().item():.2f}, {test_mel.max().item():.2f}], mean={test_mel.mean().item():.2f}")
with autocast("cuda", dtype=torch.float16):
    test_logit, _ = model(test_mel)
    test_loss = loss_fn(test_logit, test_label_g)
_p(f"[smoke] logit: shape={test_logit.shape}, range=[{test_logit.min().item():.2f}, {test_logit.max().item():.2f}], loss={test_loss.item():.4f}")
if not (torch.isnan(test_loss) or torch.isinf(test_loss)):
    scaler.scale(test_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    _p(f"[smoke] full step in {time.time()-_st:.1f}s, OK")
else:
    _p(f"[smoke] WARN: NaN/Inf in smoke loss, but continuing training (will be skipped)")
    optimizer.zero_grad(set_to_none=True)
del smoke_iter, test_wav, test_label, test_wav_g, test_label_g, test_mel, test_logit, test_loss
torch.cuda.empty_cache()
_p(f"\\n=== SMOKE TEST DONE, starting main training ===\\n")

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
        with autocast("cuda", dtype=torch.float16):
            logit, _ = model(mel)
            loss = loss_fn(logit, label_mix)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_skip_count += 1
            optimizer.zero_grad(set_to_none=True)
            if batch_idx % 50 == 0:
                _p(f"[ep{epoch}] NaN/Inf at step {batch_idx} (total skips={nan_skip_count})")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        train_losses.append(loss.item())
        if batch_idx % 50 == 0:
            _p(f"[ep{epoch}] step {batch_idx}/{len(train_loader)} loss={np.mean(train_losses[-50:]):.4f} scale={scaler.get_scale():.0f}")
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

torch.save(
    {"epoch": CFG.EPOCHS, "val_ns22": history["val_ns22"][-1],
     "state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
    },
    CFG.LAST_CKPT,
)
_p(f"\\nTraining DONE. Best val_ns22={best_val:.4f}, nan_skips={nan_skip_count}")
'''

# Re-enable DataParallel + GradScaler
NEW_MODEL_SETUP = '''model = SEDModel().to(DEVICE)
if torch.cuda.device_count() > 1:
    print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
else:
    print(f"Single GPU mode")

mel_extractor = MelExtractor().to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
# GradScaler with smaller init_scale (default 2^16 too high for fresh model)
# Created inside train_loop with init_scale=2**10
loss_fn = nn.BCEWithLogitsLoss()
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")'''

# Eval with AMP for speed
NEW_EVAL_BODY = '''    for wav, label in tqdm(loader, desc="val", leave=False):
        wav = wav.to(DEVICE, non_blocking=True)
        mel = mel_ex(wav)
        with autocast("cuda", dtype=torch.float16):
            logit, _ = model(mel)
        all_logits.append(logit.float().cpu())
        all_labels.append(label)'''

for NB, batch in NB_BATCH:
    if not NB.exists():
        print(f"[SKIP] {NB.name}")
        continue
    nb = json.load(open(NB, encoding="utf-8"))
    n_changes = 0

    # 1. Replace train_loop
    for c in nb["cells"]:
        if c.get("id") == "train_loop":
            c["source"] = NEW_TRAIN_LOOP.splitlines(keepends=True)
            n_changes += 1
            break

    # 2. Replace model_setup (re-enable DP)
    for c in nb["cells"]:
        if c.get("id") == "model_setup":
            c["source"] = NEW_MODEL_SETUP.splitlines(keepends=True)
            n_changes += 1
            break

    # 3. Update BATCH_SIZE in config
    for c in nb["cells"]:
        if c.get("id") == "config":
            src = "".join(c["source"])
            import re
            # match any BATCH_SIZE = X line (could be with comment)
            new_src = re.sub(
                r"BATCH_SIZE = \d+[^\n]*",
                f"BATCH_SIZE = {batch}  # v7: safe AMP + DP enables larger batch",
                src, count=1
            )
            if new_src != src:
                c["source"] = new_src.splitlines(keepends=True)
                n_changes += 1
            break

    # 4. Fix eval_fn for AMP
    for c in nb["cells"]:
        if c.get("id") == "eval_fn":
            src = "".join(c["source"])
            # The eval has no autocast right now (we removed it in v6). Add it back.
            old = """    for wav, label in tqdm(loader, desc="val", leave=False):
        wav = wav.to(DEVICE, non_blocking=True)
        mel = mel_ex(wav)
        logit, _ = model(mel)
        all_logits.append(logit.float().cpu())
        all_labels.append(label)"""
            new = """    for wav, label in tqdm(loader, desc="val", leave=False):
        wav = wav.to(DEVICE, non_blocking=True)
        mel = mel_ex(wav)
        with autocast("cuda", dtype=torch.float16):
            logit, _ = model(mel)
        all_logits.append(logit.float().cpu())
        all_labels.append(label)"""
            if old in src:
                src = src.replace(old, new)
                n_changes += 1
                # Also add import
                if "from torch.amp import autocast" not in src and "import autocast" not in src:
                    src = "from torch.amp import autocast\n" + src
                    n_changes += 1
                c["source"] = src.splitlines(keepends=True)
            break

    json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{NB.parent.parent.name}] {n_changes} changes (batch={batch}, safe AMP + DP)")
