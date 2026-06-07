"""Replace train_loop cell with verbose-print version to find exact hang point.
Also: do a smoke test BEFORE the main loop to isolate first-batch issue.
"""
import json
from pathlib import Path

NBS = [
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp049\notebook\nb_train_effv2s_r1.ipynb"),
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp050\notebook\nb_train_convnext_r1.ipynb"),
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp051\notebook\nb_train_swin_r1.ipynb"),
]

VERBOSE_TRAIN_LOOP = '''# ============================================================
# Cell 10: Training loop (VERBOSE for debug, prints every step)
# ============================================================
import sys
def _p(msg):
    \"\"\"flush-safe print\"\"\"
    print(msg, flush=True)
    sys.stdout.flush()

# Smoke test: fetch ONE batch first to isolate hang
_p(f"[smoke] creating train_loader iter...")
import time
_st = time.time()
smoke_iter = iter(train_loader)
_p(f"[smoke] iter created in {time.time()-_st:.1f}s, fetching batch...")
_st = time.time()
test_wav, test_label = next(smoke_iter)
_p(f"[smoke] batch fetched in {time.time()-_st:.1f}s: wav={test_wav.shape} label_sum={test_label.sum().item():.0f}")

_p(f"[smoke] move to GPU...")
_st = time.time()
test_wav_g = test_wav.to(DEVICE)
torch.cuda.synchronize()
_p(f"[smoke] moved in {time.time()-_st:.1f}s")

_p(f"[smoke] mel transform...")
_st = time.time()
with autocast():
    test_mel = mel_extractor(test_wav_g)
torch.cuda.synchronize()
_p(f"[smoke] mel in {time.time()-_st:.1f}s, shape={test_mel.shape}")

_p(f"[smoke] forward...")
_st = time.time()
with autocast():
    test_logit, _ = model(test_mel)
torch.cuda.synchronize()
_p(f"[smoke] forward in {time.time()-_st:.1f}s, logit={test_logit.shape}")

_p(f"[smoke] backward...")
_st = time.time()
test_loss = loss_fn(test_logit, test_label.to(DEVICE))
scaler.scale(test_loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
torch.cuda.synchronize()
_p(f"[smoke] backward+step in {time.time()-_st:.1f}s, loss={test_loss.item():.4f}")
del smoke_iter, test_wav, test_label, test_wav_g, test_mel, test_logit, test_loss
torch.cuda.empty_cache()

_p(f"\\n=== SMOKE TEST PASSED, starting main training ===\\n")

# ============================================================
# Main training loop
# ============================================================
history = {"train_loss": [], "val_ns22": [], "lr": [], "elapsed_min": []}
best_val = 0.0
start_t = time.time()

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
        with autocast():
            mel = mel_extractor(wav)
            mel, label_mix = spec_mixup(mel, label)
            mel = spec_augment(mel)
            logit, _ = model(mel)
            loss = loss_fn(logit, label_mix)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        train_losses.append(loss.item())
        if batch_idx % 50 == 0:
            _p(f"[ep{epoch}] step {batch_idx}/{len(train_loader)} loss={np.mean(train_losses[-50:]):.4f}")
    scheduler.step()
    torch.cuda.empty_cache()

    train_loss = float(np.mean(train_losses))
    _p(f"[ep{epoch}] train done, evaluating val...")
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
       f" {'BEST' if is_best else ''} lr={lr_now:.2e} ({ep_min:.1f}min, total {elapsed_min:.1f}min) ===")

# Save final
torch.save(
    {"epoch": CFG.EPOCHS, "val_ns22": history["val_ns22"][-1],
     "state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
    },
    CFG.LAST_CKPT,
)
_p(f"\\nTraining DONE. Best val_ns22={best_val:.4f}")
'''

for NB in NBS:
    if not NB.exists():
        print(f"[SKIP] {NB.name}")
        continue
    nb = json.load(open(NB, encoding="utf-8"))
    replaced = False
    for c in nb["cells"]:
        if c.get("id") == "train_loop":
            c["source"] = VERBOSE_TRAIN_LOOP.splitlines(keepends=True)
            replaced = True
            break
    if replaced:
        json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[{NB.parent.parent.name}] train_loop replaced with verbose version")
    else:
        print(f"[{NB.parent.parent.name}] train_loop cell not found")
