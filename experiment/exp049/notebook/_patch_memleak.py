"""Memory leak fix patches for all 3 NBs.

1. Remove librosa.resample (BC2026 is 32kHz, no resample needed)
2. Disable DataParallel (use single GPU only)
3. torch.cuda.empty_cache() every N steps (not just per epoch)
4. del intermediate vars explicitly
"""
import json
from pathlib import Path

NBS = [
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp049\notebook\nb_train_effv2s_r1.ipynb"),
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp050\notebook\nb_train_convnext_r1.ipynb"),
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp051\notebook\nb_train_swin_r1.ipynb"),
]

for NB in NBS:
    if not NB.exists():
        print(f"[SKIP] {NB.name}")
        continue
    nb = json.load(open(NB, encoding="utf-8"))
    n_changes = 0

    # Fix 1: Disable DataParallel
    for c in nb["cells"]:
        if c.get("id") == "model_setup":
            src = "".join(c["source"])
            old1 = "if torch.cuda.device_count() > 1:  # DataParallel OK with NUM_WORKERS=0"
            if old1 in src:
                src = src.replace(
                    old1 + "\n    print(f\"Using DataParallel on {torch.cuda.device_count()} GPUs\")\n    model = nn.DataParallel(model)",
                    "# DataParallel disabled (memory leak suspected)\nif False:\n    print(f\"Using DataParallel on {torch.cuda.device_count()} GPUs\")\n    model = nn.DataParallel(model)\nelse:\n    print(f\"Single GPU mode (DataParallel disabled for memory safety)\")"
                )
                n_changes += 1
            c["source"] = src.splitlines(keepends=True)
            break

    # Fix 2: Remove librosa.resample (assume 32kHz)
    for c in nb["cells"]:
        if c.get("id") == "dataset":
            src = "".join(c["source"])
            old = "            if sr != self.sr:\n                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)\n"
            if old in src:
                src = src.replace(old, "            # No resample (BC2026 is 32kHz)\n            if sr != self.sr:\n                wav = wav[::int(round(sr/self.sr))] if sr > self.sr else np.repeat(wav, int(round(self.sr/sr)))\n")
                n_changes += 1
            c["source"] = src.splitlines(keepends=True)
            break

    # Fix 3: empty_cache every 50 steps + explicit del in train loop
    for c in nb["cells"]:
        if c.get("id") == "train_loop":
            src = "".join(c["source"])
            old = """        if batch_idx % 50 == 0:
            _p(f"[ep{epoch}] step {batch_idx}/{len(train_loader)} loss={np.mean(train_losses[-50:]):.4f}")"""
            new = """        if batch_idx % 50 == 0:
            _p(f"[ep{epoch}] step {batch_idx}/{len(train_loader)} loss={np.mean(train_losses[-50:]):.4f}")
            torch.cuda.empty_cache()  # mem leak guard
        del mel, logit, loss
        if 'label_mix' in dir(): del label_mix"""
            if old in src:
                src = src.replace(old, new)
                n_changes += 1
            c["source"] = src.splitlines(keepends=True)
            break

    json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{NB.parent.parent.name}/{NB.name}] {n_changes} changes")

print("\n=== Verify ===")
for NB in NBS:
    nb = json.load(open(NB, encoding="utf-8"))
    for c in nb["cells"]:
        if c.get("id") == "model_setup":
            src = "".join(c["source"])
            if "DataParallel disabled" in src:
                print(f"  {NB.parent.parent.name} [model_setup] OK DataParallel disabled")
        if c.get("id") == "train_loop":
            src = "".join(c["source"])
            if "del mel, logit, loss" in src:
                print(f"  {NB.parent.parent.name} [train_loop] OK del intermediates")
