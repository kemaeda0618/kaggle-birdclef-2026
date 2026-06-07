"""Apply safety patches to Stage 1 NB:
1. LR 1e-3 → 5e-4 (safer convergence)
2. AMP fp16 → bf16 (Blackwell native, stable)
3. NUM_WORKERS 8 → 4 (Colab CPU safety)
4. MIXUP_ALPHA 0.5 → 1.0 (Beta uniform, less aggressive)
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp052\notebook\nb_stage1_xc_pretrain.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

n_changes = 0

# 1+3+4. config cell: LR, NUM_WORKERS, MIXUP_ALPHA
for c in nb["cells"]:
    if c.get("id") == "config":
        src = "".join(c["source"])
        replacements = [
            ("LR = 1e-3", "LR = 5e-4  # was 1e-3, safer (避 initial NaN)"),
            ("NUM_WORKERS = 8", "NUM_WORKERS = 4  # was 8, Colab CPU safety"),
            ("MIXUP_ALPHA = 0.5", "MIXUP_ALPHA = 1.0  # was 0.5, Beta(1,1) uniform (less aggressive)"),
        ]
        for old, new in replacements:
            if old in src:
                src = src.replace(old, new)
                n_changes += 1
        c["source"] = src.splitlines(keepends=True)
        break

# 2. train_loop cell: fp16 → bf16
for c in nb["cells"]:
    if c.get("id") == "train_loop":
        src = "".join(c["source"])
        if 'autocast("cuda", dtype=torch.float16)' in src:
            src = src.replace(
                'autocast("cuda", dtype=torch.float16)',
                'autocast("cuda", dtype=torch.bfloat16)'
            )
            n_changes += src.count("torch.bfloat16")  # count replacements
        c["source"] = src.splitlines(keepends=True)
        break

# Same for eval_fn
for c in nb["cells"]:
    if c.get("id") == "eval_fn":
        src = "".join(c["source"])
        if 'autocast("cuda", dtype=torch.float16)' in src:
            src = src.replace(
                'autocast("cuda", dtype=torch.float16)',
                'autocast("cuda", dtype=torch.bfloat16)'
            )
            n_changes += 1
        c["source"] = src.splitlines(keepends=True)
        break

# Also: with bf16, GradScaler is NOT needed (bf16 has same exponent range as fp32)
# Update model_setup + train_loop to remove scaler usage
for c in nb["cells"]:
    if c.get("id") == "model_setup":
        src = "".join(c["source"])
        if 'scaler = GradScaler("cuda", init_scale=2**10)' in src:
            src = src.replace(
                'scaler = GradScaler("cuda", init_scale=2**10)',
                '# scaler = GradScaler("cuda", init_scale=2**10)  # not needed with bf16'
            )
            n_changes += 1
        c["source"] = src.splitlines(keepends=True)
        break

# train_loop: replace scaler usage with direct backward + step
for c in nb["cells"]:
    if c.get("id") == "train_loop":
        src = "".join(c["source"])
        # Smoke test section
        old_smoke = """if not (torch.isnan(test_loss) or torch.isinf(test_loss)):
    scaler.scale(test_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    _p(f"[smoke] OK")"""
        new_smoke = """if not (torch.isnan(test_loss) or torch.isinf(test_loss)):
    test_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _p(f"[smoke] OK")"""
        if old_smoke in src:
            src = src.replace(old_smoke, new_smoke)
            n_changes += 1

        # Main training loop
        old_main = """        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)"""
        new_main = """        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)"""
        if old_main in src:
            src = src.replace(old_main, new_main)
            n_changes += 1

        c["source"] = src.splitlines(keepends=True)
        break

# Save
json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Total changes: {n_changes}")
print(f"\n=== Verify ===")
for c in nb["cells"]:
    if c.get("id") == "config":
        src = "".join(c["source"])
        for line in src.split("\n"):
            ls = line.strip()
            if any(k in ls for k in ["LR = ", "NUM_WORKERS = ", "MIXUP_ALPHA = "]):
                print(f"  [config] {ls[:100]}")
    if c.get("id") == "train_loop":
        src = "".join(c["source"])
        if "bfloat16" in src:
            print(f"  [train_loop] OK bf16 applied")
        if "scaler.scale(" not in src:
            print(f"  [train_loop] OK scaler removed (using bf16 direct)")
