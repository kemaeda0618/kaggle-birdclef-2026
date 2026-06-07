"""Apply 3 fixes to all 3 NBs (exp049/050/051):
1. NUM_WORKERS = 0 (was 4) — avoid CUDA fork deadlock
2. persistent_workers = False (was True) — avoid persistent worker issue
3. DataParallel disable comment — safer for T4x2
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
        print(f"[SKIP] {NB.name} not found")
        continue
    nb = json.load(open(NB, encoding="utf-8"))
    n_changes = 0

    # Fix 1: NUM_WORKERS in config
    for c in nb["cells"]:
        if c.get("id") == "config":
            src = "".join(c["source"])
            if "NUM_WORKERS = 4" in src:
                src = src.replace("NUM_WORKERS = 4", "NUM_WORKERS = 0  # was 4, avoid CUDA fork deadlock")
                n_changes += 1
            c["source"] = src.splitlines(keepends=True)

    # Fix 2: persistent_workers + drop_last in train_setup
    for c in nb["cells"]:
        if c.get("id") == "train_setup":
            src = "".join(c["source"])
            if "persistent_workers=True" in src:
                src = src.replace("persistent_workers=True", "persistent_workers=False")
                n_changes += 1
            c["source"] = src.splitlines(keepends=True)

    # Fix 3: DataParallel safety check (use only if NUM_WORKERS=0 issue gone)
    # Keep as is for now, just add comment
    for c in nb["cells"]:
        if c.get("id") == "model_setup":
            src = "".join(c["source"])
            old = "if torch.cuda.device_count() > 1:"
            new = "if torch.cuda.device_count() > 1:  # DataParallel OK with NUM_WORKERS=0"
            if old in src and new not in src:
                src = src.replace(old, new)
                n_changes += 1
            c["source"] = src.splitlines(keepends=True)

    json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{NB.name}] {n_changes} changes")

print("\n=== Verify ===")
for NB in NBS:
    nb = json.load(open(NB, encoding="utf-8"))
    for c in nb["cells"]:
        if c.get("id") == "config":
            src = "".join(c["source"])
            for line in src.split("\n"):
                if "NUM_WORKERS" in line:
                    print(f"  {NB.parent.parent.name} [config] {line.strip()}")
        if c.get("id") == "train_setup":
            src = "".join(c["source"])
            for line in src.split("\n"):
                if "persistent_workers" in line:
                    print(f"  {NB.parent.parent.name} [train_setup] {line.strip()[:100]}")
