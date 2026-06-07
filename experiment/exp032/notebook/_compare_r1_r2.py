"""Compare R1 (nb_train.ipynb) vs R2 regen-r1-pseudo cell parameters."""
import json
import re

r1 = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train.ipynb", encoding="utf-8"))
r2 = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train_r2.ipynb", encoding="utf-8"))

def get_cell(nb, cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            return "".join(c["source"])
    return None

def get_cell_idx(nb, idx):
    return "".join(nb["cells"][idx].get("source", []))

print("=== R1 nb_train.ipynb cell ids ===")
for i, c in enumerate(r1["cells"]):
    src = "".join(c.get("source", []))
    head = src[:60].replace("\n", " | ")
    print(f"  [{i:02d}] id={c.get('id','?'):20s} | {head}")

print("\n=== R2 regen-r1-pseudo key constants used ===")
regen = get_cell(r2, "regen-r1-pseudo")
# Extract referenced constants
for tok in ["SR", "N_FFT", "HOP_LENGTH", "N_MELS", "FMIN", "FMAX", "BACKBONE",
            "TRAIN_SAMPLES", "TRAIN_DURATION", "NUM_CLASSES", "POWER_GAMMA"]:
    if tok in regen:
        # find first usage
        pass

# Pull constants from R2 config
r2_cfg = get_cell(r2, "config")
print("\n=== R2 config constants ===")
for line in r2_cfg.split("\n"):
    if re.match(r"^[A-Z_]+\s*=", line):
        print(" ", line.strip())

# Now find equivalent in R1 (look for config cell)
r1_cfg = get_cell(r1, "config")
if r1_cfg:
    print("\n=== R1 config constants ===")
    for line in r1_cfg.split("\n"):
        if re.match(r"^[A-Z_]+\s*=", line):
            print(" ", line.strip())
else:
    # find by content
    for i, c in enumerate(r1["cells"]):
        src = "".join(c.get("source", []))
        if "N_FFT" in src and "BACKBONE" in src:
            print(f"\n=== R1 cell [{i}] (id={c.get('id','?')}) constants ===")
            for line in src.split("\n"):
                if re.match(r"^[A-Z_]+\s*=", line):
                    print(" ", line.strip())
            break
