"""Comprehensive train_r2 vs infer NB cross-check."""
import json
import re

train = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train_r2.ipynb", encoding="utf-8"))
infer = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_infer.ipynb", encoding="utf-8"))

def get_cell(nb, cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            return "".join(c["source"])
    return ""

# Extract config values from both
KEYS = ["BACKBONE", "USE_PERCH_DISTILL", "N_FFT", "HOP_LENGTH", "N_MELS",
        "FMIN", "FMAX", "SR", "TRAIN_DURATION", "TRAIN_SAMPLES", "NUM_CLASSES",
        "PERCH_EMBED_DIM"]

def grep_val(src, key):
    pattern = rf"^{key}\s*=\s*(.+?)(?:\s*#|$)"
    for line in src.split("\n"):
        m = re.match(pattern, line.strip())
        if m:
            return m.group(1).strip()
    return None

train_cfg = get_cell(train, "config")
infer_cfg = get_cell(infer, "config")

print("=" * 75)
print("CONFIG comparison (train_r2 vs infer)")
print("=" * 75)
print(f"{'param':25s} | {'train_r2':25s} | {'infer':25s} | match")
print("-" * 80)
for k in KEYS:
    tv = grep_val(train_cfg, k) or "?"
    iv = grep_val(infer_cfg, k) or "?"
    mark = "✓" if tv == iv else "✗"
    print(f"{k:25s} | {tv:25s} | {iv:25s} | {mark}")

# Model architecture check
print("\n" + "=" * 75)
print("MODEL architecture check (BirdSEDModel definitions)")
print("=" * 75)

def extract_model_def(src):
    """Extract class BirdSEDModel(...) ... up to next class or top-level def."""
    idx = src.find("class BirdSEDModel")
    if idx < 0:
        return None
    # find end by looking for next class/def at column 0
    rest = src[idx:]
    lines = rest.split("\n")
    end = 0
    for i, ln in enumerate(lines):
        if i > 0 and (ln.startswith("class ") or ln.startswith("def ")):
            end = i
            break
    if end == 0:
        end = len(lines)
    return "\n".join(lines[:end])

train_model = extract_model_def(get_cell(train, "model"))
infer_model = extract_model_def(get_cell(infer, "model"))

# diff
import difflib
diff = list(difflib.unified_diff(
    train_model.splitlines() if train_model else [],
    infer_model.splitlines() if infer_model else [],
    fromfile="train_r2/model", tofile="infer/model", n=2, lineterm=""
))
if not diff:
    print("  IDENTICAL ✓")
else:
    print(f"  DIFFERS ({len(diff)} diff lines):")
    for line in diff[:60]:
        print(line)

# load_state_dict check
print("\n" + "=" * 75)
print("Architecture submodules required by ckpt")
print("=" * 75)
for cls in ["GeMFreqPool", "DistillHead", "MelSpecTransform"]:
    in_train = cls in get_cell(train, "model")
    in_infer = cls in get_cell(infer, "model")
    print(f"  {cls:25s} train_r2={in_train} | infer={in_infer}")

# Check audio decode + chunking
print("\n" + "=" * 75)
print("Inference loop sanity")
print("=" * 75)
infer_loop = get_cell(infer, "infer")
print(f"  N_WINDOWS: {'N_WINDOWS = 12' in infer_loop}")
print(f"  CHUNK_N = SR*TRAIN_DURATION (5s): {'CHUNK_N = SR * TRAIN_DURATION' in infer_loop}")
print(f"  mel normalize per-chunk: {'mel[i].mean()' in infer_loop}")
print(f"  Gaussian smooth: {'GAUSSIAN_KERNEL' in infer_loop}")
print(f"  Sigmoid: {'sigmoid_np' in infer_loop}")
print(f"  60s pad: {'target_len = 60 * SR' in infer_loop}")
print(f"  blend logit space (0.5*clip+0.5*frame): {'0.5 * clip_logits + 0.5 * frame_max' in infer_loop}")
