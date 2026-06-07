"""Audit inference NB for correctness vs training NB."""
import json
from pathlib import Path

# Load both NBs
TRAIN_NB = json.load(open(Path(__file__).with_name("nb_train_ali.ipynb"), encoding="utf-8"))
INFER_NB = json.load(open(Path(__file__).with_name("nb_infer_r1_ali.ipynb"), encoding="utf-8"))

print(f"Training NB: {len(TRAIN_NB['cells'])} cells")
print(f"Inference NB: {len(INFER_NB['cells'])} cells")
print()

# Extract relevant config from train
train_src = "\n".join("".join(c.get("source", [])) for c in TRAIN_NB["cells"])
infer_src = "\n".join("".join(c.get("source", [])) for c in INFER_NB["cells"])

# === Key parameters to check ===
print("=" * 60)
print("CONFIG MATCH CHECK (training vs inference)")
print("=" * 60)

checks = [
    ("BACKBONE eca_nfnet_l0", 'BACKBONE = "eca_nfnet_l0"', 'BACKBONE = "eca_nfnet_l0"'),
    ("N_FFT = 2048",          "N_FFT          = 2048",     "N_FFT"),
    ("HOP_LENGTH = 512",      "HOP_LENGTH     = 512",      "HOP_LENGTH"),
    ("N_MELS = 256",          "N_MELS         = 256",      "N_MELS"),
    ("FMIN = 20",             "FMIN           = 20",       "FMIN"),
    ("FMAX = 16000",          "FMAX           = 16000",    "FMAX"),
    ("SR = 32000",            "SR = 32000",                "SR = 32000"),
    ("TRAIN_DURATION = 5",    "TRAIN_DURATION = 5",        "5"),
    ("hidden_dim = 512",      "hidden_dim=512",            "hidden_dim"),
    ("drop_path_rate = 0.1",  "drop_path_rate=0.1",        "drop_path"),
    ("PERCH_EMBED_DIM = 1536","PERCH_EMBED_DIM = 1536",    "PERCH_EMBED_DIM"),
]

for name, train_pat, infer_pat in checks:
    in_train = train_pat in train_src
    in_infer = False
    # Check if any infer pattern matches
    for line in infer_src.splitlines():
        if infer_pat in line:
            # Extract the value
            in_infer = True
            print(f"  [{ 'OK' if in_train else '?'}] train: {name}")
            print(f"  [INFO] infer line: {line.strip()[:120]}")
            break
    if not in_infer:
        print(f"  [?] infer: {name} pattern '{infer_pat}' not found")
    print()

# === Mel spec computation check ===
print("=" * 60)
print("MEL SPECTROGRAM COMPUTATION")
print("=" * 60)
import re

# Find MelSpectrogram instantiation in train
for nb_name, src in [("TRAIN", train_src), ("INFER", infer_src)]:
    mel_match = re.search(r"MelSpectrogram\(([^)]+)\)", src, re.DOTALL)
    if mel_match:
        print(f"\n{nb_name} MelSpectrogram args:")
        # Clean and print
        args = mel_match.group(1)
        for ln in args.split(","):
            ln = ln.strip().replace("\n", " ").replace("\\n", "")
            if ln:
                print(f"  {ln}")

# === Normalization check ===
print("\n" + "=" * 60)
print("NORMALIZATION (mel preprocessing)")
print("=" * 60)
for nb_name, src in [("TRAIN", train_src), ("INFER", infer_src)]:
    print(f"\n{nb_name}:")
    # Find mean/std normalization
    for line in src.splitlines():
        if "mel.mean()" in line or "mel.std()" in line or "AmplitudeToDB" in line or "top_db" in line:
            print(f"  {line.strip()[:120]}")

# === Model architecture: BirdSEDModel ===
print("\n" + "=" * 60)
print("MODEL CLASS DEFINITION (BirdSEDModel)")
print("=" * 60)
for nb_name, src in [("TRAIN", train_src), ("INFER", infer_src)]:
    m = re.search(r"class BirdSEDModel.*?def forward", src, re.DOTALL)
    if m:
        body = m.group(0)[:1500]
        print(f"\n{nb_name} BirdSEDModel init (first 1500 chars):")
        print(body)
        print("...")

# === Inference window logic ===
print("\n" + "=" * 60)
print("INFERENCE WINDOW / CHUNK LOGIC")
print("=" * 60)
for line in infer_src.splitlines():
    if any(k in line for k in ["N_WINDOWS", "window", "chunk", "5s", "30 -", "30 sec", "*5*", "* 5 *"]):
        s = line.strip()
        if s and not s.startswith("#"):
            print(f"  {s[:150]}")

# === ckpt loading ===
print("\n" + "=" * 60)
print("CKPT LOADING (state_dict)")
print("=" * 60)
for line in infer_src.splitlines():
    if "state" in line.lower() and ("load" in line.lower() or "torch.load" in line or "state_dict" in line):
        s = line.strip()
        if s and not s.startswith("#"):
            print(f"  {s[:200]}")
