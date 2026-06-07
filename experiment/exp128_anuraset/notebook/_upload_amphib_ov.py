"""Regenerate amphib_meta.json (AMP label order) + upload amphib_b0 OV as a Kaggle dataset."""
import json, os, io, sys, shutil
from pathlib import Path
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"; os.execv(sys.executable, [sys.executable] + sys.argv)
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle"/"kaggle.json").read_text())["key"]

BASE = Path(__file__).parent
OUT = BASE / "_out"
DS = BASE / "_ov_dataset"; DS.mkdir(exist_ok=True)

# Reconstruct AMP label order EXACTLY as the training NB (sorted amphibian primary_labels)
tax = pd.read_csv(r"C:\Users\maeke\work\kaggle\birdclef-2026\taxonomy.csv")
AMP = sorted(tax[tax["class_name"] == "Amphibia"]["primary_label"].astype(str).tolist())
assert len(AMP) == 35, len(AMP)

SR=32000; N_FFT=2048; HOP=512; N_MELS=256; FMIN=20; FMAX=16000; TOP_DB=80; WIN=SR*5
T = 1 + WIN // HOP   # torchaudio center=True -> 313
meta = {"labels": AMP, "n_mels": N_MELS, "T": int(T), "sr": SR, "hop": HOP, "n_fft": N_FFT,
        "fmin": FMIN, "fmax": FMAX, "top_db": TOP_DB, "win": WIN, "backbone": "efficientnet_b0"}
(DS / "amphib_meta.json").write_text(json.dumps(meta, indent=2))
print("AMP order:", AMP)
print("T(frames):", T)

# copy model files
for f in ["amphib_b0.xml", "amphib_b0.bin", "amphib_b0.pth"]:
    shutil.copy(OUT / f, DS / f)
print("files:", [p.name for p in sorted(DS.glob("*"))])

USER = "maekeso"; SLUG = "birdclef2026-amphib-b0-ov"
(DS / "dataset-metadata.json").write_text(json.dumps({
    "title": "birdclef2026 amphib b0 ov",
    "id": f"{USER}/{SLUG}",
    "licenses": [{"name": "CC0-1.0"}],
}, indent=2))

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
try:
    api.dataset_create_new(folder=str(DS), public=False, dir_mode="zip", quiet=False)
    print("created:", f"https://www.kaggle.com/datasets/{USER}/{SLUG}")
except Exception as e:
    print("create failed (maybe exists), try version:", str(e)[:150])
    try:
        api.dataset_create_version(folder=str(DS), version_notes="amphib b0 ov", dir_mode="zip", quiet=False)
        print("versioned:", f"https://www.kaggle.com/datasets/{USER}/{SLUG}")
    except Exception as e2:
        print("version failed:", str(e2)[:200])
