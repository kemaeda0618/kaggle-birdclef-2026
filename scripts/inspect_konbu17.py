"""Inspect konbu17 train_audio_head dataset & NB."""
import json, os, tempfile
from pathlib import Path

os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Dataset files
print("=== Dataset: konbu17/bird26-train-audio-head-v1 ===")
try:
    r = api.dataset_list_files("konbu17/bird26-train-audio-head-v1")
    for f in r.files:
        print(f"  {f}")
except Exception as e:
    print(f"  err: {e}")

print()
print("=== NB: konbu17/bird26-wliilamsam-0943-with-train-audio-head ===")
out = Path("experiment/exp010/notebook/_reference_konbu17")
out.mkdir(parents=True, exist_ok=True)
try:
    api.kernels_pull("konbu17/bird26-wliilamsam-0943-with-train-audio-head",
                     path=str(out), metadata=False)
    for p in out.iterdir():
        print(f"  {p.name} {p.stat().st_size}")
except Exception as e:
    print(f"  err: {e}")
