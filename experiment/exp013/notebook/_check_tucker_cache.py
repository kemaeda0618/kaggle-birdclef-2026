"""Try to download Tucker's audio_cache_meta.csv directly."""
import json, os
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

OUT_DIR = Path("./tucker_meta_test")
OUT_DIR.mkdir(exist_ok=True)

# Try various paths for the meta CSVs
candidates = [
    "waveform_cache/audio_cache_meta.csv",
    "audio_cache_meta.csv",
    "waveform_cache/soundscape_cache_meta.csv",
    "soundscape_cache_meta.csv",
]
for fn in candidates:
    print(f"\nTrying: {fn}")
    try:
        api.dataset_download_file(
            "tuckerarrants/birdclef-2026-waveform-cache",
            file_name=fn,
            path=str(OUT_DIR),
        )
        # Find downloaded file
        local = OUT_DIR / Path(fn).name
        local_zip = OUT_DIR / (Path(fn).name + ".zip")
        for p in [local, local_zip]:
            if p.exists():
                print(f"  ✓ Downloaded: {p} ({p.stat().st_size/1024/1024:.2f} MB)")
                break
        else:
            print(f"  (no local file found after attempt)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

# List what we got
print(f"\n=== {OUT_DIR} contents ===")
for p in sorted(OUT_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUT_DIR)!s:40s} {p.stat().st_size/1024/1024:.2f} MB")
