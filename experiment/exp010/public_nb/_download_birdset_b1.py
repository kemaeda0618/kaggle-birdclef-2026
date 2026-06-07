"""Download dacquaviva/birdset-effnet-b1-xcl config.json + README to inspect architecture."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "dacquaviva/birdset-effnet-b1-xcl"
out_dir = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\birdset_effnet_b1_xcl")
out_dir.mkdir(parents=True, exist_ok=True)

# Download config.json and README.md only (skip 76MB model.safetensors for inspection)
for fname in ["config.json", "README.md", "artifact_manifest.json"]:
    try:
        api.dataset_download_file(slug, fname, path=str(out_dir), force=True)
        # Kaggle adds .zip extension for single files sometimes; check both
        for ext in ["", ".zip"]:
            fp = out_dir / (fname + ext)
            if fp.exists():
                print(f"\n=== {fname} ===")
                if fname.endswith(".json"):
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])
                else:
                    print(fp.read_text(encoding="utf-8")[:3000])
                break
    except Exception as e:
        print(f"  FAIL {fname}: {type(e).__name__}: {str(e)[:200]}")
