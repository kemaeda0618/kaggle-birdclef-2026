"""Pull public BC2026 NBs to trace lineage."""
import json, os
from pathlib import Path

os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

TMP = Path(r"C:\Users\maeke\AppData\Local\Temp\nb_trace")
TMP.mkdir(exist_ok=True)

to_pull = [
    "sunderekkiz/birdclef-2026-eos-4",
    "hideyukizushi/bird26-reprod-perch-proto-residualssm-train-s7177",
    "youssefmo942009/lb-0-948",
]

for slug in to_pull:
    name = slug.split("/")[1]
    d = TMP / name
    d.mkdir(exist_ok=True)
    try:
        api.kernels_pull(slug, path=str(d), metadata=True)
        # Show metadata
        meta_file = d / "kernel-metadata.json"
        if meta_file.exists():
            m = json.loads(meta_file.read_text(encoding="utf-8"))
            print(f"\n=== {slug} ===")
            print(f"  title:           {m.get('title')}")
            print(f"  kernel_sources:  {m.get('kernel_sources', [])}")
            print(f"  dataset_sources: {m.get('dataset_sources', [])}")
            print(f"  competition:     {m.get('competition_sources', [])}")
            print(f"  model_sources:   {m.get('model_sources', [])}")
    except Exception as e:
        print(f"FAIL: {slug}: {e}")
