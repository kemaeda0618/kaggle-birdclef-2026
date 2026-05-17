"""Try to fetch LB / version info for kernels of interest."""
import json, os, tempfile
from pathlib import Path

os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

REFS = [
    "ahmadzulfiqar001/birdclef-2026-birdnet-baseline",
    "mattiaangeli/birdclef-2026-0-943-better-blend",
    "konbu17/bird26-wliilamsam-0943-with-train-audio-head",
    "mtoshidesu/birdclef-2026-0-941-public-lb-onnx-perch-pr",
    "kamongi/pantanal-distill-birdclef2026",
]

for ref in REFS:
    print(f"=== {ref} ===")
    try:
        with tempfile.TemporaryDirectory() as td:
            api.kernels_pull(ref, path=td, metadata=True)
            md = Path(td) / "kernel-metadata.json"
            if md.exists():
                m = json.loads(md.read_text())
                # Print key fields
                for k in ("title", "id", "is_private", "language", "kernel_type",
                          "competition_sources", "dataset_sources",
                          "kernel_sources", "model_sources"):
                    if k in m:
                        print(f"  {k}: {m[k]}")
    except Exception as e:
        print(f"  pull err: {type(e).__name__}: {str(e)[:120]}")
    print()
