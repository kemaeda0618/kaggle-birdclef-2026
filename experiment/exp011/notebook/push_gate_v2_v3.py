"""Push gate-fake008 v2 (seed+epochs) and v3 (seed+epochs+blend) to Kaggle."""
import json, os, tempfile, shutil
from pathlib import Path

if not os.environ.get("KAGGLE_API_TOKEN"):
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    with open(kaggle_json) as f:
        creds = json.load(f)
    key = creds.get("key", "")
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

HERE = Path(__file__).parent
USER = "maekeso"

VERSIONS = [
    {
        "nb":   HERE / "nb_gate_fake008_v2.ipynb",
        "slug": "birdclef2026-gate-fake008-v2-seed-epochs",
        "title": "birdclef2026 gate fake008 v2 seed epochs",
        "desc":  "v2: seed=42 fixed + ProtoSSM epochs 40->60 + ResidualSSM 20->30",
    },
    {
        "nb":   HERE / "nb_gate_fake008_v3.ipynb",
        "slug": "birdclef2026-gate-fake008-v3-seed-epochs-blend",
        "title": "birdclef2026 gate fake008 v3 seed epochs blend",
        "desc":  "v3: seed=42 + epochs + SED_W 0.40->0.35 (ProtoSSM weight up)",
    },
]

for v in VERSIONS:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy(v["nb"], td / v["nb"].name)
        meta = {
            "id": f"{USER}/{v['slug']}",
            "title": v["title"],
            "code_file": v["nb"].name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_internet": False,
            "competition_sources": ["birdclef-2026"],
            "dataset_sources": [
                "tuckerarrants/bc2026-distilled-sed-public",
                "jaejohn/perch-meta",
                "rishikeshjani/perch-onnx-for-birdclef-2026",
            ],
            "kernel_sources": ["ashok205/tf-wheels"],
            "model_sources": [
                "google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1",
            ],
        }
        (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        r = api.kernels_push(str(td))
        url = getattr(r, "url", None) or getattr(r, "ref", None)
        ver = getattr(r, "version_number", None)
        err = getattr(r, "error", "") or ""
        print(f"{v['slug']}")
        print(f"  URL: {url}  Version: {ver}")
        print(f"  {v['desc']}")
        if err:
            print(f"  Error: {err}")
        print()
