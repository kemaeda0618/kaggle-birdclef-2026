"""Push exp011 Phase 4 training notebook to Kaggle (T4x2).

Phase 4: ogg 直読 + GPU mel + raw waveform mixup (mel cache 不要 -> kernel_sources 空)
Slug: birdclef2026-exp011-train-phase4-sed (title 由来)
"""
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

NB = Path(__file__).with_name("nb1_train_p4.ipynb")
USER = "maekeso"
# title を slug 形式に揃える (feedback_kaggle_slug_from_title.md)
SLUG = "birdclef2026-exp011-train-phase4-sed"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": "BirdCLEF2026 exp011 train phase4 sed",
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",   # T4x2 (enable_gpu は書かない)
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [],   # mel cache 不要 (ogg 直読)
        "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None) or getattr(r, "ref", None))
    print("Version:", getattr(r, "version_number", None))
    err = getattr(r, "error", "") or ""
    if err:
        print("Error:", err)
