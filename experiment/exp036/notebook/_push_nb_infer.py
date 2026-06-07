"""Push exp036 nb_infer_softauc.ipynb to Kaggle (CPU, internet OFF).

Inputs:
- birdclef-2026 (competition)
- maekeso/birdclef2026-exp036-softauc-weights (Soft AUC fine-tune r2 ckpts; uploaded from Colab train NB)

期待 LB: exp017 R2 base 0.921 + val Δ+0.0020 → standalone 0.922-0.924
"""
import json, os, sys, io, tempfile, shutil
from pathlib import Path

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if not os.environ.get("KAGGLE_API_TOKEN"):
    creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    key = creds.get("key", "")
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB    = Path(__file__).with_name("nb_infer_softauc.ipynb")
USER  = "maekeso"
# Per feedback_kernel_naming_include_model.md: slug に backbone / loss を含める
SLUG  = "birdclef2026-exp036-eca-nfnet-l0-softauc"
TITLE = "birdclef2026 exp036 eca nfnet l0 softauc"  # title = slug-friendly form

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": TITLE,
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        # CPU only for submission (no machine_shape, no enable_gpu)
        "enable_internet": False,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            f"{USER}/birdclef2026-exp036-softauc-weights",
        ],
        "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    url = getattr(r, "url", None) or getattr(r, "ref", None)
    ver = getattr(r, "version_number", None)
    err = getattr(r, "error", "") or ""
    print(f"URL: {url}")
    print(f"Version: {ver}")
    if err:
        print(f"Error: {err}")
