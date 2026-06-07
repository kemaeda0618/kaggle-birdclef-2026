"""Push exp037 NB4 + ResidualSSM (Level A mtoshidesu 路線) to Kaggle."""
import json, os, sys, io, tempfile, shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Windows cp932 回避: PYTHONUTF8=1 が未設定なら env に追加して re-exec
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

# KGAT_ トークンを ~/.kaggle/kaggle.json から自動取得
if not os.environ.get("KAGGLE_API_TOKEN"):
    _kgat = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    os.environ["KAGGLE_API_TOKEN"] = _kgat

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB = Path(__file__).with_name("nb_protossm_resssm.ipynb")
USER = "maekeso"
# Per feedback_kernel_naming_include_model.md: slug は model family (protossm-resssm) を含める
SLUG = "birdclef2026-exp037-nb4-protossm-resssm"
TITLE = "birdclef2026 exp037 nb4 protossm resssm"  # title = slug (slug-friendly form)

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
        "enable_internet": False,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            "rishikeshjani/perch-onnx-for-birdclef-2026",
            "maekeso/birdclef2026-perch-embed-anura",
            "maekeso/birdclef2026-perch-embed-inat-nonbird",
        ],
        "kernel_sources": ["maekeso/birdclef2026-exp010-nb1-embedding"],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", "(no url)"), "Version:", getattr(r, "version_number", "(no ver)"))
    err = getattr(r, "error", "") or ""
    if err:
        print(f"Error: {err}")
