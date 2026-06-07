"""Push exp146 (33/33/33) + exp147 (50/0/50) — both [1,2]+Tucker5 reweights (submit NB, internet=False)."""
import json, os, io, sys, tempfile, shutil
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"; os.execv(sys.executable, [sys.executable] + sys.argv)
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle"/"kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

USER = "maekeso"
DS = ["jaejohn/perch-meta", "rishikeshjani/perch-onnx-for-birdclef-2026"]
KS = ["ashok205/tf-wheels", "maekeso/birdclef2026-tucker-sed-ov",
      "maekeso/birdclef2026-e106-3fold-ov", "ttahara/birdclef-2026-download-wheels"]
MS = ["google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1"]

CONFIGS = [
    (Path("experiment/exp146/notebook/nb_exp146_fold12_w333.ipynb"),
     "birdclef2026-exp146-fold12-w333", "birdclef2026 exp146 fold12 w333"),
    (Path("experiment/exp147/notebook/nb_exp147_fold12_w50_0_50.ipynb"),
     "birdclef2026-exp147-fold12-w50-0-50", "birdclef2026 exp147 fold12 w50 0 50"),
]
for nb_path, slug, title in CONFIGS:
    assert nb_path.exists(), f"missing {nb_path}"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); shutil.copy(nb_path, td/nb_path.name)
        meta = {
            "id": f"{USER}/{slug}", "title": title, "code_file": nb_path.name,
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_internet": False, "competition_sources": ["birdclef-2026"],
            "dataset_sources": DS, "kernel_sources": KS, "model_sources": MS,
        }
        (td/"kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        r = api.kernels_push(str(td)); print(f"{slug}: URL={r.url} V={r.version_number}")
