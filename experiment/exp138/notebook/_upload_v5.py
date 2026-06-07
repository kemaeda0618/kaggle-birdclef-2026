"""Upload amphib_b0_v5 (pth + meta) as a Kaggle dataset."""
import json, os, io, sys, shutil
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"; os.execv(sys.executable, [sys.executable] + sys.argv)
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle"/"kaggle.json").read_text())["key"]
BASE = Path(__file__).parent; OUT = BASE/"_out"; DS = BASE/"_v5_dataset"; DS.mkdir(exist_ok=True)
for f in ["amphib_b0_v5.pth", "amphib_v5_meta.json"]:
    shutil.copy(OUT/f, DS/f)
USER="maekeso"; SLUG="birdclef2026-amphib-b0-v5"
(DS/"dataset-metadata.json").write_text(json.dumps({"title":"birdclef2026 amphib b0 v5","id":f"{USER}/{SLUG}","licenses":[{"name":"CC0-1.0"}]},indent=2))
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
try:
    api.dataset_create_new(folder=str(DS), public=False, dir_mode="zip", quiet=False); print("created", SLUG)
except Exception as e:
    print("create err, version:", str(e)[:120])
    api.dataset_create_version(folder=str(DS), version_notes="v5", dir_mode="zip", quiet=False); print("versioned", SLUG)
