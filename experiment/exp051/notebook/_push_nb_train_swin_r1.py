"""Push exp051 Swin-Tiny R1 train NB."""
import json, os, sys, io, tempfile, shutil
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"; os.execv(sys.executable, [sys.executable] + sys.argv)
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB = Path(__file__).with_name("nb_train_swin_r1.ipynb")
with tempfile.TemporaryDirectory() as td:
    td = Path(td); shutil.copy(NB, td / NB.name)
    meta = {
        "id": "maekeso/birdclef2026-exp051-transformer-r1",
        "title": "birdclef2026 exp051 transformer r1",
        "code_file": NB.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_internet": True,
        # machine_shape omitted: 2 GPU session 上限到達のため
        # UI で run 時に T4x2 を選択すること
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [], "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print(f"URL: {getattr(r, 'url', None)}")
    print(f"Version: {getattr(r, 'version_number', None)}")
    print(f"Error: {getattr(r, 'error', None)}")
    print(f"Ref: {getattr(r, 'ref', None)}")
    # Dump full response
    print(f"\nFull response attrs:")
    for attr in dir(r):
        if not attr.startswith("_"):
            try:
                val = getattr(r, attr)
                if not callable(val):
                    print(f"  {attr}: {val}")
            except: pass
