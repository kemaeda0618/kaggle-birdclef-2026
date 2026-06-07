"""Push exp028 nb_pseudo_l0r2.ipynb to Kaggle (CPU, internet OFF).

l0 R2 5-fold ensemble の train_soundscapes 予測を生成、pseudo_l0r2.csv として保存。
Multi-teacher R3 pseudo の teacher 1。

Note: 修正 push (slug は `pseudo-eca-nfnet-l0-r2` 新 slug、CPU 実行)。
旧 slug の Kaggle NB 上のコードが誤って e17 のものになっていた bug 修正。

Inputs:
- birdclef-2026 (competition、train_soundscapes ~10,658 files)
- maekeso/birdclef2026-exp020-weights-5fold (R2 ckpts: r2_fold0..4_ckpt_best_ns22.pth)
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

NB    = Path(__file__).with_name("nb_pseudo_l0r2.ipynb")
USER  = "maekeso"
SLUG  = "birdclef2026-exp028-pseudo-eca-nfnet-l0-r2"
TITLE = "birdclef2026 exp028 pseudo eca nfnet l0 r2"

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
        # ★ Kaggle T4 GPU (NvidiaTeslaT4 = T4x2)、5-fold ensemble inference ~2.5h
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": False,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            f"{USER}/birdclef2026-exp020-weights-5fold",   # R2 5-fold ckpts
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
