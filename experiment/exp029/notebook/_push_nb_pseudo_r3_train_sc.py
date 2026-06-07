"""Push exp029 pseudo NB to Kaggle (GPU T4).

Clean version of pseudo gen NB:
- Properly organized under experiment/exp029/notebook/
- Accurate header/comments (was wrongly attributed to exp020 5-fold in exp028/)
- Same logic as the previous Kaggle kernel that already produced pseudo_exp029.csv

This NB will produce pseudo_exp029.csv on train_soundscapes using exp029 R3 single fold.

Inputs:
- birdclef-2026 (competition, train_soundscapes ~10,658 files)
- maekeso/birdclef2026-exp029-l1-single (r3_fold0_ckpt_best_ns22.pth)

Output:
- /kaggle/working/pseudo_exp029.csv (~410MB)
- /kaggle/working/pseudo_exp029.npy (~60MB, float16)
- /kaggle/working/pseudo_exp029_meta.npy

Note: An older version under exp028/notebook/nb_pseudo_exp029.ipynb already
ran successfully on Kaggle (slug: birdclef2026-exp028-pseudo-eca-nfnet-l1-exp029)
and produced the pseudo CSV. This is a cleaned-up rewrite for clarity.
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

NB    = Path(__file__).with_name("nb_pseudo_r3_train_sc.ipynb")
USER  = "maekeso"
SLUG  = "birdclef2026-exp029-pseudo-r3-train-sc"
TITLE = "birdclef2026 exp029 pseudo r3 train sc"

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
        "machine_shape": "NvidiaTeslaT4",   # T4 GPU
        "enable_internet": False,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            f"{USER}/birdclef2026-exp029-l1-single",   # exp029 R3 fold0 ckpt
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
