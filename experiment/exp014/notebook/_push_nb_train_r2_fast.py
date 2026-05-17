"""Push exp014 R2-fast NB to Kaggle (T4x2 GPU, internet ON, self-resume).

Slug rule (title slugifies to slug):
  title = "birdclef2026 exp014 train r2 fast"
  slug  = "birdclef2026-exp014-train-r2-fast"
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

NB    = Path(__file__).with_name("nb_train_r2_fast.ipynb")
USER  = "maekeso"
SLUG  = "birdclef2026-exp014-train-r2-fast"
TITLE = "birdclef2026 exp014 train r2 fast"


def push(dataset_sources, kernel_sources=None):
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
            "machine_shape": "NvidiaTeslaT4",
            "enable_internet": True,
            "competition_sources": ["birdclef-2026"],
            "dataset_sources": dataset_sources,
            "kernel_sources": kernel_sources or [],
        }
        (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return api.kernels_push(str(td))


PERCH_CACHE = f"{USER}/birdclef2026-perch-emb-cache"
R1_PSEUDO   = f"{USER}/birdclef2026-exp014-pseudo-r1"
R2_STATE    = f"{USER}/exp014-state-r2-fast"
SELF_KERNEL = f"{USER}/{SLUG}"

ATTEMPTS = [
    ([PERCH_CACHE, R1_PSEUDO, R2_STATE], [SELF_KERNEL]),
    ([PERCH_CACHE, R1_PSEUDO], [SELF_KERNEL]),
]

last_err = None
pushed = False
for ds_sources, ks_sources in ATTEMPTS:
    try:
        print(f"\nPushing with dataset_sources={ds_sources}, kernel_sources={ks_sources}")
        r = push(ds_sources, kernel_sources=ks_sources)
        err = getattr(r, "error", "") or ""
        if err and ("not found" in err.lower() or "404" in str(err)):
            raise RuntimeError(err)
        url = getattr(r, "url", None) or getattr(r, "ref", None)
        ver = getattr(r, "version_number", None)
        print(f"URL: {url}")
        print(f"Version: {ver}")
        if err:
            print(f"Error: {err}")
        pushed = True
        break
    except Exception as e:
        msg = str(e)
        print(f"  failed ({msg[:200]})")
        last_err = msg
        if "not found" in msg.lower() or "404" in msg or "exp014-state-r2-fast" in msg:
            continue
        else:
            raise

if not pushed:
    print(f"\nAll push attempts failed: {last_err}")
    sys.exit(1)
