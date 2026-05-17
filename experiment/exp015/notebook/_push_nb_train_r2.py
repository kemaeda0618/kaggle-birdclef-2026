"""Push exp015 R2 nb_train_r2.ipynb to Kaggle (T4x2 GPU, internet ON, self-resume).

Handles two graceful-fallback datasets:
- maekeso/exp015-r1-pseudo: REQUIRED for R2 training, but won't exist before
  nb_pseudo finishes. Push will fail at training time, but kernel push itself
  should succeed if the dataset was published first. If push fails due to
  missing pseudo, fall back to push without it (will fail at runtime).
- maekeso/exp015-state-r2: gracefully missing on first run (just like R1).

Usage:  python _push_nb_train_r2.py

Slug rule (title slugifies to slug):
  title = "birdclef2026 exp015 train r2"
  slug  = "birdclef2026-exp015-train-r2"
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

NB    = Path(__file__).with_name("nb_train_r2.ipynb")
USER  = "maekeso"
SLUG  = "birdclef2026-exp015-train-r2"
TITLE = "birdclef2026 exp015 train r2"


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


PERCH      = "tuckerarrants/perch-v2-no-dft-onnx"
R1_PSEUDO  = f"{USER}/birdclef2026-exp015-pseudo-r1"   # match exp014 convention (URL確認済)
R2_STATE   = f"{USER}/exp015-state-r2"
SELF_KERNEL = f"{USER}/{SLUG}"  # for self-resume via kernel_sources

# Try in order of completeness; fall back as needed.
# kernel_sources includes itself so Session N+1 can read Session N's ckpt_latest.pth
ATTEMPTS = [
    ([PERCH, R1_PSEUDO, R2_STATE], [SELF_KERNEL]),
    ([PERCH, R1_PSEUDO], [SELF_KERNEL]),
    ([PERCH], [SELF_KERNEL]),
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
        if "not found" in msg.lower() or "404" in msg or "exp015-" in msg:
            print(f"  -> retrying with smaller dataset_sources list")
            continue
        else:
            raise

if not pushed:
    print(f"\nAll push attempts failed: {last_err}")
    sys.exit(1)
