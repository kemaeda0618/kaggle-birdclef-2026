"""Push exp013 fine-tune notebook to Kaggle.

Usage:
  python push_nb_finetune.py <model_key>            # first push (uses v2)
  python push_nb_finetune.py <model_key> --resume   # resume
  python push_nb_finetune.py <model_key> --v1       # use legacy v1 (broken, kept for ref)

model_key in {"eca_nfnet_l0", "b3", "b0_amphibia"}
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

# --- Argument parsing ---
if len(sys.argv) < 2:
    print("Usage: python push_nb_finetune.py <model_key> [--resume]")
    print("  model_key: eca_nfnet_l0 | b3 | b0_amphibia")
    sys.exit(1)

MODEL_KEY = sys.argv[1]
INCLUDE_SELF = "--resume" in sys.argv
USE_V1 = "--v1" in sys.argv

# Per-model slug
SLUGS = {
    "eca_nfnet_l0": "birdclef2026-exp013-eca-nfnet-l0",
    "b3":           "birdclef2026-exp013-b3",
    "b0_amphibia":  "birdclef2026-exp013-b0-amphibia",
}
TITLES = {
    "eca_nfnet_l0": "birdclef2026 exp013 eca nfnet l0",
    "b3":           "birdclef2026 exp013 b3",
    "b0_amphibia":  "birdclef2026 exp013 b0 amphibia",
}
assert MODEL_KEY in SLUGS, f"unknown model_key: {MODEL_KEY}"

NB_NAME = f"nb_finetune_{MODEL_KEY}.ipynb" if USE_V1 else f"nb_finetune_v2_{MODEL_KEY}.ipynb"
NB    = Path(__file__).with_name(NB_NAME)
assert NB.exists(), f"NB not found: {NB}"
USER  = "maekeso"
SLUG  = SLUGS[MODEL_KEY]
TITLE = TITLES[MODEL_KEY]

print(f"Pushing {MODEL_KEY}: slug={SLUG}, resume={INCLUDE_SELF}")

kernel_sources = []
if INCLUDE_SELF:
    kernel_sources.append(f"{USER}/{SLUG}")

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
        "machine_shape": "NvidiaTeslaT4",   # T4x2
        "enable_internet": True,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            "nikitababich/birdclef2025-1st-place-ensemble",
        ],
        "kernel_sources": kernel_sources,
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
