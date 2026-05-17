"""Check train_r2 kernel latest output (look for ckpt_latest.pth from Session 1)."""
import json, os, sys, io, tempfile
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

USER = "maekeso"
SLUG = "birdclef2026-exp014-train-r2"
ref  = f"{USER}/{SLUG}"

print(f"=== Kernel: {ref} ===")
try:
    s = api.kernels_status(ref)
    print(f"  status: {getattr(s, 'status', '?')}")
    print(f"  failureMessage: {getattr(s, 'failureMessage', None)}")
except Exception as e:
    print(f"  status check ERROR: {e}")

print(f"\n=== List output files ===")
try:
    out = api.kernels_list_files(ref)
    if hasattr(out, 'files'):
        for f in out.files[:30]:
            sz = getattr(f, 'totalBytes', getattr(f, 'total_bytes', 0))
            print(f"  {f.name:60s} {sz/1024/1024:.2f} MB")
        print(f"  Total: {len(out.files)} files")
    else:
        print(f"  Unexpected response: {out}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")
