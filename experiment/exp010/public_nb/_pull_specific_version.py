"""Try pulling specific version of mtoshidesu/0-932-test-mod-version-4 (scriptVersionId=313904345)."""
import json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Try listing all versions first
import inspect
try:
    print("Available kernels_* methods:")
    for n in sorted(dir(api)):
        if "kernel" in n.lower() and not n.startswith("_"):
            print(f"  {n}")
except Exception as e:
    print(f"err: {e}")

# Pull with kernel_version parameter
import tempfile, shutil
target_dir = r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\mtoshidesu_v313904345"
Path(target_dir).mkdir(parents=True, exist_ok=True)

slug = "mtoshidesu/0-932-test-mod-version-4"
target_ver = 313904345

# Inspect kernels_pull signature
print(f"\nkernels_pull signature:")
print(f"  {inspect.signature(api.kernels_pull)}")

# Try with kernel_version
try:
    api.kernels_pull(slug, path=target_dir, metadata=True, kernel_version=target_ver)
    print(f"OK pulled version {target_ver} to {target_dir}")
except TypeError as e:
    print(f"kernel_version not supported: {e}")
    # Fall back to default pull (latest)
    api.kernels_pull(slug, path=target_dir, metadata=True)
    print(f"Pulled LATEST (kernel_version not supported)")
except Exception as e:
    print(f"err: {type(e).__name__}: {str(e)[:300]}")
