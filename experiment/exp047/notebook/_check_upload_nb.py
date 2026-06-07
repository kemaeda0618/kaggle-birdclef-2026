"""Check AnuraSet upload-only NB status + log."""
import json, os, re
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle/kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp047-anuraset-upload"
print(f"=== {slug} status ===")
print(api.kernels_status(slug))

out = Path("/tmp/upload_log")
out.mkdir(parents=True, exist_ok=True)
try:
    api.kernels_output(slug, str(out))
except Exception as e:
    print(f"DL err: {e}")

print(f"\n=== Log key lines ===")
for f in out.rglob("*.log"):
    text = f.read_text(encoding="utf-8", errors="ignore")
    for line in text.split("\n"):
        if any(k in line for k in ["Source","Copying","audio files","metadata","cleanup","Uploading","OK","err","URL:","RuntimeError","successful"]):
            m = re.search(r'"data":"([^"]+)"', line)
            if m: line = m.group(1)
            line = line.strip().replace("\\n", "")
            if line and len(line) < 300:
                print(f"  {line}")
    break

# Re-check Dataset existence
print(f"\n=== Dataset existence check ===")
try:
    files = api.dataset_list_files("maekeso/birdclef2026-exp047-anuraset-extracted")
    n = len(files.files) if hasattr(files, 'files') else 0
    print(f"  ✓ EXISTS: {n} files visible")
except Exception as e:
    print(f"  ✗ NOT FOUND: {str(e)[:120]}")
