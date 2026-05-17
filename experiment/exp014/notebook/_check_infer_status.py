"""Check exp014-infer NB status + log."""
import json, os
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

SLUG = "maekeso/birdclef2026-exp014-infer"

# Status
print(f"=== {SLUG} status ===")
try:
    info = api.kernels_status(SLUG)
    print(info)
except Exception as e:
    print(f"  status check failed: {e}")

# Output files
print(f"\n=== Output files ===")
import tempfile
try:
    with tempfile.TemporaryDirectory() as td:
        api.kernels_output(SLUG, path=td)
        for p in sorted(Path(td).rglob("*")):
            if p.is_file():
                sz = p.stat().st_size
                print(f"  {p.relative_to(td)!s:50s} {sz/1024:.1f} KB")
        # Show log content
        log_file = next(Path(td).rglob("*.log"), None)
        if log_file:
            print(f"\n=== {log_file.name} (last 80 lines) ===")
            content = log_file.read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")
            for line in lines[-80:]:
                print(line)
except Exception as e:
    print(f"  output fetch failed: {e}")
