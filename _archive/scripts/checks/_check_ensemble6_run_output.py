"""Fetch ensemble6 kernel output log to verify execution was clean."""
import json, os, tempfile
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Try to download kernel output (log)
print("=== Fetching kernel output ===")
try:
    with tempfile.TemporaryDirectory() as td:
        api.kernels_output("maekeso/birdclef2026-ensemble6-infer", path=td, quiet=False)
        # Look for log file
        td_path = Path(td)
        for f in td_path.rglob("*"):
            if f.is_file():
                if f.suffix == ".log" or f.name == "submission.csv":
                    print(f"  Found: {f.name} ({f.stat().st_size/1024:.1f}KB)")

        # Read log
        log_files = list(td_path.rglob("*.log"))
        if log_files:
            log = log_files[0]
            print(f"\n=== Log content (last 200 lines) ===\n")
            content = log.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            for ln in lines[-200:]:
                print(ln)
except Exception as e:
    print(f"  ERR: {type(e).__name__}: {str(e)[:300]}")
