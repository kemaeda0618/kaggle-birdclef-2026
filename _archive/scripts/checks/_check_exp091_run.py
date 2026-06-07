"""Check exp091 analysis NB run status + log."""
import json, os, tempfile, re
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp091-analysis-infer"
with tempfile.TemporaryDirectory() as td:
    try:
        api.kernels_output(slug, path=td, quiet=True)
    except Exception as e:
        print(f"output ERR: {e}")
    log_files = list(Path(td).rglob("*.log"))
    if not log_files:
        print("No log")
        raise SystemExit(0)
    content = log_files[0].read_text(encoding="utf-8", errors="replace")
    # Try JSON parse first (more robust than regex)
    try:
        entries = json.loads(content)
        for e in entries:
            if isinstance(e, dict) and "data" in e:
                s = e["data"].rstrip("\n")
                if s.strip(): print(s)
    except Exception as ex:
        print(f"[JSON parse failed: {ex}, falling back to raw print last 5KB]")
        print(content[-5000:])
