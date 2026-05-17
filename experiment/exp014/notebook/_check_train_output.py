"""Check exp014-train NB output."""
import json, os
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

SLUG = "maekeso/birdclef2026-exp014-train"

print(f"=== {SLUG} status ===")
print(api.kernels_status(SLUG))

# Output files
print(f"\n=== Output files ===")
import tempfile
try:
    with tempfile.TemporaryDirectory() as td:
        api.kernels_output(SLUG, path=td)
        for p in sorted(Path(td).rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(td)!s:50s} {p.stat().st_size/1024/1024:.2f} MB")
        # Tail of log
        log_file = next(Path(td).rglob("*.log"), None)
        if log_file:
            content = log_file.read_text(encoding="utf-8", errors="replace")
            # Find upload section
            print("\n=== Log tail (last 100 lines) ===")
            for line in content.split("\n")[-100:]:
                if "data" in line:
                    # papermill format
                    try:
                        obj = json.loads(line.rstrip(","))
                        print(obj.get("data", ""), end="")
                    except:
                        print(line[:200])
                else:
                    print(line[:200])
except Exception as e:
    print(f"  failed: {e}")
