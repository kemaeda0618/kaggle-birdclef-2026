"""Check exp084 R1 kernel run status + val_ns22."""
import json, os, tempfile
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

candidates = [
    "maekeso/birdclef2026-exp084-train-r1",
    "maekeso/birdclef2026-exp084-r1",
    "maekeso/birdclef2026-exp084-train",
    "maekeso/birdclef2026-exp084-train-r1-convnext-pico",
    "maekeso/birdclef2026-exp084-train-r1-hi-time",
]
found_any = False
for slug in candidates:
    try:
        with tempfile.TemporaryDirectory() as td:
            api.kernels_output(slug, path=td, quiet=True)
            td_path = Path(td)
            log_files = list(td_path.rglob("*.log"))
            print(f"=== {slug} ===")
            found_any = True
            if log_files:
                content = log_files[0].read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
                ns22_lines = [ln for ln in lines if "val_ns22" in ln.lower() or "BEST" in ln or "Best" in ln]
                if ns22_lines:
                    for ln in ns22_lines[-15:]:
                        if '"data":"' in ln:
                            try:
                                idx = ln.index('"data":"')
                                snippet = ln[idx+8:].split('\\n')[0][:200]
                                print(f"  {snippet}")
                            except: print(f"  {ln[:200]}")
                        else:
                            print(f"  {ln[:200]}")
                else:
                    print("  (no val_ns22 lines found in log)")
    except Exception as e:
        msg = str(e)[:120]
        if "404" not in msg and "Not Found" not in msg:
            print(f"{slug}: ERR {type(e).__name__}: {msg}")

if not found_any:
    print("\nNo exp084 R1 kernel found in Kaggle.")
