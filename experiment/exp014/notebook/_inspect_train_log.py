"""Inspect exp014 train log for issues."""
import json, os, tempfile
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

SLUG = "maekeso/birdclef2026-exp014-train"

with tempfile.TemporaryDirectory() as td:
    api.kernels_output(SLUG, path=td)
    td_path = Path(td)

    # Find log + history
    log_file = next(td_path.rglob("*.log"), None)
    hist_file = next(td_path.rglob("history.json"), None)

    # Extract stdout from log (papermill format)
    if log_file:
        print(f"=== Stdout lines from {log_file.name} ===")
        content = log_file.read_text(encoding="utf-8", errors="replace")
        stdout_lines = []
        for line in content.split("\n"):
            line = line.strip().rstrip(",")
            if not line: continue
            try:
                obj = json.loads(line)
                if obj.get("stream_name") == "stdout":
                    stdout_lines.append(obj.get("data", "").rstrip("\n"))
            except:
                pass
        # Print all stdout
        full = "\n".join(stdout_lines)
        print(full[-15000:])  # last 15K chars

    # Print history
    if hist_file:
        print(f"\n\n=== history.json ===")
        hist = json.loads(hist_file.read_text())
        print(f"Total epochs logged: {len(hist)}")
        for h in hist:
            print(f"  Ep{h['epoch']:2d}: train_loss={h['train_loss']:.4f}, "
                  f"cls={h.get('cls_loss', '?'):.4f if isinstance(h.get('cls_loss'), float) else 0}, "
                  f"dist={h.get('dist_loss', '?'):.4f if isinstance(h.get('dist_loss'), float) else 0}, "
                  f"ns22={h.get('val_ns22', '?'):.4f if isinstance(h.get('val_ns22'), float) else 0}, "
                  f"macro={h.get('val_macro', '?'):.4f if isinstance(h.get('val_macro'), float) else 0}, "
                  f"lr={h.get('lr', '?'):.2e if isinstance(h.get('lr'), float) else 0}, "
                  f"{h.get('elapsed_sec', 0):.0f}s")
