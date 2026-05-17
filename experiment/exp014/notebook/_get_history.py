"""Just download history.json from train NB output."""
import json, os, tempfile
from pathlib import Path
os.environ["KAGGLE_API_TOKEN"] = json.loads(
    (Path.home() / ".kaggle" / "kaggle.json").read_text()
)["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

SLUG = "maekeso/birdclef2026-exp014-train"
# Try downloading just history.json
with tempfile.TemporaryDirectory() as td:
    try:
        api.kernels_output(SLUG, path=td, force=False)
        # find history.json
        for p in Path(td).rglob("history.json"):
            hist = json.loads(p.read_text())
            print(f"Total epochs logged: {len(hist)}")
            print()
            print(f"{'Ep':4} {'train':>8} {'cls':>8} {'dist':>8} {'val_ns22':>10} {'val_macro':>10} {'lr':>10} {'sec':>6}")
            print("-" * 80)
            for h in hist:
                ep = h.get('epoch', '?')
                tl = h.get('train_loss', 0)
                cl = h.get('cls_loss', 0)
                dl = h.get('dist_loss', 0)
                ns22 = h.get('val_ns22', 0)
                macro = h.get('val_macro', 0)
                lr = h.get('lr', 0)
                sec = h.get('elapsed_sec', 0)
                print(f"{ep:4} {tl:8.4f} {cl:8.4f} {dl:8.4f} {ns22:10.4f} {macro:10.4f} {lr:10.2e} {sec:6.0f}")
            break
        else:
            print("No history.json found in output")
            print("Files:")
            for p in Path(td).rglob("*"):
                if p.is_file():
                    print(f"  {p.relative_to(td)} {p.stat().st_size/1024:.1f}KB")
    except Exception as e:
        print(f"failed: {e}")
