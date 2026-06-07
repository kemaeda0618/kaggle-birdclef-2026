import json, os, tempfile
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Try kernel files (kernel output is more accessible)
print("=== Kernel output files ===")
try:
    files = api.kernels_list_files("maekeso/birdclef2026-exp083-train-r1")
    for f in files.files[:30]:
        print(f"  {f.name} ({f.size})")
except Exception as e:
    print(f"  ERR: {type(e).__name__}: {str(e)[:200]}")

# Try downloading history.json from kernel output
print("\n=== Try DL history.json from kernel output ===")
try:
    with tempfile.TemporaryDirectory() as td:
        api.kernels_output("maekeso/birdclef2026-exp083-train-r1", path=td, quiet=False)
        # Find history.json
        hist_files = list(Path(td).rglob("history.json"))
        if hist_files:
            for hf in hist_files:
                print(f"\n  Found: {hf} ({hf.stat().st_size} bytes)")
                hist = json.loads(hf.read_text(encoding="utf-8"))
                print(f"  Total epochs trained: {len(hist)}")
                if hist:
                    best_ns22 = -1
                    best_ep = 0
                    for h in hist:
                        v = h.get("val_ns22")
                        if v is not None and isinstance(v, (int, float)) and v > best_ns22 and v == v:
                            best_ns22 = v
                            best_ep = h.get("epoch", 0)
                    best_macro = -1
                    best_ep_m = 0
                    for h in hist:
                        v = h.get("val_macro")
                        if v is not None and isinstance(v, (int, float)) and v > best_macro and v == v:
                            best_macro = v
                            best_ep_m = h.get("epoch", 0)
                    print(f"\n  Best val_ns22:  {best_ns22:.4f} at epoch {best_ep}")
                    print(f"  Best val_macro: {best_macro:.4f} at epoch {best_ep_m}")
                    # Print last 5 epochs
                    print(f"\n  Last 5 epochs:")
                    for h in hist[-5:]:
                        ep = h.get("epoch")
                        ns22 = h.get("val_ns22")
                        macro = h.get("val_macro")
                        loss = h.get("train_loss")
                        print(f"    ep {ep:2d}: train_loss={loss:.4f} val_ns22={ns22:.4f} val_macro={macro:.4f}")
                        if "val_per_taxon" in h:
                            print(f"           taxon: {h['val_per_taxon']}")
        else:
            print("  history.json not found in output")
            print(f"  Files in output: {[p.name for p in Path(td).iterdir()]}")
except Exception as e:
    print(f"  ERR: {type(e).__name__}: {str(e)[:300]}")
