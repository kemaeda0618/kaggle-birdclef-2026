import json, os, tempfile
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Check kernel status
for slug in ["birdclef2026-exp083-train-r1"]:
    print(f"=== Kernel: maekeso/{slug} ===")
    try:
        st = api.kernels_status(f"maekeso/{slug}")
        print(f"  Status: {st}")
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {str(e)[:200]}")

# Check state Dataset
for slug in ["exp083-state"]:
    print(f"\n=== Dataset: maekeso/{slug} ===")
    try:
        files = api.dataset_list_files(f"maekeso/{slug}")
        for f in files.files[:20]:
            sz = getattr(f, "total_bytes", None) or getattr(f, "totalBytes", "?")
            print(f"  {f.name} ({sz})")
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {str(e)[:200]}")

# Try to peek at history.json to see val metrics
print(f"\n=== Peeking at history.json (val metrics) ===")
try:
    with tempfile.TemporaryDirectory() as td:
        api.dataset_download_file("maekeso/exp083-state", "history.json", path=td, force=True)
        # File may have .zip extension
        p = Path(td) / "history.json"
        if not p.exists():
            zp = Path(td) / "history.json.zip"
            if zp.exists():
                import zipfile
                with zipfile.ZipFile(zp) as z:
                    z.extractall(td)
        if p.exists():
            hist = json.loads(p.read_text(encoding="utf-8"))
            print(f"  Total epochs: {len(hist)}")
            if hist:
                # Best ns22 / macro
                best_ns22 = max((h.get("val_ns22", 0) for h in hist if not (h.get("val_ns22") is None or (isinstance(h.get("val_ns22"), float) and h.get("val_ns22") != h.get("val_ns22")))), default=0)
                best_macro = max((h.get("val_macro", 0) for h in hist if not (h.get("val_macro") is None or (isinstance(h.get("val_macro"), float) and h.get("val_macro") != h.get("val_macro")))), default=0)
                print(f"  Best val_ns22:  {best_ns22:.4f}")
                print(f"  Best val_macro: {best_macro:.4f}")
                # Find best epoch
                for h in hist:
                    if h.get("val_ns22") == best_ns22:
                        print(f"  Best epoch (ns22): ep {h['epoch']}, train_loss={h.get('train_loss')}, taxon={h.get('val_per_taxon')}")
                        break
                # Final epoch
                last = hist[-1]
                print(f"\n  Final ep {last['epoch']}: val_ns22={last.get('val_ns22'):.4f}, val_macro={last.get('val_macro'):.4f}")
                print(f"    taxon: {last.get('val_per_taxon')}")
except Exception as e:
    print(f"  ERR: {type(e).__name__}: {str(e)[:300]}")
