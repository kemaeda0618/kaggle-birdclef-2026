import json, os, tempfile
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

print("=== exp084-weights Dataset files ===")
try:
    files = api.dataset_list_files("maekeso/birdclef2026-exp084-weights")
    for f in files.files[:25]:
        sz = getattr(f, "total_bytes", None) or getattr(f, "totalBytes", "?")
        print(f"  {f.name} ({sz})")
except Exception as e:
    print(f"  ERR: {e}")

# Try to get history.json
print("\n=== Peek history (R1 or R2) ===")
try:
    with tempfile.TemporaryDirectory() as td:
        for cand in ["r1/history.json", "r2/history.json", "history.json"]:
            try:
                api.dataset_download_file("maekeso/birdclef2026-exp084-weights", cand,
                                          path=td, force=True)
                p = Path(td) / Path(cand).name
                if not p.exists():
                    zp = Path(td) / (Path(cand).name + ".zip")
                    if zp.exists():
                        import zipfile
                        with zipfile.ZipFile(zp) as z: z.extractall(td)
                if p.exists():
                    print(f"  Found: {cand}")
                    hist = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(hist, list) and hist:
                        best_ns22 = max((h.get("val_ns22", 0) for h in hist if h.get("val_ns22") is not None), default=0)
                        best_macro = max((h.get("val_macro", 0) for h in hist if h.get("val_macro") is not None), default=0)
                        print(f"    Total epochs: {len(hist)}")
                        print(f"    Best val_ns22:  {best_ns22:.4f}")
                        print(f"    Best val_macro: {best_macro:.4f}")
                        for h in hist:
                            if h.get("val_ns22") == best_ns22:
                                print(f"    Best ep {h.get('epoch')}: taxon={h.get('val_per_taxon')}")
                                break
                        last = hist[-1]
                        print(f"    Last ep {last.get('epoch')}: val_ns22={last.get('val_ns22'):.4f}")
                        break
            except Exception:
                continue
except Exception as e:
    print(f"  ERR: {e}")
