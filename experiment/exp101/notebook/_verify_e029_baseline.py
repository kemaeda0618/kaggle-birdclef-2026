"""Download exp029 R3 ckpt and read best_ns22 to settle the val ambiguity."""
import json, os, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle"/"kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

OWNER, SLUG = "maekeso", "birdclef2026-exp029-l1-single"
OUT = Path(__file__).parent / "_exp029_dl"
OUT.mkdir(exist_ok=True)

# Just list first
print("=== files ===")
files = api.dataset_list_files(f"{OWNER}/{SLUG}").files
for f in files:
    print(f"  {f.name}")

# Download history.json (smallest, has all val curves)
print("\n=== downloading history ===")
for fname_candidate in ["r3_fold0_history.json", "history.json", "r3_history.json"]:
    try:
        api.dataset_download_file(f"{OWNER}/{SLUG}", fname_candidate, path=str(OUT), force=True)
        print(f"  downloaded: {fname_candidate}")
        break
    except Exception as e:
        print(f"  {fname_candidate}: {e}")
        continue

# Print contents of any history.json found
print("\n=== history content ===")
for f in OUT.iterdir():
    if "history" in f.name:
        print(f"\n--- {f.name} ---")
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for ep_data in data[-5:]:  # last 5 epochs
                    print(f"  ep{ep_data.get('epoch','?')}: val_ns22={ep_data.get('val_ns22','?')} val_macro={ep_data.get('val_macro','?')}")
                # Best ns22
                best = max(data, key=lambda x: x.get("val_ns22", -1) or -1)
                print(f"  best ns22 over all epochs: ep{best.get('epoch')} = {best.get('val_ns22')}")
            else:
                print(data)
        except Exception as e:
            print(f"  parse err: {e}")
