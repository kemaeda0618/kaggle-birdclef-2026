"""Poll V4 kernel status until complete, then download outputs."""
import time, json, os
from pathlib import Path
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp091-v3-exp090-oof"
MAX_WAIT_MIN = 90
POLL_SEC = 60

print(f"[{time.strftime('%H:%M:%S')}] Polling V4 (max {MAX_WAIT_MIN} min, every {POLL_SEC}s)...")
start = time.time()
last_status = None
final = None
while True:
    elapsed_min = (time.time() - start) / 60
    if elapsed_min > MAX_WAIT_MIN:
        print(f"[TIMEOUT] {MAX_WAIT_MIN} min exceeded")
        final = "TIMEOUT"
        break
    try:
        st = api.kernels_status(slug)
        if hasattr(st, "to_dict"):
            d = st.to_dict()
        elif isinstance(st, dict):
            d = st
        else:
            d = vars(st) if hasattr(st, "__dict__") else {"raw": str(st)}
        status = d.get("status", "?")
        if status != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] status={status} ({elapsed_min:.1f} min)")
            last_status = status
        if str(status).lower() in {"complete", "error"}:
            final = status
            break
    except Exception as e:
        print(f"  poll ERR: {type(e).__name__}: {str(e)[:200]}")
    time.sleep(POLL_SEC)

print(f"\n[{time.strftime('%H:%M:%S')}] FINAL status: {final}")
print(f"[{time.strftime('%H:%M:%S')}] Downloading outputs...")

DATA_DIR = Path("experiment/streamlit_analyze/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
try:
    api.kernels_output(slug, path=str(DATA_DIR), quiet=False, force=True)
    print(f"[{time.strftime('%H:%M:%S')}] Download done")
except Exception as e:
    print(f"download ERR: {type(e).__name__}: {str(e)[:300]}")

print(f"\nFiles in {DATA_DIR}:")
for p in sorted(DATA_DIR.iterdir()):
    if p.is_file():
        print(f"  {p.name}  {p.stat().st_size/1e6:.2f}MB")

# Check V4 patch worked: look for V4 PATCH marker in log
log_path = DATA_DIR / "birdclef2026-exp091-v3-exp090-oof.log"
if log_path.exists():
    content = log_path.read_text(encoding="utf-8", errors="replace")
    if "[V4 PATCH] Using 66 labeled SS files" in content:
        print("\n[OK] V4 patch active: 66 labeled SS files used")
    elif "V4 PATCH" in content:
        # Show what number
        for ln in content.split("\\n"):
            if "V4 PATCH" in ln:
                print(f"\n[V4] {ln.strip()[:200]}")
    else:
        print("\n[WARN] V4 PATCH marker not in log — patch may not have triggered")

    # Look for errors
    for keyword in ["Traceback", "NameError", "ValueError", "RuntimeError"]:
        if keyword in content:
            # Find context
            idx = content.find(keyword)
            ctx = content[max(0, idx-200):idx+500]
            print(f"\n[ERROR FOUND] '{keyword}' in log:")
            print(ctx.replace("\\n", "\n")[-700:])
            break
