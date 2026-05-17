"""Wait for Perch pre-compute NB to complete, then push R2-fast NB.

Polls Kaggle kernel status every 2 minutes. On completion:
1. Waits 60s for dataset upload to finalize
2. Verifies the emb cache dataset exists
3. Pushes R2-fast NB
"""
import json, os, sys, io, time, subprocess
from pathlib import Path

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

if not os.environ.get("KAGGLE_API_TOKEN"):
    creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    key = creds.get("key", "")
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

USER = "maekeso"
SLUG_PRECOMPUTE = "birdclef2026-exp014-perch-precompute"
EMB_DATASET = f"{USER}/birdclef2026-perch-emb-cache"

POLL_INTERVAL_SEC = 120
MAX_POLL_HOURS = 6.0  # safety guard


def status_str(s):
    if isinstance(s, dict):
        return s.get("status", "UNKNOWN")
    s_str = str(s)
    # kernels_status returns JSON-like string: {"status": "RUNNING", "failureMessage": null}
    try:
        parsed = json.loads(s_str)
        return parsed.get("status", "UNKNOWN")
    except Exception:
        return s_str


t0 = time.time()
print(f"Polling {USER}/{SLUG_PRECOMPUTE} every {POLL_INTERVAL_SEC}s")
print(f"  start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

last_status = None
while True:
    elapsed_hr = (time.time() - t0) / 3600
    if elapsed_hr > MAX_POLL_HOURS:
        print(f"[abort] polling exceeded {MAX_POLL_HOURS}h")
        sys.exit(1)

    try:
        st = api.kernels_status(f"{USER}/{SLUG_PRECOMPUTE}")
    except Exception as e:
        print(f"  [{time.strftime('%H:%M:%S')}] kernels_status error: {e}")
        time.sleep(POLL_INTERVAL_SEC)
        continue

    s = status_str(st)
    s_lower = s.lower()
    if s != last_status:
        print(f"  [{time.strftime('%H:%M:%S')}] status: {s}", flush=True)
        last_status = s

    if "complete" in s_lower:
        print(f"\n>>> Pre-compute COMPLETE after {elapsed_hr*60:.1f} min")
        break
    if any(k in s_lower for k in ["error", "cancel", "fail"]):
        print(f"\n!!! Pre-compute FAILED: {st}")
        sys.exit(1)

    time.sleep(POLL_INTERVAL_SEC)


print("\nWaiting 60s for dataset upload to finalize...")
time.sleep(60)

# Verify emb cache dataset exists
print(f"\nVerifying {EMB_DATASET} exists...")
try:
    ds_files = api.dataset_list_files(EMB_DATASET).files
    print(f"  found {len(ds_files)} files:")
    for f in ds_files[:10]:
        print(f"    {f.name}")
    has_emb = any("emb.npy" in str(f.name) for f in ds_files)
    has_meta = any("meta.csv" in str(f.name) for f in ds_files)
    if not (has_emb and has_meta):
        print(f"!!! dataset incomplete: has_emb={has_emb}, has_meta={has_meta}")
        print(f"    waiting another 120s and retrying...")
        time.sleep(120)
        ds_files = api.dataset_list_files(EMB_DATASET).files
        has_emb = any("emb.npy" in str(f.name) for f in ds_files)
        has_meta = any("meta.csv" in str(f.name) for f in ds_files)
        if not (has_emb and has_meta):
            print(f"!!! still incomplete; aborting before pushing R2-fast")
            sys.exit(1)
    print(f"  OK: emb.npy + meta.csv present")
except Exception as e:
    print(f"  dataset_list_files error: {e}")
    print(f"  proceeding to push anyway (may rely on kernel_sources fallback)")

# Push R2-fast NB
print("\nPushing R2-fast NB...")
push_script = Path(__file__).with_name("_push_nb_train_r2_fast.py")
result = subprocess.run([sys.executable, str(push_script)], capture_output=False)
print(f"\nPush exit code: {result.returncode}")
if result.returncode == 0:
    print(f"\n>>> R2-fast NB pushed and running on Kaggle")
    print(f"    URL: https://www.kaggle.com/code/{USER}/birdclef2026-exp014-train-r2-fast")
sys.exit(result.returncode)
