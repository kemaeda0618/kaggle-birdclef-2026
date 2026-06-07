"""Fetch dry-run timing from key sub logs."""
import json, os, base64, requests, re
from pathlib import Path

kj = Path.home() / ".kaggle" / "kaggle.json"
creds = json.loads(kj.read_text())
token = creds["key"]
if token.startswith("KGAT_"):
    headers = {"Authorization": f"Bearer {token}"}
else:
    basic = base64.b64encode(f"{creds['username']}:{token}".encode()).decode()
    headers = {"Authorization": f"Basic {basic}"}

SLUGS = [
    "birdclef2026-exp090-tuned-tax",
    "birdclef2026-exp111-e102-fold12",
    "birdclef2026-exp104-standalone-3fold",
    "birdclef2026-exp104-v3-tucker3fold",
    "birdclef2026-exp106-standalone-3fold",
    "birdclef2026-exp112-e106-fold12",
]

for slug in SLUGS:
    print(f"\n{'='*70}\n{slug}\n{'='*70}")
    url = f"https://www.kaggle.com/api/v1/kernels/output?userName=maekeso&kernelSlug={slug}"
    r = requests.get(url, headers=headers)
    if not r.ok:
        print(f"  [ERR] {r.status_code}: {r.text[:200]}")
        continue
    j = r.json()
    log = j.get("log", "")
    if not log:
        print(f"  [no log] hasLog={j.get('hasLog')}")
        continue

    # Find time markers
    last_time = 0.0
    audio_cache_files = None
    wall_time_total = None
    inference_done_match = None

    for line in log.split("\n"):
        m = re.search(r'"time":([0-9.]+),"data":"(.*?)"\}', line)
        if not m:
            continue
        t = float(m.group(1))
        d = m.group(2).encode().decode("unicode_escape", errors="ignore")
        last_time = t
        # interesting lines
        if any(kw in d for kw in [
            "audio_cache built", "Total wall time", "submission.csv saved",
            "Submission:", "SED OV done", "e102-ov done", "e106-ov done",
            "e17 done", "SED done", "Perch pass", "ProtoSSM training",
            "MLP probes for", "Trained ", "ResidualSSM training",
            "Total time:", "all_probs shape",
        ]):
            print(f"  {t:7.1f}s  {d.strip()[:160]}")

    print(f"  ---- LAST EVENT @ {last_time:.1f}s = {last_time/60:.2f} min ----")
