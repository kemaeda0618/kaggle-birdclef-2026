"""Compare exp112 V2 vs V3 dry-run timing."""
import json, os, base64, requests, re
from pathlib import Path

kj = Path.home() / ".kaggle" / "kaggle.json"
creds = json.loads(kj.read_text())
token = creds["key"]
headers = {"Authorization": f"Bearer {token}"} if token.startswith("KGAT_") else None

VERSIONS = [
    ("V2_unbatched", "323333136"),
    ("V3_batched",   "323350981"),
]

for name, version_id in VERSIONS:
    print(f"\n{'='*70}\n{name} (version {version_id})\n{'='*70}")
    url = f"https://www.kaggle.com/api/v1/kernels/output?userName=maekeso&kernelSlug=birdclef2026-exp112-e106-fold12&versionNumber={version_id}"
    r = requests.get(url, headers=headers)
    j = r.json()
    log = j.get("log", "")
    if not log:
        print("  [no log]")
        continue

    # Parse JSON-like entries for timing
    last_time = 0.0
    for line in log.split("\n"):
        m = re.search(r'"time":([0-9.]+),"data":"(.*?)"\}', line)
        if not m:
            continue
        t = float(m.group(1))
        d = m.group(2).encode().decode("unicode_escape", errors="ignore")
        last_time = t
        if any(kw in d for kw in [
            "Perch pass", "ProtoSSM training", "Trained ",
            "Total wall time", "audio_cache built",
            "SED OV", "SED-OV", "e106-ov", "e102-ov",
            "compiled", "[ov-install]", "Submission",
        ]):
            print(f"  {t:7.1f}s  {d.strip()[:160]}")

    print(f"  ---- LAST EVENT @ {last_time:.1f}s = {last_time/60:.2f} min ----")
