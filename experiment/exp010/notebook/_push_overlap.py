"""Generate species_overlap.ipynb and push to Kaggle as v2."""
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8")

# Step 1: regenerate notebook
r = subprocess.run(
    [sys.executable, str(HERE / "_gen_species_overlap.py")],
    capture_output=True, text=True, encoding="utf-8",
)
print(r.stdout)
if r.returncode != 0:
    print("GEN ERROR:", r.stderr)
    sys.exit(1)

# Step 2: push to Kaggle
import requests

creds = json.loads(Path.home().joinpath(".kaggle/kaggle.json").read_text())
token = creds["key"]

nb = json.loads((HERE / "species_overlap.ipynb").read_text(encoding="utf-8"))

payload = {
    "slug": "birdclef2026-species-overlap",
    "text": json.dumps(nb),
    "language": "python",
    "kernelType": "notebook",
    "kernelExecutionType": "SaveAndRunAll",
    "isPrivate": True,
    "enableInternet": True,
    "competitionDataSources": ["birdclef-2026"],
}

r2 = requests.post(
    "https://www.kaggle.com/api/v1/kernels/push",
    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    json=payload,
    timeout=60,
)
print("Status:", r2.status_code)
print(json.dumps(r2.json(), indent=2))
