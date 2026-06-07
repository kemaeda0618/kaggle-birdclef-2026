"""Pull selected public NBs and inspect contents."""
import json, os, tempfile, shutil
from pathlib import Path

if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Pull each NB to local dir for inspection
PULL_ROOT = Path("C:/Users/maeke/work/kaggle/birdclef-2026/_public_nb_pulls")
PULL_ROOT.mkdir(exist_ok=True)

candidates = [
    "anthonytherrien/birdclef-2026-ensemble-0-950",
    "yaroslavkholmirzayev/0950-replay",
    "fleongg/birdclef-2026-eos-9-claude-fork",
    "marynaborovska/birdclef-26-two-pass-ssm-advanced-pp",
    "liuyanfeng/birdclef-2026-add-transformer",
    "gendaijin/birdclef2026-day0529-raunak-v9-latest",
    "raunakdey07/birdclef-2026-multi-model-ensemble",
]

for ref in candidates:
    print(f"\n=== {ref} ===")
    out_dir = PULL_ROOT / ref.replace("/", "__")
    out_dir.mkdir(exist_ok=True)
    try:
        api.kernels_pull(ref, path=str(out_dir), quiet=True)
        files = list(out_dir.iterdir())
        for f in files:
            print(f"  {f.name} ({f.stat().st_size/1024:.1f}KB)")
    except Exception as e:
        print(f"  ERR: {type(e).__name__}: {str(e)[:300]}")

print(f"\n[Pulled to {PULL_ROOT}]")
