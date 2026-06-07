"""Inspect top public NBs for details: descriptions, dataset sources, etc."""
import json, os, tempfile
from pathlib import Path

if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

candidates = [
    # 明示 0.950 系
    "anthonytherrien/birdclef-2026-ensemble-0-950",
    "yaroslavkholmirzayev/0950-replay",
    # Transformer 軸 (paradigm 新軸)
    "liuyanfeng/birdclef-2026-add-transformer",
    "liuyanfeng/birdclef-2026-development-try-transformer",
    # 最新 EoS series (iterative improvement)
    "fleongg/birdclef-2026-eos-9-claude-fork",
    "ryutoyoda/birdclef-2026-exp013-eos8-sidecar",
    # 0.934 明示 (mid-tier)
    "needless090/birdclef-2026-iter-pseudo-perch-sed-lb-0-934-s",
    # Two-Pass SSM (interesting variant)
    "marynaborovska/birdclef-26-two-pass-ssm-advanced-pp",
    # Multi-Model Ensemble (interesting structure)
    "raunakdey07/birdclef-2026-multi-model-ensemble",
    # Distill / SED variants
    "tuckerarrants/bc2026-distilled-sed",
    # Karnak series (newest iterations)
    "gendaijin/birdclef2026-day0528-karnak-hier",
    "gendaijin/birdclef2026-day0529-raunak-v9-latest",
    # Kitchen / Hybrid
    "thomaszyxu/bc26-v2538d-smart-hybrid",
]

for ref in candidates:
    print(f"\n{'='*70}")
    print(f"NB: {ref}")
    print('='*70)
    try:
        # Get kernel info via listing with search
        nbs = api.kernels_list(search=ref.split('/')[-1], page_size=5)
        for n in nbs:
            n_ref = getattr(n, "ref", "")
            if n_ref == ref:
                title = getattr(n, "title", "?")
                votes = getattr(n, "totalVotes", 0)
                last_run = getattr(n, "lastRunTime", "?")
                lang = getattr(n, "language", "?")
                print(f"  Title:    {title}")
                print(f"  Votes:    {votes}")
                print(f"  Lang:     {lang}")
                print(f"  Last run: {last_run}")
                break
        else:
            print(f"  (info not found in search)")
    except Exception as e:
        print(f"  ERR fetching info: {type(e).__name__}: {str(e)[:120]}")

    # Get metadata via download (metadata only, not files)
    try:
        with tempfile.TemporaryDirectory() as td:
            api.kernels_pull(ref, path=td, metadata=True, quiet=True)
            meta = json.load(open(Path(td) / "kernel-metadata.json"))
            print(f"  Datasets: {meta.get('dataset_sources', [])}")
            print(f"  Kernels:  {meta.get('kernel_sources', [])}")
            print(f"  Models:   {meta.get('model_sources', [])}")
            print(f"  Internet: {meta.get('enable_internet', '?')}")
            print(f"  GPU:      {meta.get('enable_gpu', '?')}")
    except Exception as e:
        print(f"  ERR fetching metadata: {type(e).__name__}: {str(e)[:120]}")
