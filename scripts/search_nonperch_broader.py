"""Broader search for non-Perch BC2026 NBs.

Strategy:
1. List more pages of BC2026 NBs (page 1-5, 250 NBs)
2. For each, pull metadata only (lightweight)
3. Filter: no 'perch' in dataset_sources/model_sources
4. Show survivors
"""
import json, os, tempfile
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

ALL_KERNELS = []
for page in range(1, 6):
    res = api.kernels_list(
        competition="birdclef-2026",
        sort_by="scoreDescending",
        page_size=50,
        page=page,
    )
    print(f"page {page}: {len(res)}")
    for r in res:
        ALL_KERNELS.append(getattr(r, "ref", None))
    if len(res) < 50:
        break

print(f"\nTotal kernels: {len(ALL_KERNELS)}")

# Pull metadata only (fast)
non_perch = []
for i, slug in enumerate(ALL_KERNELS):
    if i % 25 == 0:
        print(f"  scanning {i}/{len(ALL_KERNELS)}...")
    with tempfile.TemporaryDirectory() as td:
        try:
            api.kernels_pull(slug, path=td, metadata=True, quiet=True)
        except Exception:
            continue
        meta_path = Path(td) / "kernel-metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        datasets = meta.get("dataset_sources", [])
        kernels = meta.get("kernel_sources", [])
        models = meta.get("model_sources", [])
        all_sources = datasets + kernels + models
        has_perch = any(
            ("perch" in s.lower()) or ("bird-vocalization-classifier" in s.lower())
            for s in all_sources
        )
        if not has_perch:
            non_perch.append({
                "slug": slug,
                "title": meta.get("title", "?"),
                "datasets": datasets,
                "kernels": kernels,
                "models": models,
                "enable_gpu": meta.get("enable_gpu"),
            })

print(f"\n=== Found {len(non_perch)} non-Perch BC2026 NBs (in top {len(ALL_KERNELS)}) ===\n")
for r in non_perch:
    print(f"slug:  {r['slug']}")
    print(f"title: {r['title'][:60]}")
    print(f"datasets ({len(r['datasets'])}): {r['datasets'][:4]}")
    if r['kernels']:
        print(f"kernel_sources: {r['kernels']}")
    if r['models']:
        print(f"model_sources:  {r['models']}")
    print()
