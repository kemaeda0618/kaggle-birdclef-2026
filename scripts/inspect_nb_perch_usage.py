"""For each candidate kernel, pull metadata + .ipynb, check for Perch usage.

Output a table: slug | uses_perch | dataset_sources | model_sources
"""
import json, os, sys, time, tempfile, shutil
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Pick candidates likely to have high LB but potentially non-Perch
CANDIDATES = [
    # Karnakbayev tree (high LB):
    "karnakbaevarthur/gated-rank-fusion-pipeline",
    "karnakbaevarthur/power-optimization",
    "karnakbaevarthur/optimized-dual-architecture-ensemble",
    # LB 0.948 named:
    "youssefmo942009/lb-0-948",
    "safar1/lb-score-0-948",
    "mtoshidesu/test-0-948",
    # Other candidates:
    "pilkwang/birdclef-26-acoustic-time-window-rank-fusion",
    "meenalsinha/birdclef-2026-improved",
    "anthonytherrien/birdclef-2026-ensemble",
    "raunakdey07/birdclef-2026-v6",
    "mtoshidesu/birdclef-2026-visual-cpu-inference",
    "zeyadmohamadezzat/birdclef-2026-eos-parity-inference",
    "kijiang/birdclef2026-v337",
]

PERCH_KEYWORDS = [
    "perch",
    "Perch",
    "PERCH",
    "google/bird-vocalization-classifier",
    "perch_v2",
    "perch-v2",
    "perch-meta",
    "embedding_v2",
    "BirdVocalization",
]

print(f"Inspecting {len(CANDIDATES)} kernels for Perch usage...\n")

results = []
for slug in CANDIDATES:
    print(f"--- {slug} ---")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        try:
            api.kernels_pull(slug, path=str(td), metadata=True)
        except Exception as e:
            print(f"  ERROR: {str(e)[:200]}")
            continue

        meta_path = td / "kernel-metadata.json"
        if not meta_path.exists():
            print(f"  no metadata, skipping")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        title = meta.get("title", "?")
        dataset_sources = meta.get("dataset_sources", [])
        kernel_sources = meta.get("kernel_sources", [])
        model_sources = meta.get("model_sources", [])

        # Check NB content
        nbs = list(td.glob("*.ipynb"))
        nb_content = ""
        if nbs:
            nb_content = nbs[0].read_text(encoding="utf-8")

        uses_perch_meta = any(any(kw in s for kw in PERCH_KEYWORDS)
                              for s in dataset_sources + kernel_sources + model_sources)
        uses_perch_code = any(kw in nb_content for kw in PERCH_KEYWORDS)
        uses_perch = uses_perch_meta or uses_perch_code

        # Also check tf-wheels (often for Perch)
        uses_tf_wheels = any("tf-wheels" in s or "tf_wheels" in s for s in dataset_sources + kernel_sources)

        # Find Perch mentions in code (sample)
        sample_lines = []
        if uses_perch_code:
            for line in nb_content.split("\n"):
                if any(kw in line for kw in PERCH_KEYWORDS):
                    sample_lines.append(line.strip()[:120])
                    if len(sample_lines) >= 3:
                        break

        print(f"  title:           {title[:60]}")
        print(f"  uses_perch:      {uses_perch}  (meta={uses_perch_meta}, code={uses_perch_code})")
        print(f"  uses_tf_wheels:  {uses_tf_wheels}")
        if sample_lines:
            for sl in sample_lines:
                print(f"    > {sl}")
        print(f"  datasets ({len(dataset_sources)}): {dataset_sources[:4]}")
        print(f"  model_sources:   {model_sources}")
        print()

        results.append({
            "slug": slug,
            "title": title,
            "uses_perch": uses_perch,
            "uses_perch_meta": uses_perch_meta,
            "uses_perch_code": uses_perch_code,
            "uses_tf_wheels": uses_tf_wheels,
            "datasets": dataset_sources,
        })

print("\n=== SUMMARY: Non-Perch candidates ===")
non_perch = [r for r in results if not r["uses_perch"]]
print(f"Found {len(non_perch)} candidates without any Perch reference:")
for r in non_perch:
    print(f"  - {r['slug']}  {r['title'][:50]}")
    print(f"      datasets: {r['datasets'][:3]}")
