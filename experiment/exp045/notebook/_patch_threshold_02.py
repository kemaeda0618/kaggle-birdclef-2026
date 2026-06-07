"""Update CHUNK_FILTER_MAX_PROB from 0.5 to 0.2 based on Phase 0 analysis."""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp045\notebook\nb_train_r3_l1_filtered.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

for c in nb["cells"]:
    if c.get("id") == "config":
        src = "".join(c["source"])
        new_src = src.replace(
            "CHUNK_FILTER_MAX_PROB = 0.5",
            "CHUNK_FILTER_MAX_PROB = 0.2",
        )
        new_src = new_src.replace(
            "CHUNK_FILTER_MAX_PROB = 0.5 で paper 256 spec (Sydorskyi 2nd place の primary_label_min_prob)",
            "CHUNK_FILTER_MAX_PROB = 0.2 で BC2026 データの bimodal 谷 cutoff (Phase 0 分析推奨)\n"
            "# (paper 256 は 0.5 だが、我々データは class coverage 47% 落ちで rare class 学習失う)",
        )
        c["source"] = new_src.splitlines(keepends=True)
        print("[config] OK threshold 0.5 -> 0.2")
        break

# Update hdr to reflect threshold change
for c in nb["cells"]:
    if c.get("id") == "hdr":
        src = "".join(c["source"])
        new_src = src.replace(
            "`CHUNK_FILTER_MAX_PROB = 0.5` を pseudo CSV 読込直後に適用",
            "`CHUNK_FILTER_MAX_PROB = 0.2` を pseudo CSV 読込直後に適用 (BC2026 bimodal 谷、Phase 0 分析推奨)",
        )
        new_src = new_src.replace(
            "paper 256 流 chunk filter (max_prob > 0.5 で chunk drop) の効果を実測。",
            "chunk filter (max_prob > 0.2 で chunk drop) の効果を実測。\n"
            "paper 256 spec は 0.5 だが、BC2026 pseudo の bimodal 分布 (Phase 0 分析) で 0.5 は class coverage 47% 落ち、0.2 が natural cutoff。",
        )
        c["source"] = new_src.splitlines(keepends=True)
        print("[hdr] OK threshold annotation updated")
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nSaved {NB}")
