"""Patch the per-epoch log print to include per-taxon and per-class distribution stats."""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp045\notebook\nb_train_r3_l1_filtered.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

for c in nb["cells"]:
    if c.get("id") == "train_r2":
        src = "".join(c["source"])
        anchor = 'f"({ep_elapsed:.0f}s, session {total_elapsed/60:.1f}min) ===\\n")'
        if anchor not in src:
            print(f"WARN: anchor not found")
        else:
            # Insert per-taxon + per-class log AFTER the closing of the multi-line print
            new_block = anchor + '''

        # ★ exp045: per-taxon + per-class distribution log (filter ablation diagnostic)
        if "per_taxon" in r:
            taxon_str = " ".join(f"{t}={a:.3f}" for t, a in r["per_taxon"].items())
            print(f"      taxon: {taxon_str}")
        if "per_class_dist" in r:
            d = r["per_class_dist"]
            if d.get("n_valid", 0) > 0:
                print(f"      class: n={d['n_valid']} median={d['median']:.3f} "
                      f"p25={d['p25']:.3f} p75={d['p75']:.3f} "
                      f"#>0.5={d['n_above_0.5']} #>0.7={d['n_above_0.7']} "
                      f"#>0.9={d['n_above_0.9']} #perfect={d['n_perfect_1.0']}")
'''
            src = src.replace(anchor, new_block, 1)
            c["source"] = src.splitlines(keepends=True)
            print("[train_r2] OK per-epoch log augmented")
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB}")
