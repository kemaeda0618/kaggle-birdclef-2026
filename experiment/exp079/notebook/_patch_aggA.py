"""Patch exp079 NB: Aggressive A weights (exp029 dominant) + title update."""
import json
from pathlib import Path

NB_PATH = Path(__file__).with_name("nb_blend_agg_a.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# Update title (Cell 0)
title_cell = nb["cells"][0]
title_src = "".join(title_cell["source"])
title_new = title_src.replace(
    "# exp048: TopN N=1 PP (paper 256 spec, K=1, POW=1.0)",
    "# exp079: Aggressive A — exp029 dominant (0.15/0.30/0.55)"
).replace(
    "# exp040 = NB4 v11 + ResSSM (0.30) + Tucker SED (0.40) + exp029 (0.30)",
    "# exp079 = exp048 base + AGGRESSIVE blend weights (exp029 dominant)"
)
title_cell["source"] = title_new.splitlines(keepends=True)

# Patch blend weights (Cell 12)
patches = [
    {"old": 'BLEND_W_E10  = 0.30   # exp040 (inherited from exp038): NB4 0.35→0.30\\n',
     "new": 'BLEND_W_E10  = 0.15   # ★ exp079 Aggressive A: NB4 0.30→0.15 (down)\\n'},
    {"old": 'BLEND_W_SED  = 0.40   # Tucker public SED weight\\n',
     "new": 'BLEND_W_SED  = 0.30   # ★ exp079: Tucker 0.40→0.30 (down)\\n'},
    {"old": 'BLEND_W_E17  = 0.30   # exp040 (inherited from exp038): exp029 0.25→0.30 — note var name kept \\"E17\\" for code reuse\\n',
     "new": 'BLEND_W_E17  = 0.55   # ★★ exp079 Aggressive A: exp029 0.30→0.55 (up, ceiling test)\\n'},
]

total = 0
for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])
    new_src = src
    for p in patches:
        # Match by trying various escaping
        old_real = p["old"].replace("\\n", "\n").replace('\\"', '"')
        new_real = p["new"].replace("\\n", "\n").replace('\\"', '"')
        if old_real in new_src:
            new_src = new_src.replace(old_real, new_real, 1)
            total += 1
            print(f"  applied: {old_real[:50]}...")
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None

print(f"Total patches applied: {total}/{len(patches)}")
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {NB_PATH}")
