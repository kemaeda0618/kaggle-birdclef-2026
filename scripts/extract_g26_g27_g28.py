"""Extract G26/G27/G28 implementations from 0.946 NB."""
import json
from pathlib import Path

nb_path = Path("experiment/exp010/notebook/_reference_0946/0-946-replay-with-robust-inputs.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

KEYS_BY_TOPIC = {
    "G26 isotonic / threshold": ["IsotonicRegression", "isotonic_regression"],
    "G27 cross-branch gate": ["fake_only", "proto_cont", "sed_only"],
    "G28 student-t kernel": ["t_kernel", "tdist", "Student-t", "df=1.5"],
    "Sonotype mirroring": ["MIRROR_PAIRS", "sonotype_mirror"],
    "Rare suppression": ["rare_taxon", "amph_mam", "rare suppress"],
}

for topic, keys in KEYS_BY_TOPIC.items():
    print(f"\n{'='*60}\n{topic}\n{'='*60}")
    found = []
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if any(k in src for k in keys):
            ct = c["cell_type"]
            found.append((i, ct, src))
    print(f"  matches: {len(found)} cells")
    for i, ct, src in found[:3]:
        print(f"\n--- Cell {i} ({ct}) [first 4000 chars] ---")
        print(src[:4000])
        print()

# Also list all cell starts for orientation
print(f"\n{'='*60}\nCELL OVERVIEW (first lines)\n{'='*60}")
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    first_line = src.split("\n")[0][:120]
    ct = c["cell_type"]
    print(f"  [{i:3d} {ct[:4]}] {first_line}")
