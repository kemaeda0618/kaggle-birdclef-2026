"""Re-verify the 2 missed items by checking source strings (not JSON-encoded)."""
import json, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_exp085_4model.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

all_src = "\n\n".join("".join(c.get("source", [])) for c in nb["cells"])

checks = [
    ("r2_ov literal in source", 'r2_ov' in all_src),
    ('"r2_ov" path expression', '"r2_ov"' in all_src or "'r2_ov'" in all_src),
    ('submission.to_csv("submission.csv")', 'submission.to_csv("submission.csv"' in all_src),
]
for name, ok in checks:
    print(f"  {'[OK]' if ok else '[MISS]'} {name}")

# Also count cells with key markers
e015 = next(c for c in nb["cells"] if c.get("id") == "e015-infer")
print(f"\ne015-infer cell length: {len(''.join(e015['source']))} chars")
