"""Dump e29-infer cell of exp090 NB."""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path(__file__).parent / "nb_exp090_tuned_tax.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))

for c in nb["cells"]:
    if c.get("id") == "e29-infer":
        s = "".join(c["source"])
        out = Path(__file__).parent / "_e29_infer_dump.txt"
        out.write_text(s, encoding="utf-8")
        print(f"Dumped {len(s)} chars to {out}")
        break
