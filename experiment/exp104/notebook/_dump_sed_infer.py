"""Dump sed-infer cell of exp090 NB."""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path("experiment/exp090/notebook/nb_exp090_tuned_tax.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

for c in nb["cells"]:
    if c.get("id") == "sed-infer":
        s = c["source"]
        src = "".join(s) if isinstance(s, list) else s
        out = Path("experiment/exp104/notebook/_sed_infer_dump.txt")
        out.write_text(src, encoding="utf-8")
        print(f"Dumped {len(src)} chars to {out}")
        break
