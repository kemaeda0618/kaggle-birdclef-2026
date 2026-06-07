"""Dump cell metadata + key code sections from Babych inference NB."""
import json, sys, io, os
if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

p = r"experiment/exp010/notebook/_reference_babych/birdclef2025-1st-place-inference/birdclef2025-1st-place-inference.ipynb"
nb = json.load(open(p, encoding="utf-8"))
print(f"=== Total {len(nb['cells'])} cells ===\n")
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    head = src.split("\n")[0][:80] if src else "(empty)"
    print(f"Cell {i} ({c.get('cell_type'):8s}) {len(src):6d} chars | {head}")
