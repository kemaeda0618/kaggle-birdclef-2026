import json, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ.setdefault("PYTHONUTF8", "1")

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("path")
args = ap.parse_args()

nb = json.load(open(args.path, encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    head = src.split("\n")[0][:80] if src else "(empty)"
    print(f"Cell {i} ({c.get('cell_type'):8s}) {len(src):6d} chars | {head}")
