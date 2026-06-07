"""Dump specific Babych NB cells: model arch, inference dataset, configs, gauss_convolve."""
import json, sys, io, os
if os.name == "nt":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

p = r"experiment/exp010/notebook/_reference_babych/birdclef2025-1st-place-inference/birdclef2025-1st-place-inference.ipynb"
nb = json.load(open(p, encoding="utf-8"))

import sys as _sys
which = _sys.argv[1] if len(_sys.argv) > 1 else "all"
TARGETS = {"all": [5,7,16], "5": [5], "7": [7], "9": [9], "16": [16], "14": [14]}
for i in TARGETS.get(which, [int(x) for x in which.split(',')]):
    src = "".join(nb['cells'][i].get('source', []))
    print(f"### Cell {i} ({len(src)} chars) ###")
    print(src)
    print()
