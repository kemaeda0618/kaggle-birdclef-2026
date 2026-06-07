"""Dump train_r2 epoch loop from exp102 NB for line-by-line review."""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path(__file__).with_name("nb_train_r3.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

# Find the train_r2 cell (contains 'def train_r2(')
target_cell = None
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if "def train_r2(" in src:
        target_cell = c
        break

assert target_cell is not None, "train_r2 cell not found"
src = "".join(target_cell["source"])

# Dump to file
out = Path(__file__).parent / "_train_r2_dump.txt"
out.write_text(src, encoding="utf-8")
print(f"Dumped {len(src)} chars to {out}")

# Also dump config (Cell with USE_EMA / WRS_WARMUP_EPOCHS)
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if "USE_EMA = True" in src and "EARLY_STOP_PATIENCE" in src:
        out2 = Path(__file__).parent / "_config_dump.txt"
        out2.write_text(src, encoding="utf-8")
        print(f"Dumped config to {out2}")
        break
