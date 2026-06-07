"""Verify exp113 (INT8) vs exp106-standalone (FP32): only IR path should differ."""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

fp32 = json.loads(Path("experiment/exp106/notebook/nb_exp106_standalone_3fold.ipynb").read_text(encoding="utf-8"))
int8 = json.loads(Path("experiment/exp113/notebook/nb_exp113_e106_int8_standalone.ipynb").read_text(encoding="utf-8"))


def cells(nb):
    return {c.get("id"): ("".join(c["source"]) if isinstance(c["source"], list) else c["source"]) for c in nb["cells"]}


fc, ic = cells(fp32), cells(int8)
allids = list(dict.fromkeys(list(fc) + list(ic)))

print("=== cell diff (exp106-standalone FP32 vs exp113 INT8) ===")
for cid in allids:
    a, b = fc.get(cid, ""), ic.get(cid, "")
    tag = "SAME" if a == b else "DIFF"
    print(f"  {cid:16s}: {tag}")

# show the DIFF cells in full
print("\n=== DIFF cell contents ===")
for cid in allids:
    a, b = fc.get(cid, ""), ic.get(cid, "")
    if a != b:
        print(f"\n----- {cid} : FP32 -----")
        print(a)
        print(f"\n----- {cid} : INT8 -----")
        print(b)
