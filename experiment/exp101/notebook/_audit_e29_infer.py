"""Audit exp101 e29-infer cell line by line vs exp090 (= exp029 R3 source)."""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Read both
exp090 = json.loads(Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp090\notebook\nb_exp090_tuned_tax.ipynb").read_text(encoding="utf-8"))
exp101 = json.loads(Path(__file__).with_name("nb_exp101_e100_swap.ipynb").read_text(encoding="utf-8"))

src90 = next("".join(c["source"]) for c in exp090["cells"] if c.get("id") == "e29-infer")
src01 = next("".join(c["source"]) for c in exp101["cells"] if c.get("id") == "e29-infer")

l90 = src90.splitlines()
l01 = src01.splitlines()

print(f"exp090 e29-infer: {len(l90)} lines")
print(f"exp101 e29-infer: {len(l01)} lines")

print("\n=== Diff per line ===")
import difflib
n_diff = 0
for line in difflib.unified_diff(l90, l01, lineterm="", n=1):
    if line.startswith("---") or line.startswith("+++"): continue
    if line.startswith("@@"):
        print(f"\n{line}")
        continue
    n_diff += 1
    print(line)

print(f"\nTotal diff lines: {n_diff}")
