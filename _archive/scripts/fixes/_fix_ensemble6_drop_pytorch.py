"""Emergency fix: drop PyTorch R1 models (exp083, exp084), use only 4 OV R2 models.
Rebalance weights to 1.0.

Speedup: 50-85 min → ~20-30 min (PyTorch model 2 つ削除)
"""
import json
from pathlib import Path

P = Path("experiment/ensemble6/notebook/nb_infer_ensemble6.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    if "MODELS_CFG = {" in src:
        # Replace MODELS_CFG with 4-model version
        new_src = src
        # Find start of exp083_r1 block
        idx_83 = new_src.find("    'exp083_r1':")
        idx_84_end = new_src.find("    },\n", new_src.find("    'exp084_r1':"))
        if idx_83 != -1 and idx_84_end != -1:
            # Remove exp083 + exp084 blocks
            new_src = new_src[:idx_83] + new_src[idx_84_end + 7:]
            # Rebalance weights for 4 remaining models
            # Old (6 model): 0.25+0.20+0.10+0.15+0.15+0.15 = 1.00
            # New (4 model): rebalance to 1.00
            #   exp017 R2: 0.30, exp081 R2: 0.25, exp082 R2: 0.15, exp015 R2: 0.30
            new_src = new_src.replace("'weight': 0.25,\n    },\n    'exp081_r2':",
                                       "'weight': 0.30,\n    },\n    'exp081_r2':")
            new_src = new_src.replace("'weight': 0.20,\n    },\n    'exp082_r2':",
                                       "'weight': 0.25,\n    },\n    'exp082_r2':")
            new_src = new_src.replace("'weight': 0.10,\n    },\n    'exp015_r2':",
                                       "'weight': 0.15,\n    },\n    'exp015_r2':")
            # exp015 R2 ends with weight - look for that
            new_src = new_src.replace("'weight': 0.15,\n    },\n}",
                                       "'weight': 0.30,\n    },\n}")
            c["source"] = new_src.splitlines(keepends=True)
            print("OK Dropped exp083_r1, exp084_r1 (PyTorch). Weights rebalanced.")
            print("  New weights: exp017=0.30, exp081=0.25, exp082=0.15, exp015=0.30 = 1.00")
        else:
            print(f"WARN: could not find exp083_r1/exp084_r1 blocks")
        break

P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# Verify
nb = json.loads(P.read_text(encoding="utf-8"))
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    if "MODELS_CFG = {" in src:
        # Count models
        n_models = sum(1 for ln in src.splitlines() if ln.strip().startswith("'exp") and ln.strip().endswith(":"))
        # Weight sum check
        import re
        weights = [float(m) for m in re.findall(r"'weight':\s*([\d.]+)", src)]
        print(f"\nModel count: {n_models}")
        print(f"Weights: {weights}")
        print(f"Weight sum: {sum(weights):.3f}")
        # Show models
        for ln in src.splitlines():
            if ln.strip().startswith("'exp") and ln.strip().endswith(":"):
                print(f"  {ln.strip()[:40]}")
        break

# Syntax check
import ast
n_ok, n_err = 0, 0
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                      for ln in src.splitlines())
    try:
        ast.parse(clean); n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  SyntaxError: {e}")
print(f"\nSyntax: {n_ok} OK, {n_err} ERRORS")
