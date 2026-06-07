"""Fix duplicate power kwarg in rank_aware_scaling call.

Current (BROKEN):
  probs = rank_aware_scaling(   probs, n_windows=N_WINDOWS, power=0.6,
                                 power=0.4)

Should be:
  probs = rank_aware_scaling(   probs, n_windows=N_WINDOWS,
                                 power=0.6)
"""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_exp090_tuned_tax.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

for c in nb["cells"]:
    if c.get("id") == "exp037_full_pipeline":
        src = "".join(c["source"])
        # The BROKEN current code (single string for replace)
        broken = (
            "probs = rank_aware_scaling(   probs, n_windows=N_WINDOWS, power=0.6,\n"
            "                               power=0.4)"
        )
        # The CORRECT replacement
        fixed = (
            "probs = rank_aware_scaling(   probs, n_windows=N_WINDOWS,\n"
            "                               power=0.6)"
        )
        if broken in src:
            new_src = src.replace(broken, fixed)
            c["source"] = new_src.splitlines(keepends=True)
            print("[OK] Fixed duplicate power kwarg in rank_aware_scaling")
        else:
            print("[WARN] broken pattern not found, dumping rank_aware lines:")
            for i, ln in enumerate(src.split("\n")):
                if "rank_aware" in ln:
                    print(f"  L{i}: {repr(ln)}")
                    for j in range(i+1, min(i+3, len(src.split('\n')))):
                        print(f"  L{j}: {repr(src.split(chr(10))[j])}")
        break

# Save
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nSaved: {NB_PATH}")

# Re-verify
import ast
print("\n=== Re-verify syntax with compile() (strict) ===")
n_ok, n_err = 0, 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        compile(clean, f"<cell-{i}>", "exec")
        n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  cell {i} (id={c.get('id','?')}): {e}")
print(f"Result: {n_ok} OK, {n_err} SyntaxErrors")

# Specific check of rank_aware_scaling call
for c in nb["cells"]:
    if c.get("id") == "exp037_full_pipeline":
        src = "".join(c["source"])
        lines = src.split("\n")
        for i, ln in enumerate(lines):
            if "rank_aware_scaling(" in ln and "def " not in ln:
                print(f"\nFixed rank_aware_scaling at L{i}:")
                for j in range(i, min(i+3, len(lines))):
                    print(f"  L{j:3d}: {lines[j][:140]}")
                # Count power kwargs in this call (next 3 lines)
                block = "\n".join(lines[i:i+3])
                n_power = block.count("power=")
                print(f"  power kwarg count in call: {n_power} (should be 1)")
        break
