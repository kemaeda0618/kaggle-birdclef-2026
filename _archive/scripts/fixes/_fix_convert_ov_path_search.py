"""Fix ckpt path search in convert-to-OV NB:
1. Add rglob fallback (search whole dataset tree)
2. Print directory contents for debugging if not found
"""
import json
from pathlib import Path

P = Path("experiment/convert_to_ov/notebook/convert_to_ov.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

# Update conversion cells (cells 4-7, indexed by model ID in source)
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    # Find conversion cells that have the assert pattern
    if "assert ckpt_path is not None" in src and "ckpt_candidates" in src:
        # Replace path search block with robust version
        old = """# Find ckpt
ckpt_path = None
for cand in ckpt_candidates:
    p = Path('/kaggle/input') / dataset_slug / cand
    if p.exists():
        ckpt_path = p; break
assert ckpt_path is not None, f'ckpt not found in {dataset_slug}'"""

        new = """# Find ckpt (with rglob fallback for unknown structure)
dataset_root = Path('/kaggle/input') / dataset_slug
ckpt_path = None
# Try direct paths first
for cand in ckpt_candidates:
    p = dataset_root / cand
    if p.exists():
        ckpt_path = p; break

# Fallback: rglob search (whole dataset tree)
if ckpt_path is None:
    target_name = Path(ckpt_candidates[0]).name
    print(f'  Direct paths not found, rglob fallback for {target_name}...')
    for f in dataset_root.rglob(target_name):
        ckpt_path = f
        print(f'    Found via rglob: {f}')
        break

# Debug: list dataset contents if still not found
if ckpt_path is None:
    print(f'  [DEBUG] {dataset_root} contents:')
    for f in sorted(dataset_root.rglob('*'))[:30]:
        if f.is_file():
            print(f'    {f.relative_to(dataset_root)} ({f.stat().st_size/1e6:.1f}MB)')

assert ckpt_path is not None, f'ckpt not found in {dataset_slug}'"""

        if old in src:
            src = src.replace(old, new)
            c["source"] = src.splitlines(keepends=True)
            print(f"  Fixed conversion cell: dataset_slug pattern")
        else:
            print(f"  WARN: old pattern not found in cell, checking...")
            if "for cand in ckpt_candidates" in src:
                print(f"    Cell has expected pattern but different whitespace")

P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# Syntax check
import ast
nb = json.loads(P.read_text(encoding="utf-8"))
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
