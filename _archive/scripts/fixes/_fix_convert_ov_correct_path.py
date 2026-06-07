"""Fix dataset path to use /kaggle/input/datasets/maekeso/<slug>/.
Kaggle Dataset attach mount path is /kaggle/input/datasets/<owner>/<slug>/.
"""
import json
from pathlib import Path

P = Path("experiment/convert_to_ov/notebook/convert_to_ov.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    if "dataset_root = Path('/kaggle/input') / dataset_slug" in src:
        # Replace path construction with correct one
        old = """dataset_root = Path('/kaggle/input') / dataset_slug
ckpt_path = None
# Try direct paths first
for cand in ckpt_candidates:
    p = dataset_root / cand
    if p.exists():
        ckpt_path = p; break"""
        new = """# ★ Kaggle Dataset mount: /kaggle/input/datasets/<owner>/<slug>/
DATASET_ROOTS = [
    Path('/kaggle/input/datasets/maekeso') / dataset_slug,  # ★ correct mount
    Path('/kaggle/input') / dataset_slug,                    # fallback (古い)
]
dataset_root = None
for p in DATASET_ROOTS:
    if p.exists():
        dataset_root = p
        print(f'  dataset_root: {p}')
        break
assert dataset_root is not None, f'Dataset not mounted: tried {DATASET_ROOTS}'

ckpt_path = None
# Try direct paths first
for cand in ckpt_candidates:
    p = dataset_root / cand
    if p.exists():
        ckpt_path = p; break"""
        if old in src:
            src = src.replace(old, new)
            c["source"] = src.splitlines(keepends=True)
            print(f"  Fixed conversion cell")

    # Also fix upload helper
    if "existing = Path(f'/kaggle/input/{dataset_slug}')" in src:
        old2 = "existing = Path(f'/kaggle/input/{dataset_slug}')"
        new2 = """# ★ Kaggle Dataset mount: try datasets/maekeso/<slug> first
        existing = None
        for cand_existing in [
            Path(f'/kaggle/input/datasets/maekeso/{dataset_slug}'),
            Path(f'/kaggle/input/{dataset_slug}'),
        ]:
            if cand_existing.exists():
                existing = cand_existing
                break
        if existing is None:
            existing = Path(f'/kaggle/input/datasets/maekeso/{dataset_slug}')  # for log only"""
        if old2 in src:
            src = src.replace(old2, new2)
            c["source"] = src.splitlines(keepends=True)
            print(f"  Fixed upload helper path")

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
