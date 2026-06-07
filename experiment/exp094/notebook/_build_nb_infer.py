"""Build exp094 inference NB by cloning exp017 nb_infer.ipynb.

Changes:
1. Header: exp094 R1-Ali context
2. STATE_DIR ckpt search: look in maekeso/birdclef2026-exp094-ali-r1 dataset
3. Title/slug naming: exp094 namespace

Prerequisite (user action):
- Upload R1-Ali ckpt from Drive to Kaggle Dataset:
  /content/drive/MyDrive/kaggle/birdclef2026/output/exp094/r1/ckpt_best_ns22.pth
  → maekeso/birdclef2026-exp094-ali-r1
"""
import json, io, sys, shutil
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp017\notebook\nb_infer.ipynb")
DST = Path(__file__).with_name("nb_infer_r1_ali.ipynb")

print(f"[clone] {SRC.name} -> {DST.name}")
shutil.copy(SRC, DST)

nb = json.loads(DST.read_text(encoding="utf-8"))

# Insert header markdown
HEADER_MD = """# exp094 — R1-Ali inference (standalone LB check)

R1-Ali (eca_nfnet_l0 + Tucker MSE + Ali Hinton KL, T=1, ALPHA_LOGIT=0.1) を Kaggle CPU で推論。
val_ns22=0.9165 (R1 baseline ~0.91 比 +0.005)。LB 想定 0.88-0.90。

## Input dataset
- `maekeso/birdclef2026-exp094-ali-r1` (R1-Ali ckpt、Drive から upload 想定)

## Output
- submission.csv (R1-Ali standalone 予測)
"""

nb["cells"].insert(0, {
    "cell_type": "markdown",
    "id": "hdr_exp094_infer",
    "metadata": {},
    "source": HEADER_MD.splitlines(keepends=True),
})
print("[OK] Header inserted")

# Replace exp017 references with exp094
patches = [
    # ckpt path search: exp017 -> exp094
    ('"r2_ckpt_best_ns22.pth",   # R2 best (TBD val)',
     '"r1_ali_ckpt_best_ns22.pth",  # exp094 R1-Ali best (preferred name)'),
    ('"r2_ckpt_best_macro.pth",',
     '"r1_ali_ckpt_best_macro.pth",'),
    ('"ckpt_best_ns22.pth",      # R1 v2 fallback (val 0.9166)',
     '"ckpt_best_ns22.pth",      # ★ exp094 R1-Ali fallback (val 0.9165)'),
]

for old, new in patches:
    found = False
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if old in src:
            new_src = src.replace(old, new)
            c["source"] = new_src.splitlines(keepends=True)
            found = True
            print(f"  [OK] {old[:60]!r}")
            break
    if not found:
        print(f"  [MISS] {old[:60]!r}")

DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nSaved: {DST}")

# Compile check
print("\n=== Compile check ===")
n_ok = n_err = 0
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
        print(f"  [FAIL] cell {i}: {e}")
print(f"Result: {n_ok} OK, {n_err} ERRORS")
