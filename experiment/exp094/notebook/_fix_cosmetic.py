"""Fix cosmetic issues in inference NB.

3 issues to fix:
1. Cell 1 markdown: exp017 → exp094 context
2. Cell 3 STATE_DIR candidates: exp017 dataset → exp094-ali-r1
3. CKPT_PRIORITY: clean non-existent r1_ali_* prefix entries
"""
import json
from pathlib import Path

NB = Path(__file__).with_name("nb_infer_r1_ali.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

# === Patch 1: Replace exp017 inference header with exp094 context ===
OLD_HDR = '''# exp017 — Inference (Kaggle CPU 90min, internet=off)

Load best eca_nfnet_l0 SED ckpt from Kaggle Dataset, run inference on test_soundscapes,
write submission.csv.

## Constraints
- Kaggle Code-Only competition: CPU only, no internet, 90 min total
- Submit format: `row_id` + 234 species columns

## Inputs (attach when pushing)
- `birdclef-2026` (competition data; test_soundscapes is auto-mounted at score time)
- `maekeso/birdclef2026-exp017-weights` (R1 v2 / R2 ckpts; uploaded from train NB Cell 17)

## Pipeline
1. Locate ckpt under /kaggle/input
2. Rebuild model architecture (identical to training)
3. Per test file: decode -> 12 chunks (5s each) -> mel -> model -> sigmoid
4. Blend in **logit space**: 0.5*clip + 0.5*frame_max, then Gaussian smooth, then sigmoid
5. Write submission.csv'''

NEW_HDR = '''# exp094 R1-Ali Inference Pipeline (Kaggle CPU 90min, internet=off)

Load R1-Ali ckpt (eca_nfnet_l0 SED + Tucker MSE + Ali Hinton KL distillation),
run inference on test_soundscapes, write submission.csv.

## Constraints
- Kaggle Code-Only competition: CPU only, no internet, 90 min total
- Submit format: `row_id` + 234 species columns

## Inputs (attach when pushing)
- `birdclef-2026` (competition data; test_soundscapes is auto-mounted at score time)
- `maekeso/birdclef2026-exp094-ali-r1` (R1-Ali ckpt, uploaded from Drive)

## Pipeline
1. Locate ckpt under /kaggle/input
2. Rebuild model architecture (identical to training: Tucker MSE emb + Ali KL logit at T=1, ALPHA=0.1)
3. Per test file: decode -> 12 chunks (5s each) -> mel -> per-sample normalize -> model
4. Blend in **logit space**: 0.5*clip + 0.5*frame_max, then Gaussian smooth, then sigmoid
5. Write submission.csv'''

# === Patch 2: STATE_DIR candidates (exp017 → exp094-ali-r1) ===
OLD_CANDIDATES = '''CANDIDATES = [
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp017-weights"),
    Path("/kaggle/input/birdclef2026-exp017-weights"),
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp017-train"),
]'''

NEW_CANDIDATES = '''CANDIDATES = [
    # ★ exp094 R1-Ali ckpt dataset (primary)
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp094-ali-r1"),
    Path("/kaggle/input/birdclef2026-exp094-ali-r1"),
    # Fallback: legacy exp017 weights (in case re-runs against old datasets)
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp017-weights"),
    Path("/kaggle/input/birdclef2026-exp017-weights"),
]'''

# === Patch 3: STATE_DIR assertion message + CKPT_PRIORITY ===
OLD_ASSERT = '''assert STATE_DIR is not None, (
    "ckpt not found. Attach maekeso/birdclef2026-exp017-weights as dataset_sources"
)'''

NEW_ASSERT = '''assert STATE_DIR is not None, (
    "ckpt not found. Attach maekeso/birdclef2026-exp094-ali-r1 as dataset_sources"
)'''

OLD_PRIORITY = '''CKPT_PRIORITY = [
    "r1_ali_ckpt_best_ns22.pth",  # exp094 R1-Ali best (preferred name)
    "r1_ali_ckpt_best_macro.pth",
    "r2_ckpt_latest.pth",
    "ckpt_best_ns22.pth",      # ★ exp094 R1-Ali fallback (val 0.9165)
    "ckpt_best_macro.pth",
    "ckpt_latest.pth",
]'''

NEW_PRIORITY = '''CKPT_PRIORITY = [
    "ckpt_best_ns22.pth",   # ★ exp094 R1-Ali best ns22 (val 0.9165)
    "ckpt_best_macro.pth",  # exp094 R1-Ali best macro (fallback)
    "ckpt_latest.pth",      # latest epoch (fallback)
]'''

patches = [
    (OLD_HDR, NEW_HDR),
    (OLD_CANDIDATES, NEW_CANDIDATES),
    (OLD_ASSERT, NEW_ASSERT),
    (OLD_PRIORITY, NEW_PRIORITY),
]

print(f"Applying {len(patches)} cosmetic patches...")
for old, new in patches:
    found = False
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if old in src:
            new_src = src.replace(old, new)
            c["source"] = new_src.splitlines(keepends=True)
            found = True
            print(f"  [OK] {old[:60].splitlines()[0]!r}")
            break
    if not found:
        print(f"  [MISS] {old[:60].splitlines()[0]!r}")

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nSaved: {NB}")

# Verify
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
print("\n=== Verify ===")
checks = [
    ("Cell 1 has exp094 R1-Ali header", "exp094 R1-Ali Inference Pipeline" in all_src),
    ("exp017 inference reference removed from Cell 1", "exp017 — Inference" not in all_src),
    ("exp094-ali-r1 dataset path", "birdclef2026-exp094-ali-r1" in all_src),
    ("exp017 fallback retained (intentional)", "birdclef2026-exp017-weights" in all_src),
    ("Assertion mentions exp094", "Attach maekeso/birdclef2026-exp094-ali-r1" in all_src),
    ("CKPT_PRIORITY simplified (3 entries)", '"ckpt_best_ns22.pth",   # ★ exp094 R1-Ali' in all_src),
    ("non-existent r1_ali_* entries removed", "r1_ali_ckpt_best_ns22.pth" not in all_src),
    # Sanity (logic intact)
    ("BACKBONE eca_nfnet_l0", 'BACKBONE = "eca_nfnet_l0"' in all_src),
    ("Mel params intact (N_FFT)", "N_FFT      = 2048" in all_src),
    ("Per-sample normalization intact", "mel[i].mean()" in all_src and "mel[i].std()" in all_src),
    ("Half-Babych blend intact", "0.5 * clip_logits + 0.5 * frame_max" in all_src),
    ("Gaussian kernel intact", "GAUSSIAN_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1])" in all_src),
]
all_ok = True
for name, ok in checks:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")
    if not ok: all_ok = False

# Compile
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
