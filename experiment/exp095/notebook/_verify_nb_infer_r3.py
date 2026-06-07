"""Verify exp095 nb_infer_r3.ipynb: backbone + dataset + compile."""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path(__file__).with_name("nb_infer_r3.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")

print(f"NB: {NB.name}")
print(f"Cells: {len(nb['cells'])}")
print()

checks = [
    ('BACKBONE = "eca_nfnet_l0"',               'BACKBONE = "eca_nfnet_l0"' in all_src),
    ('NO eca_nfnet_l1',                         '"eca_nfnet_l1"' not in all_src),
    ('Dataset path: exp095-l0-r2spec',          'birdclef2026-exp095-l0-r2spec' in all_src),
    ('NO old dataset exp029-l1-single',         'birdclef2026-exp029-l1-single' not in all_src),
    ('onnxruntime wheel attached',              'romantamrazov/onnxruntime-1-24-4' in all_src),
    ('ckpt glob r3_fold*_ckpt_best_ns22.pth',   'r3_fold*_ckpt_best_ns22.pth' in all_src),
    ('USE_PERCH_DISTILL = True',                'USE_PERCH_DISTILL = True' in all_src),
    ('PERCH_EMBED_DIM = 1536',                  'PERCH_EMBED_DIM = 1536' in all_src),
    ('SR = 32000',                              'SR = 32000' in all_src),
    ('TRAIN_DURATION = 5',                      'TRAIN_DURATION = 5' in all_src),
    ('N_FFT = 2048',                            'N_FFT      = 2048' in all_src),
    ('HOP_LENGTH = 512',                        'HOP_LENGTH = 512' in all_src),
    ('N_MELS = 256',                            'N_MELS     = 256' in all_src),
    ('FMIN = 20',                               'FMIN       = 20' in all_src),
    ('FMAX = 16000',                            'FMAX       = 16000' in all_src),
    ('N_WINDOWS = 12',                          'N_WINDOWS = 12' in all_src),
    ('GAUSSIAN_KERNEL [0.1,0.2,0.4,0.2,0.1]',   '[0.1, 0.2, 0.4, 0.2, 0.1]' in all_src),
    ('ONNX export dynamo=False',                'dynamo=False' in all_src),
    ('CPU only inference',                      'CPUExecutionProvider' in all_src),
    ('submission.csv written',                  'submission.csv' in all_src),
]
print("=== Invariants ===")
for n, ok in checks:
    print(f"  {'[OK]' if ok else '[FAIL]'} {n}")

print("\n=== Compile check ===")
n_ok = n_err = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        compile(clean, f"<cell-{i}-{c.get('id','?')}>", "exec")
        n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  [FAIL] cell {i} id={c.get('id','?')}: {e}")
print(f"Result: {n_ok} OK, {n_err} ERRORS")
