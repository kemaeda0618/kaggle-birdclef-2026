import json
paths = [
    ('hideyukizushi', r'C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\hideyukizushi_resssm\bird26-reproduce-perch-protossm-resssm-inf-train.ipynb'),
    ('marynaborovska', r'C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\marynaborovska_twopass\birdclef-26-two-pass-ssm-advanced-pp.ipynb'),
]
for name, path in paths:
    nb = json.load(open(path, encoding='utf-8'))
    print(f"\n=== {name}: cells={len(nb['cells'])} ===")
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        first = src.split('\n')[0][:120] if src else '(empty)'
        print(f"  [{i:02d}] {c['cell_type']:8s} {len(src):5d}c | {first}")
