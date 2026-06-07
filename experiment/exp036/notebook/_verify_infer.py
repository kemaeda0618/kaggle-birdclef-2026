import json, ast
nb = json.load(open(r'C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp036\notebook\nb_infer_softauc.ipynb', encoding='utf-8'))
print(f'cells: {len(nb["cells"])}')
all_src = '\n'.join(''.join(c.get('source', [])) for c in nb['cells'])
checks = [
    ('exp036 references', 'exp036' in all_src),
    ('exp017 references gone', 'exp017' not in all_src or all_src.count('exp017') <= 2),  # may stay in comment
    ('softauc dataset', 'birdclef2026-exp036-softauc-weights' in all_src),
    ('old dataset gone', 'birdclef2026-exp017-weights' not in all_src),
    ('eca_nfnet_l0 backbone', 'eca_nfnet_l0' in all_src),
    ('ckpt_best_ns22', 'ckpt_best_ns22' in all_src),
]
for label, ok in checks:
    print(f'  [{"OK" if ok else "FAIL"}] {label}')

# syntax check
fail = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code': continue
    src = ''.join(c.get('source', []))
    try:
        ast.parse(src)
        print(f'  OK [{i}] {c.get("id",""):15s} {len(src):5d}c')
    except SyntaxError as e:
        print(f'  FAIL [{i}] {c.get("id",""):15s} {e.msg} line {e.lineno}')
        fail += 1
print(f'syntax: {fail} failures')
