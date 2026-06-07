import json, ast
nb = json.loads(open(r'C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp040\notebook\nb_blend.ipynb', encoding='utf-8').read())
print(f'cells: {len(nb["cells"])}')
all_src = '\n'.join(''.join(c.get('source', [])) for c in nb['cells'])
checks = [
    ('ResidualSSM class', 'class ResidualSSM' in all_src),
    ('RESIDUAL_CORRECTION_WEIGHT', 'RESIDUAL_CORRECTION_WEIGHT = 0.30' in all_src),
    ('rs_model train', 'rs_model = ResidualSSM' in all_src),
    ('rs_corr apply', 'rs_corr = rs_model' in all_src),
    ('exp038 weight w30-40-30', 'BLEND_W_E10  = 0.30' in all_src),
    ('NB4 keep: KD', 'LAMBDA_KD' in all_src),
    ('NB4 keep: Retrieval', 'LAMBDA_RETRIEVAL' in all_src),
    ('NB4 keep: E19', 'E19_BETA' in all_src),
    ('NB4 keep: Season prior', 'SEASON_PRIORS' in all_src),
    ('NB4 keep: Mirror', 'MIRROR_PAIRS' in all_src),
    ('exp040 ref', 'exp040' in all_src),
]
for label, ok in checks:
    print(f'  [{"OK" if ok else "FAIL"}] {label}')

fail = 0
print('\n=== syntax ===')
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code': continue
    src = ''.join(c.get('source', []))
    try:
        ast.parse(src)
        print(f'  OK [{i:02d}] {c.get("id","")}')
    except SyntaxError as e:
        print(f'  FAIL [{i:02d}] {c.get("id","")} {e.msg} line {e.lineno}: {e.text}')
        fail += 1
print(f'\n{fail} failures')
