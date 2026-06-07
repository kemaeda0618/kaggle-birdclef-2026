import json
nb = json.load(open(r'C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp037\notebook\nb_protossm_resssm.ipynb', encoding='utf-8'))
print(f'cells: {len(nb["cells"])}')
all_src = '\n'.join(''.join(c.get('source', [])) for c in nb['cells'])
print()
print('=== patch verification ===')
checks = [
    ('W_PROTO=0.5', 'W_PROTO = 0.5' in all_src),
    ('W_MLP=0.5',   'W_MLP = 0.5' in all_src),
    ('FCS_POWER=0.05', 'FCS_POWER = 0.05' in all_src),
    ('PER_CLASS_TEMP', 'PER_CLASS_TEMP' in all_src),
    ('TEXTURE_TAXA', 'TEXTURE_TAXA' in all_src),
    ('ResidualSSM class', 'class ResidualSSM' in all_src),
    ('rs_model train', 'rs_model = ResidualSSM' in all_src),
    ('rs_corr apply', 'rs_corr = rs_model' in all_src),
    ('RESIDUAL_CORRECTION_WEIGHT', 'RESIDUAL_CORRECTION_WEIGHT' in all_src),
    ('adaptive_delta_smooth', 'adaptive_delta_smooth' in all_src),
    ('ADAPTIVE_BASE_ALPHA', 'ADAPTIVE_BASE_ALPHA = 0.20' in all_src),
    ('temperature divide', 'l_blend = l_blend / PER_CLASS_TEMP_T' in all_src),
    ('Mirror still present', 'MIRROR_PAIRS' in all_src),
    ('exp037 references', 'exp037' in all_src),
    ('exp028 cleared', 'exp028' not in all_src),
]
for label, ok in checks:
    print(f'  [{"OK" if ok else "FAIL"}] {label}')

print()
print('=== cell summary ===')
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    cid = c.get('id', '')
    print(f'  [{i:02d}] {c["cell_type"]:8s} id={cid:20s} len={len(src):5d}')
