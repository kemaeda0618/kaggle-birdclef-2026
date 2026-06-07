import json, ast
nb = json.load(open(r'C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp039\notebook\nb_blend.ipynb', encoding='utf-8'))
print(f'cells: {len(nb["cells"])}\n')
print('=== Cell summary ===')
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    cid = c.get('id', '')
    first = src.split('\n')[0][:120] if src else '(empty)'
    print(f'  [{i:02d}] {c["cell_type"]:8s} id={cid:25s} {len(src):6d}c | {first}')

print('\n=== Syntax check ===')
fail = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c.get('source', []))
    if not src.strip():
        continue
    try:
        ast.parse(src)
        print(f'  OK [{i:02d}] {c.get("id",""):25s}')
    except SyntaxError as e:
        print(f'  FAIL [{i:02d}] {c.get("id",""):25s} {e.msg} line {e.lineno}: {e.text}')
        fail += 1
print(f'\n{fail} syntax failures')

# Quick scan for key bindings
print('\n=== Variable presence check ===')
all_src = '\n'.join(''.join(c.get('source', [])) for c in nb['cells'])
checks = [
    ('probs_exp037 saved', 'probs_exp037 = probs.copy()' in all_src),
    ('probs_exp010 set', 'probs_exp010 = probs_exp037.reshape' in all_src),
    ('audio_cache built', 'audio_cache = []' in all_src),
    ('all_row_ids set', 'all_row_ids = meta_te' in all_src),
    ('test_files set', 'test_files = [str(p)' in all_src),
    ('sed-infer present', 'probs_sed = []' in all_src),
    ('e29-infer present', 'E17_BACKBONE' in all_src),
    ('blend present', 'BLEND_W_E10' in all_src),
    ('NO submission.csv in exp037 cell', 'sub.to_csv("submission.csv", index=False)' not in all_src[:80000]),  # in first chunk (exp037 part)
]
for label, ok in checks:
    print(f'  [{"OK" if ok else "FAIL"}] {label}')
