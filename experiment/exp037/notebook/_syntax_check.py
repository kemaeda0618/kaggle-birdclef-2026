"""Compile each cell's source as Python to catch syntax errors before pushing."""
import json, ast, sys
nb = json.load(open(r'C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp037\notebook\nb_protossm_resssm.ipynb', encoding='utf-8'))
fail = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c.get('source', []))
    try:
        ast.parse(src, filename=f'cell_{i}')
        print(f'  OK [{i}] {c.get("id",""):20s} {len(src):6d}c')
    except SyntaxError as e:
        print(f'  FAIL [{i}] {c.get("id",""):20s} {e.msg} at line {e.lineno}')
        print(f'       offending: {e.text}')
        fail += 1
print(f'\n{fail} syntax failures out of {sum(1 for c in nb["cells"] if c["cell_type"]=="code")} code cells')
sys.exit(fail)
