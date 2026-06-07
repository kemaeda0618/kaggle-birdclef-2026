import json
path = r'C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\mtoshidesu_0-932-test-mod-version-4\0-932-test-mod-version-4.ipynb'
nb = json.load(open(path, encoding='utf-8'))
print(f'cells={len(nb["cells"])}')
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    first = src.split('\n')[0][:130] if src else '(empty)'
    print(f'  [{i:02d}] {c["cell_type"]:8s} {len(src):5d}c | {first}')
