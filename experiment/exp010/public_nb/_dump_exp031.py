import json, sys
path = r'C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp031\notebook\nb_blend.ipynb'
nb = json.load(open(path, encoding='utf-8'))
ids = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else None
if ids:
    for i in ids:
        src = ''.join(nb['cells'][i].get('source', []))
        print(f"\n=== CELL {i} ({nb['cells'][i]['cell_type']}, {len(src)}c) ===\n{src}")
else:
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        first = src.split('\n')[0][:130] if src else '(empty)'
        print(f"  [{i:02d}] {c['cell_type']:8s} {len(src):5d}c | {first}")
