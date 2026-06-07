import json, sys
nb = json.loads(open(r'C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\mtoshidesu_v313904345\0-932-v313904345.ipynb', encoding='utf-8').read())
ids = [int(x) for x in sys.argv[1:]]
for i in ids:
    src = ''.join(nb['cells'][i].get('source', []))
    print(f"\n=== v313904345 CELL {i} ({len(src)}c) ===\n{src}\n")
