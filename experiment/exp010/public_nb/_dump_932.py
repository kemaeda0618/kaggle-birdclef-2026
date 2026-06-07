import json, sys
path = r'C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp010\public_nb\_zushi\mtoshidesu_0-932-test-mod-version-4\0-932-test-mod-version-4.ipynb'
nb = json.load(open(path, encoding='utf-8'))
ids = [int(x) for x in sys.argv[1:]]
for i in ids:
    src = ''.join(nb['cells'][i].get('source', []))
    print(f"\n=== CELL {i} ===\n{src}")
