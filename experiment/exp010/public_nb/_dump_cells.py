import json, sys
path = sys.argv[1]
cells_idx = [int(x) for x in sys.argv[2:]]
nb = json.load(open(path, encoding='utf-8'))
for i in cells_idx:
    src = ''.join(nb['cells'][i].get('source', []))
    print(f"\n=== CELL {i} ({nb['cells'][i]['cell_type']}, {len(src)} chars) ===\n{src}")
