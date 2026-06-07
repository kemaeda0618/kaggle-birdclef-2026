import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
for p in sys.argv[1:]:
    print(f'=== {p} ===')
    nb = json.load(open(p, encoding='utf-8'))
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        head = src.split('\n')[0][:80] if src else '(empty)'
        print(f'  Cell {i:2d} ({c.get("cell_type"):8s}) {len(src):6d} chars | {head}')
