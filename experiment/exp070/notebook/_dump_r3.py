import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
nb = json.load(open('experiment/exp029/notebook/nb_train_r3_l1_single.ipynb', encoding='utf-8'))
for i in [7, 8, 9, 10, 11, 12]:
    c = nb['cells'][i]
    src = ''.join(c.get('source', []))
    print(f'=== Cell {i} ===')
    print(src)
    print()
