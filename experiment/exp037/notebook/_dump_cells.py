import json, sys
nb = json.load(open(r'C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp037\notebook\nb_protossm_resssm.ipynb', encoding='utf-8'))
ids = [int(x) for x in sys.argv[1:]]
for i in ids:
    src = ''.join(nb['cells'][i].get('source', []))
    print(f"\n=== CELL {i} ({nb['cells'][i].get('id','')}, {len(src)}c) ===\n{src}\n")
