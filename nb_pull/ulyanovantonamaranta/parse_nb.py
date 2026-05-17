import json, sys

path = r'C:\Users\maeke\work\kaggle\birdclef-2026\nb_pull\ulyanovantonamaranta\birdclef-2026-gate-fake008.ipynb'
out_path = r'C:\Users\maeke\work\kaggle\birdclef-2026\nb_pull\ulyanovantonamaranta\nb_cells.txt'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

lines = []
lines.append(f"Total cells: {len(nb['cells'])}\n")
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    lines.append(f"\n{'='*60}")
    lines.append(f"Cell {i} ({cell['cell_type']}) len={len(src)}")
    lines.append('='*60)
    lines.append(src)

with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("Done. Written to", out_path)
print("Total cells:", len(nb['cells']))
