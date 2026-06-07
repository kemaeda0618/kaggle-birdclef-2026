"""Find BirdSEDModel forward in training NB."""
import json
from pathlib import Path

nb = json.load(open(Path(__file__).with_name("nb_train_ali.ipynb"), encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    if "class BirdSEDModel" in src:
        # Show whole class
        start = src.find("class BirdSEDModel")
        end = src.find("\nclass ", start + 1)
        if end < 0:
            end = len(src)
        print(f"=== Cell {i} BirdSEDModel ===")
        print(src[start:end])
        print()
        break
