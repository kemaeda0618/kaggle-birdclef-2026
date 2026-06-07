"""Extract TTA-related cells from candidate NBs."""
import json
from pathlib import Path

p = Path(r"C:\Users\maeke\.claude\projects\C--Users-maeke-work-kaggle-birdclef-2026\10b3c073-565f-4aa3-897d-438b8fd4776a\tool-results\mcp-kaggle-get_notebook_info-1779606284410.txt")
data = json.loads(p.read_text(encoding="utf-8"))
# Metadata first
meta = data.get("metadata", {})
print(f"=== {meta.get('title')} ===")
print(f"  author: {meta.get('author')}")
print(f"  ref: {meta.get('ref')}")
print(f"  votes: {meta.get('total_votes')}")
print(f"  enable_gpu: {meta.get('enable_gpu')}")
print(f"  dataset_sources: {meta.get('dataset_data_sources', [])[:5]}")
print(f"  competition: {meta.get('competition_data_sources')}")
print()

# Notebook content
nb = json.loads(data["blob"]["source"])
print(f"  cells: {len(nb['cells'])}")
print()

# Search for TTA-related cells
tta_keywords = ["TTA", "test_time", "test-time", "shift", "augment_test", "sliding"]
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if c.get("cell_type") != "code": continue
    for kw in tta_keywords:
        if kw.lower() in src.lower():
            cid = c.get("id", f"c{i}")
            print(f"--- Cell {i} (id={cid}, kw='{kw}') ---")
            # Print first 60 lines
            lines = src.split("\n")[:60]
            for ln in lines:
                print(f"  {ln}")
            print()
            break
