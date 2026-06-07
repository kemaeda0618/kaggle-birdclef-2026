"""Add pip install onnxruntime to tucker NB (CPU Kaggle env doesn't have it preinstalled)."""
import json
from pathlib import Path

NB_PATH = Path(__file__).with_name("nb_pseudo_tucker.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# Find Cell 1 (Setup) and prepend pip install
cell1_src = "".join(nb["cells"][1]["source"])
new_src = "!pip install onnxruntime --quiet\n\n" + cell1_src
nb["cells"][1]["source"] = new_src.splitlines(keepends=True)
nb["cells"][1]["outputs"] = []
nb["cells"][1]["execution_count"] = None

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Added pip install onnxruntime to {NB_PATH.name}")
