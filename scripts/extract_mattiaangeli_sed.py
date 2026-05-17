"""Extract Tucker SED inference cells from mattiaangeli's 0.943 NB."""
import json
from pathlib import Path

nb_path = Path("experiment/exp010/notebook/_reference_mattiaangeli/birdclef-2026-0-943-better-blend.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

KEYS = ("sed_fold", "distilled-sed", "InferenceSession", "sed_session", "sed_model",
        "bc2026-distilled-sed", "_distill", "ort.")
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if any(k in src for k in KEYS):
        ct = c["cell_type"]
        print(f"=== Cell {i} ({ct}) ===")
        print(src[:5000])
        print()
