"""Extract konbu17 head usage cells."""
import json
from pathlib import Path

nb_path = Path("experiment/exp010/notebook/_reference_konbu17/bird26-wliilamsam-0943-with-train-audio-head.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

KEYS = ("train_audio_head", "head_weights_train_audio", "LinearHead", "linear_head",
        "head_weights", "Wta", "bta", "audio_head")
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if any(k in src for k in KEYS):
        ct = c["cell_type"]
        print(f"=== Cell {i} ({ct}) ===")
        print(src[:5000])
        print()
