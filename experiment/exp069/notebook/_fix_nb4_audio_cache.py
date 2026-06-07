"""Surgical fix: remove audio_cache.append in nb_pseudo_nb4.ipynb"""
import json
from pathlib import Path

NB_PATH = Path(__file__).with_name("nb_pseudo_nb4.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# Cell 9: replace audio_cache.append with comment
src9 = "".join(nb["cells"][9]["source"])
old = "    audio_cache.append(raw_60s)\n"
new = "    # ★ streaming: skip audio_cache.append (memory OOM avoidance)\n"
if old in src9:
    src9_new = src9.replace(old, new)
    nb["cells"][9]["source"] = src9_new.splitlines(keepends=True)
    nb["cells"][9]["outputs"] = []
    nb["cells"][9]["execution_count"] = None
    print("  Cell 9: audio_cache.append removed (4-space indent matched)")
else:
    print(f"  Cell 9: pattern NOT found, current state has these audio_cache.append lines:")
    for ln in src9.split("\n"):
        if "audio_cache.append" in ln:
            print(f"    >{ln}<")

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {NB_PATH}")
