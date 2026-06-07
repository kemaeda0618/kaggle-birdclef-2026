"""Inject the 162 Aves list into the NB by replacing __AVES_PLACEHOLDER__."""
import json
from pathlib import Path

# Read the list from _aves_list.py
aves_src = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_aves_list.py").read_text(encoding="utf-8")
# Extract lines starting with 4-space indent + (
lines = []
for line in aves_src.split("\n"):
    s = line.strip()
    if s.startswith("(") and s.endswith("),"):
        lines.append("    " + s)
print(f"Extracted {len(lines)} Aves species tuples")

aves_block = "\n".join(lines)

# Patch the gen script then re-run it
gen_script = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_gen_nb_xc_api_dl.py")
script_src = gen_script.read_text(encoding="utf-8")
new_script_src = script_src.replace("__AVES_PLACEHOLDER__", aves_block)
gen_script.write_text(new_script_src, encoding="utf-8")
print("Patched _gen_nb_xc_api_dl.py with embedded Aves list")
