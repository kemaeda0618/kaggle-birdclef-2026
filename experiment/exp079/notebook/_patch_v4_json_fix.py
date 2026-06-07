"""Patch nb_blend_exp048_m7.ipynb cell-13 to add `import json as _json`.

Fix V3 error: NameError 'json' not defined in cell-13.
"""
import json
from pathlib import Path

NB = Path(__file__).with_name("nb_blend_exp048_m7.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

# Find cell-13 (M7 overlay first cell)
patched = False
for i, c in enumerate(nb["cells"]):
    if c.get("id") == "cell-13":
        src = "".join(c["source"])
        # Add import json at top
        if "import json as _json" not in src:
            # Insert after the first "# === ..." line
            lines = src.splitlines(keepends=True)
            new_lines = []
            inserted = False
            for ln in lines:
                new_lines.append(ln)
                if not inserted and ln.strip().startswith("#") and "M7 overlay" in ln:
                    new_lines.append("import json as _json   # ★ V4 fix: ensure json available\n")
                    inserted = True
            c["source"] = new_lines
            # Replace `json.load(f)` with `_json.load(f)`
            src2 = "".join(c["source"]).replace("json.load(f)", "_json.load(f)")
            c["source"] = src2.splitlines(keepends=True)
            patched = True
            print(f"Patched cell-13 (M7 overlay)")
            break

if not patched:
    print("[WARN] cell-13 not patched")

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved {NB}")

# Verify
nb2 = json.loads(NB.read_text(encoding="utf-8"))
for c in nb2["cells"]:
    if c.get("id") == "cell-13":
        first200 = "".join(c["source"])[:300]
        print(f"\nCell-13 first 300 chars:\n{first200}")
