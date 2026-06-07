"""Build exp104 V3: V2 + Tucker SED 5-fold -> 3-fold to fit 90 min budget.

V2 (timeout): NB4 + Tucker OV 5-fold + exp102 OV 3-fold = ~95 min actual
V3 (fit):     NB4 + Tucker OV 3-fold + exp102 OV 3-fold = ~89 min target

Change: limit Tucker to fold 0, 1, 2 only (drop fold 3, 4).
Expected LB cost: -0.0005 to -0.001 (Tucker 5->3 fold variance reduction loss).
Expected time saving: ~6 min on 600 files inference.
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Load existing V2 NB (already has Tucker OV + exp102 3-fold OV)
V2_NB = Path("experiment/exp104/notebook/nb_exp104_e29_swap_3fold.ipynb")
V3_NB = Path("experiment/exp104/notebook/nb_exp104_v3_tucker3fold.ipynb")
nb = json.loads(V2_NB.read_text(encoding="utf-8"))

# Find sed-infer cell, patch to limit folds to [0, 1, 2]
patched = False
for c in nb["cells"]:
    if c.get("id") != "sed-infer":
        continue
    src = c["source"]
    src = "".join(src) if isinstance(src, list) else src

    # Original line: assert len(sed_xml_paths) == 5
    # Replace with: limit to first 3 folds + assert >= 3
    old_block = """sed_xml_paths = sorted(sed_ov_dir.glob("sed_fold*.xml"),
                        key=lambda p: int(re.search(r"sed_fold(\\d+)", p.name).group(1)))
print(f"SED OV dir: {sed_ov_dir}")
print(f"SED OV folds: {[p.name for p in sed_xml_paths]}")
assert len(sed_xml_paths) == 5, f"expected 5 folds, got {len(sed_xml_paths)}"
"""

    new_block = """sed_xml_paths = sorted(sed_ov_dir.glob("sed_fold*.xml"),
                        key=lambda p: int(re.search(r"sed_fold(\\d+)", p.name).group(1)))
print(f"SED OV dir: {sed_ov_dir}")
print(f"SED OV folds available: {[p.name for p in sed_xml_paths]}")
assert len(sed_xml_paths) >= 3, f"expected >= 3 folds, got {len(sed_xml_paths)}"
# ★ V3: limit to first 3 folds (fold 0, 1, 2) to save ~6 min vs 5-fold
sed_xml_paths = sed_xml_paths[:3]
print(f"SED OV folds USED (V3 = 3-fold subset): {[p.name for p in sed_xml_paths]}")
"""

    if old_block in src:
        src = src.replace(old_block, new_block)
        patched = True
        # update source as list (notebook format)
        c["source"] = src.splitlines(keepends=True)
        break

assert patched, "sed_xml_paths anchor not found in V2 NB sed-infer cell"

V3_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {V3_NB} ({V3_NB.stat().st_size} bytes)")

# Verify: confirm patch applied
nb_check = json.loads(V3_NB.read_text(encoding="utf-8"))
for c in nb_check["cells"]:
    if c.get("id") == "sed-infer":
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        assert "sed_xml_paths = sed_xml_paths[:3]" in src, "patch verify failed"
        assert "V3 = 3-fold subset" in src, "comment verify failed"
        print("[OK] V3 patch verified in sed-infer cell")
        break
