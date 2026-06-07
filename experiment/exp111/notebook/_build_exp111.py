"""Build exp111 = exp104 V3 base + blend weight 0.30/0.40/0.30 + exp102 fold [1, 2].

Strategy:
  - Use exp104 V3 (Tucker OV 3-fold + exp102 OV 3-fold) as starting point
  - Tucker: keep 3-fold (V3) OR upgrade to 5-fold (more LB but more time)
  - exp102: limit to fold [1, 2] (drop fold 0 weakest)
  - Blend weight: change from 0.35/0.40/0.25 to 0.30/0.40/0.30 (exp110)

Decision: Tucker 5-fold + exp102 2-fold
  - Tucker 5-fold: +0.001 LB benefit (vs 3-fold), -3 min time vs Tucker ONNX
  - exp102 2-fold (1,2): drop fold 0 (weakest, LB ~0.917)
  - Estimated sub time: ~82-85 min (vs V3 with 3-fold ~85)
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

V3_NB = Path("experiment/exp104/notebook/nb_exp104_v3_tucker3fold.ipynb")
EXP111_NB = Path("experiment/exp111/notebook/nb_exp111_e102_fold12_w30_40_30.ipynb")
EXP111_NB.parent.mkdir(parents=True, exist_ok=True)

nb = json.loads(V3_NB.read_text(encoding="utf-8"))

# --- Patch 1: sed-infer cell — REMOVE V3 3-fold limit (restore 5-fold) ---
sed_patched = False
for c in nb["cells"]:
    if c.get("id") != "sed-infer":
        continue
    src = c["source"]
    src = "".join(src) if isinstance(src, list) else src

    # V3 has: sed_xml_paths = sed_xml_paths[:3]   (limit to 3 folds)
    # Restore 5-fold (remove the slice)
    V3_LIMIT = """# ★ V3: limit to first 3 folds (fold 0, 1, 2) to save ~6 min vs 5-fold
sed_xml_paths = sed_xml_paths[:3]
print(f"SED OV folds USED (V3 = 3-fold subset): {[p.name for p in sed_xml_paths]}")
"""
    EXP111_FULL = """# ★ exp111: restore Tucker SED 5-fold full ensemble (exp102 fold 1+2 で時間節約済、Tucker 5-fold 維持可能)
print(f"SED OV folds USED (exp111 = 5-fold full): {[p.name for p in sed_xml_paths]}")
"""
    if V3_LIMIT in src:
        src = src.replace(V3_LIMIT, EXP111_FULL)
        c["source"] = src.splitlines(keepends=True)
        sed_patched = True
        break

assert sed_patched, "sed-infer V3 limit anchor not found"
print("[OK] Patch 1: sed-infer restored to 5-fold")

# --- Patch 2: e29-infer cell — exp102 FOLDS [0,1,2] → [1,2] (drop fold 0) ---
e29_patched = False
for c in nb["cells"]:
    if c.get("id") != "e29-infer":
        continue
    src = c["source"]
    src = "".join(src) if isinstance(src, list) else src

    OLD_FOLDS = "FOLDS_E102 = [0, 1, 2]"
    NEW_FOLDS = "FOLDS_E102 = [1, 2]   # ★ exp111: drop fold 0 (val 0.9351、Aves plateau、推定 LB 最低)"
    assert OLD_FOLDS in src, "FOLDS_E102 = [0, 1, 2] anchor not found"
    src = src.replace(OLD_FOLDS, NEW_FOLDS)
    c["source"] = src.splitlines(keepends=True)
    e29_patched = True
    break

assert e29_patched, "e29-infer FOLDS_E102 anchor not found"
print("[OK] Patch 2: e29-infer FOLDS_E102 = [1, 2]")

# --- Patch 3: blend cell — weight 0.35/0.40/0.25 → 0.30/0.40/0.30 ---
blend_patched = False
for c in nb["cells"]:
    if c.get("id") != "blend":
        continue
    src = c["source"]
    src = "".join(src) if isinstance(src, list) else src

    OLD_W = """BLEND_W_E10  = 0.35   # NB4 v7 weight
BLEND_W_SED  = 0.40   # Tucker public SED weight
BLEND_W_E17  = 0.25   # exp029 R3 (eca_nfnet_l1) weight — note var name kept "E17" for code reuse"""
    NEW_W = """BLEND_W_E10  = 0.30   # ★ exp111: NB4 weight 0.35 → 0.30 (exp110 weight tune adoption)
BLEND_W_SED  = 0.40   # Tucker SED weight (維持)
BLEND_W_E17  = 0.30   # ★ exp111: exp102 stream weight 0.25 → 0.30 (exp110 weight tune adoption)"""
    assert OLD_W in src, "blend weight anchor not found"
    src = src.replace(OLD_W, NEW_W)
    c["source"] = src.splitlines(keepends=True)
    blend_patched = True
    break

assert blend_patched, "blend weight anchor not found"
print("[OK] Patch 3: blend weight = 0.30/0.40/0.30")

# Save
EXP111_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"\nWrote {EXP111_NB} ({EXP111_NB.stat().st_size} bytes)")

# Final verify
nb_check = json.loads(EXP111_NB.read_text(encoding="utf-8"))
verify_results = {}
for c in nb_check["cells"]:
    cid = c.get("id")
    if cid in ("sed-infer", "e29-infer", "blend"):
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        verify_results[cid] = src

assert "sed_xml_paths[:3]" not in verify_results["sed-infer"], "V3 limit still present"
assert "5-fold full" in verify_results["sed-infer"], "5-fold restoration missing"
assert "FOLDS_E102 = [1, 2]" in verify_results["e29-infer"], "FOLDS_E102 not patched"
assert "BLEND_W_E10  = 0.30" in verify_results["blend"]
assert "BLEND_W_E17  = 0.30" in verify_results["blend"]
print("\n[ALL OK] exp111 NB ready")
