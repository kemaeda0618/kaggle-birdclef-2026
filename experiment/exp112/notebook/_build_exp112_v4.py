"""Build exp112 V4 = V3 + path optimization.

Issue: V3 took 385s for setup (install + dir locate) vs V1 43s.
Cause: kernel_sources mount path /kaggle/input/notebooks/maekeso/... NOT in direct check list.
       rglob fallback runs ~200s recursive search of /kaggle/input/.

Fix: add /kaggle/input/notebooks/maekeso/<slug>/ to direct check paths in:
  - sed-infer (Tucker OV dir locate)
  - e29-infer (e106 OV dir locate, already has it but verify)
  - ov install wheel locate
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

V3_NB = Path("experiment/exp112/notebook/nb_exp112_v2_batched.ipynb")
V4_NB = Path("experiment/exp112/notebook/nb_exp112_v4_pathfix.ipynb")

nb = json.loads(V3_NB.read_text(encoding="utf-8"))

# Patch sed-infer cell: add notebooks/ path
sed_patched = False
for c in nb["cells"]:
    if c.get("id") != "sed-infer":
        continue
    src = c["source"]
    src = "".join(src) if isinstance(src, list) else src

    OLD_SED_FIND = """def find_sed_ov_dir():
    # Locate Tucker SED OV IR directory (kernel_sources mount path may vary)
    for p in [
        Path("/kaggle/input/birdclef2026-tucker-sed-ov"),
        Path("/kaggle/input/datasets/maekeso/birdclef2026-tucker-sed-ov"),
    ]:
        if p.exists():
            return p
    hits = sorted(Path("/kaggle/input").rglob("sed_fold0.xml"))
    assert hits, "sed_fold0.xml not found — attach maekeso/birdclef2026-tucker-sed-ov"
    return hits[0].parent
"""
    NEW_SED_FIND = """def find_sed_ov_dir():
    # ★ V4: add notebooks/ direct path (kernel_sources mount), avoid rglob fallback
    for p in [
        Path("/kaggle/input/notebooks/maekeso/birdclef2026-tucker-sed-ov"),  # ← kernel_sources mount
        Path("/kaggle/input/birdclef2026-tucker-sed-ov"),
        Path("/kaggle/input/datasets/maekeso/birdclef2026-tucker-sed-ov"),
    ]:
        if p.exists():
            return p
    hits = sorted(Path("/kaggle/input").rglob("sed_fold0.xml"))
    assert hits, "sed_fold0.xml not found — attach maekeso/birdclef2026-tucker-sed-ov"
    return hits[0].parent
"""
    assert OLD_SED_FIND in src, "find_sed_ov_dir anchor not found"
    src = src.replace(OLD_SED_FIND, NEW_SED_FIND)

    # Also optimize openvino wheel path lookup
    OLD_WHEEL_GLOB = """    _hits = sorted(glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True))
    assert _hits, "openvino wheel not found — attach ttahara/birdclef-2026-download-wheels"
    _WHEEL_DIR = str(Path(_hits[0]).parent)"""
    NEW_WHEEL_GLOB = """    # ★ V4: try direct path first to avoid expensive recursive glob
    _direct = Path("/kaggle/input/notebooks/ttahara/birdclef-2026-download-wheels/wheels")
    if _direct.exists():
        _WHEEL_DIR = str(_direct)
    else:
        _hits = sorted(glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True))
        assert _hits, "openvino wheel not found — attach ttahara/birdclef-2026-download-wheels"
        _WHEEL_DIR = str(Path(_hits[0]).parent)"""
    if OLD_WHEEL_GLOB in src:
        src = src.replace(OLD_WHEEL_GLOB, NEW_WHEEL_GLOB)

    c["source"] = src.splitlines(keepends=True)
    sed_patched = True

assert sed_patched, "sed-infer not patched"
print("[OK] sed-infer: added notebooks/ direct paths")

# Patch e29-infer cell: similar
e29_patched = False
for c in nb["cells"]:
    if c.get("id") != "e29-infer":
        continue
    src = c["source"]
    src = "".join(src) if isinstance(src, list) else src

    OLD_E106_LOCATE = """# Locate exp106 OV IR files
EXP106_DIR = None
for _p in [
    Path("/kaggle/input/birdclef2026-e106-3fold-ov"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-e106-3fold-ov"),
]:
    if _p.exists():
        EXP106_DIR = _p; break"""
    NEW_E106_LOCATE = """# Locate exp106 OV IR files (★ V4: notebooks/ direct first)
EXP106_DIR = None
for _p in [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-e106-3fold-ov"),  # ← kernel_sources
    Path("/kaggle/input/birdclef2026-e106-3fold-ov"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-e106-3fold-ov"),
]:
    if _p.exists():
        EXP106_DIR = _p; break"""
    if OLD_E106_LOCATE in src:
        src = src.replace(OLD_E106_LOCATE, NEW_E106_LOCATE)

    # openvino wheel
    OLD_WHEEL_E106 = """    _hits = sorted(glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True))
    assert _hits, "openvino wheel not found"
    _WHEEL_DIR = str(Path(_hits[0]).parent)"""
    NEW_WHEEL_E106 = """    _direct = Path("/kaggle/input/notebooks/ttahara/birdclef-2026-download-wheels/wheels")
    if _direct.exists():
        _WHEEL_DIR = str(_direct)
    else:
        _hits = sorted(glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True))
        assert _hits, "openvino wheel not found"
        _WHEEL_DIR = str(Path(_hits[0]).parent)"""
    if OLD_WHEEL_E106 in src:
        src = src.replace(OLD_WHEEL_E106, NEW_WHEEL_E106)

    c["source"] = src.splitlines(keepends=True)
    e29_patched = True

assert e29_patched, "e29-infer not patched"
print("[OK] e29-infer: added notebooks/ direct paths")

V4_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {V4_NB} ({V4_NB.stat().st_size} bytes)")

# Verify patches
nb_check = json.loads(V4_NB.read_text(encoding="utf-8"))
for c in nb_check["cells"]:
    cid = c.get("id")
    if cid in ("sed-infer", "e29-infer"):
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        assert "/kaggle/input/notebooks/maekeso" in src, f"{cid} missing notebooks/ path"
        assert "/kaggle/input/notebooks/ttahara/birdclef-2026-download-wheels/wheels" in src, f"{cid} missing wheel direct path"
        print(f"  [OK] {cid} verified")
