"""V4 patch: fix the 3 V3 bugs.

1. NameError: json not imported in save_addon → add import
2. exp037 dry-run override fails: patch exp037_full_pipeline cell to use labeled SS in dry-run fallback
3. truth_ss empty: depends on (2) being fixed
"""
import json, io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "nb_exp090_oof_v3.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# ============================================================
# Fix 1: Patch exp037_full_pipeline cell — override dry-run fallback
# ============================================================
target_old = '    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:n]'
target_new = '''    # ★ V4 patch: use labeled SS files instead of n dry-run files
    _sc_lbl_v4 = pd.read_csv(BASE / "train_soundscapes_labels.csv")
    _labeled_files_v4 = sorted(_sc_lbl_v4["filename"].unique().tolist())
    test_paths = []
    for _fn_v4 in _labeled_files_v4:
        _fn_str_v4 = str(_fn_v4)
        if not _fn_str_v4.endswith(".ogg"):
            _fn_str_v4 = _fn_str_v4 + ".ogg"
        _p_v4 = (BASE / "train_soundscapes") / _fn_str_v4
        if _p_v4.exists():
            test_paths.append(_p_v4)
    print(f"[V4 PATCH] Using {len(test_paths)} labeled SS files as test_paths (not dry-run)")'''

for c in nb["cells"]:
    if c.get("id") == "v313_23":
        src = "".join(c["source"])
        if target_old in src:
            new_src = src.replace(target_old, target_new)
            c["source"] = new_src.splitlines(keepends=True)
            print(f"[OK] Patched v313_23 (dry-run fallback)")
        else:
            print(f"[WARN] target line not found in v313_23")
        break

# ============================================================
# Fix 2: Add json import to save_addon in blend cell
# ============================================================
for c in nb["cells"]:
    if c.get("id") == "blend":
        src = "".join(c["source"])
        # Find the save_addon section and add import json
        old = "# Apply tax smoothing\nsubmission = f_tax_smoothing(submission, genus_alpha=0.15, class_alpha=0.05)\n\nsubmission.to_csv(\"submission.csv\", index=False)"
        # Actually the addon starts later. Let me find a more specific marker.
        # The bug was: json.dump(PRIMARY_LABELS, _f) inside save_addon
        # We need to add `import json` near the top of save_addon
        marker = "# ★ V3: Save labeled SS predictions for analysis"
        if marker in src:
            # Insert `import json` right after the marker
            insert_after = "# ============================================================\n"
            # Find specifically the section
            idx = src.index(marker)
            # Find the next line after this marker line
            end_of_marker_line = src.index("\n", idx) + 1
            # Find end of the next ====== line
            end_of_sep_line = src.index("\n", end_of_marker_line) + 1
            new_src = src[:end_of_sep_line] + "import json  # ★ V4 fix: missing import\n" + src[end_of_sep_line:]
            c["source"] = new_src.splitlines(keepends=True)
            print("[OK] Added `import json` to save_addon")
        else:
            print(f"[WARN] V3 save marker not found in blend cell")
        break

# ============================================================
# Save
# ============================================================
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nSaved: {NB_PATH}")

# Compile-check
print("\n=== Compile check ===")
n_ok = n_err = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        compile(clean, f"<cell-{i}>", "exec")
        n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  [FAIL] cell {i} (id={c.get('id','?')}): {e}")
print(f"Result: {n_ok} OK, {n_err} ERRORS")

# Verify fix 1 location
print("\n=== Fix 1 verification (v313_23) ===")
for c in nb["cells"]:
    if c.get("id") == "v313_23":
        src = "".join(c["source"])
        if "V4 PATCH" in src:
            print("  [OK] V4 PATCH marker found in v313_23")
        else:
            print("  [FAIL] V4 PATCH not in v313_23")
        # Show the patched block
        for ln in src.split("\n"):
            if "V4 PATCH" in ln or "_labeled_files_v4" in ln or "test_paths.append" in ln:
                print(f"    {ln[:140]}")
        break

# Verify fix 2
print("\n=== Fix 2 verification (blend cell json import) ===")
for c in nb["cells"]:
    if c.get("id") == "blend":
        src = "".join(c["source"])
        if "import json  # ★ V4 fix" in src:
            print("  [OK] json import added to save_addon")
        else:
            print("  [FAIL] json import not added")
        break
