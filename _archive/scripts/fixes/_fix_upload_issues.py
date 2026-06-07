"""Fix:
1. exp082 cell 4: stale "exp081 / 10s" comments → "exp082 / 20s"
2. exp082 cell 18 (upload_state): robust upload with re-auth, drop preserve-R1
3. exp082 cell 20: DELETE (exp017 hardcoded, dangerous)
4. exp081 cell 19 (upload): same robust upload pattern (so R2 upload won't fail)
"""
import json
from pathlib import Path

# ============================================================
# Robust upload cell template
# ============================================================
ROBUST_UPLOAD_TEMPLATE = '''# ============================================================
# Cell: Upload weights to Kaggle Dataset (robust pattern)
# ============================================================
import tempfile, time
from kaggle.api.kaggle_api_extended import KaggleApi

# ★ Re-authenticate at upload start (long training session may have expired auth)
print("Re-authenticating Kaggle API for upload...")
# Re-load kaggle.json + KGAT token (in case env got cleared)
KAGGLE_CFG = Path.home() / ".kaggle" / "kaggle.json"
if KAGGLE_CFG.exists():
    creds = json.loads(KAGGLE_CFG.read_text())
    if creds.get("key", "").startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = creds["key"]
api = KaggleApi(); api.authenticate()
print("  Re-auth OK")

USER  = "maekeso"
SLUG  = "__SLUG__"
TITLE = "__TITLE__"

# ★ R2: r2_ prefix で R2 ckpts upload (R1 ckpts は古い version に保存され、pinning でアクセス可)
# ★ R1: そのままの名前で upload
TARGETS = __TARGETS__

print(f"\\nUpload target: {USER}/{SLUG}")
print(f"  Source dir: {OUT_DIR}")

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    # Stage files
    n_copied = 0
    total_mb = 0.0
    for src_name, dst_name in TARGETS:
        src = OUT_DIR / src_name
        if not src.exists():
            print(f"  skip (missing): {src_name}")
            continue
        shutil.copy2(str(src), str(td / dst_name))
        sz_mb = src.stat().st_size / 1e6
        total_mb += sz_mb
        n_copied += 1
        print(f"  staged: {dst_name}  ({sz_mb:.1f}MB)")
    assert n_copied > 0, "No files to upload!"
    print(f"  TOTAL: {n_copied} files, {total_mb:.1f}MB")

    # Metadata
    meta = {
        "title": TITLE,
        "id": f"{USER}/{SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    version_notes = __VERSION_NOTES__
    uploaded = False
    last_err = None

    # ★ 戦略: まず create_version (既存に追加) を試す、404 ならば create_new
    # 古い version_view check 撤去 (信頼性低かった)
    print(f"\\n[1/2] Try create_version (existing dataset)...")
    try:
        t0 = time.time()
        api.dataset_create_version(folder=str(td),
                                    version_notes=version_notes,
                                    dir_mode="zip", quiet=False)
        print(f"  OK Uploaded (new version, {time.time()-t0:.0f}s) — {version_notes}")
        uploaded = True
    except Exception as e:
        msg = str(e)[:600]
        print(f"  create_version FAILED: type={type(e).__name__}")
        print(f"    msg: {msg}")
        last_err = e
        # Fall through to create_new only if "not found" / 404
        if "not found" in msg.lower() or "404" in msg or "Could not find" in msg or "doesn\\'t exist" in msg.lower():
            print(f"\\n[2/2] Dataset doesn\\'t exist yet → try create_new...")
            try:
                t0 = time.time()
                api.dataset_create_new(folder=str(td), public=False, dir_mode="zip", quiet=False)
                print(f"  OK Created (first time, {time.time()-t0:.0f}s)")
                uploaded = True
            except Exception as e2:
                msg2 = str(e2)[:600]
                print(f"  create_new FAILED: type={type(e2).__name__}")
                print(f"    msg: {msg2}")
                last_err = e2

    if uploaded:
        print(f"\\n[OK] Dataset URL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
    else:
        print(f"\\n[FAIL] Upload failed. Last error:")
        print(f"  {type(last_err).__name__}: {str(last_err)[:500]}")
        print(f"\\nFiles remain at {OUT_DIR}, manual upload via Kaggle UI possible:")
        for src_name, _ in TARGETS:
            p = OUT_DIR / src_name
            if p.exists():
                print(f"  {p}")
'''


def make_robust_upload_cell(slug, title, targets, version_notes_expr):
    """Build robust upload cell source.
    targets: list of (src_name, dst_name) tuples
    version_notes_expr: Python expression string for version_notes
    """
    targets_repr = "[\n" + ",\n".join(f'    ({repr(s)}, {repr(d)})' for s, d in targets) + ",\n]"
    src = (ROBUST_UPLOAD_TEMPLATE
           .replace("__SLUG__", slug)
           .replace("__TITLE__", title)
           .replace("__TARGETS__", targets_repr)
           .replace("__VERSION_NOTES__", version_notes_expr))
    return src


# ============================================================
# Fix exp082 nb_train_r2.ipynb
# ============================================================
P = Path("experiment/exp082/notebook/nb_train_r2.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

# Find cells by id
cell_map = {c.get("id"): (i, c) for i, c in enumerate(nb["cells"])}

# 1. Fix cell 4 (load_data) — stale "exp081 / 10s" comments
i, c = cell_map["load_data"]
src = "".join(c["source"])
src = src.replace(
    "    # ★ exp081: Aggregate labeled SS 5s -> 10s windows (max() merge)",
    "    # ★ exp082: Aggregate labeled SS 5s -> 20s windows (4× max merge)")
src = src.replace(
    "    # Pair adjacent 5s chunks (start_sec, start_sec+5) -> 10s window at start_sec",
    "    # Aggregate 4 consecutive 5s chunks (start_sec, +5, +10, +15) -> 20s window")
src = src.replace(
    "    # Result: 6 non-overlapping 10s windows per file (vs 12 5s windows)",
    "    # Result: 3 non-overlapping 20s windows per file (vs 12 5s windows)")
src = src.replace(
    '    print("[exp081] Aggregating labeled SS 5s -> 10s windows...")',
    '    print("[exp082] Aggregating labeled SS 5s -> 20s windows...")')
src = src.replace(
    '            start_sec_10s = float(group_sorted.iloc[k]["start_sec"])  # 20s window start',
    '            start_sec_20s = float(group_sorted.iloc[k]["start_sec"])  # 20s window start')
src = src.replace(
    '            new_sc_rows.append({"filename": fname, "start_sec": start_sec_10s, "site": site_v})',
    '            new_sc_rows.append({"filename": fname, "start_sec": start_sec_20s, "site": site_v})')
src = src.replace(
    '    print(f"  [exp081] After labeled SS 10s aggregation: {len(sc_meta)} windows, "',
    '    print(f"  [exp082] After labeled SS 20s aggregation: {len(sc_meta)} windows, "')
c["source"] = src.splitlines(keepends=True)
print(f"OK Fixed exp082 cell 4 (load_data) — stale comments and variable name")

# 2. Replace cell 18 (upload_state) with robust pattern
i, c = cell_map["upload_state"]
new_src = make_robust_upload_cell(
    slug="birdclef2026-exp082-weights",
    title="birdclef2026 exp082 weights",
    targets=[
        ("ckpt_best_ns22.pth",  "r2_ckpt_best_ns22.pth"),
        ("ckpt_best_macro.pth", "r2_ckpt_best_macro.pth"),
        ("ckpt_latest.pth",     "r2_ckpt_latest.pth"),
        ("history.json",        "r2_history.json"),
    ],
    version_notes_expr='f"R2 best_ns22={best_ns22:.4f}, best_macro={best_macro:.4f}"',
)
c["source"] = new_src.splitlines(keepends=True)
print(f"OK Fixed exp082 cell 18 (upload_state) — robust pattern, TITLE fixed, dropped preserve-R1")

# 3. Delete cell 20 (manual upload, exp017 hardcoded)
# Find cells with exp017 hardcoded
to_delete = []
for i, c in enumerate(nb["cells"]):
    cid = c.get("id", "")
    src = "".join(c.get("source", []))
    if c.get("cell_type") != "code":
        continue
    # Specifically the orphan exp017 helper cell
    if 'output" / "exp017"' in src and 'birdclef2026-exp017-weights' in src:
        print(f"  Cell {i} id={cid} contains exp017 hardcoded → DELETE")
        to_delete.append(i)

# Delete in reverse order
for i in reversed(to_delete):
    del nb["cells"][i]
    print(f"  Deleted cell at index {i}")

P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nOK Saved {P}\n")


# ============================================================
# Fix exp081 nb_train_r1.ipynb upload (same robust pattern)
# ============================================================
P = Path("experiment/exp081/notebook/nb_train_r1.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

cell_map = {c.get("id"): (i, c) for i, c in enumerate(nb["cells"])}

# Replace cell "upload" with robust pattern (exp081 R1: no r2_ prefix needed)
if "upload" in cell_map:
    i, c = cell_map["upload"]
    new_src = make_robust_upload_cell(
        slug="birdclef2026-exp081-weights",
        title="birdclef2026 exp081 weights",
        targets=[
            ("ckpt_best_ns22.pth",  "ckpt_best_ns22.pth"),
            ("ckpt_best_macro.pth", "ckpt_best_macro.pth"),
            ("ckpt_latest.pth",     "ckpt_latest.pth"),
            ("history.json",        "history.json"),
        ],
        version_notes_expr='f"R1 best_ns22={best_ns22:.4f}, best_macro={best_macro:.4f}"',
    )
    c["source"] = new_src.splitlines(keepends=True)
    print(f"OK Fixed exp081 cell id=upload — robust pattern")

P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"OK Saved {P}\n")


# ============================================================
# Verify syntax
# ============================================================
import ast
for nb_path in ["experiment/exp082/notebook/nb_train_r2.ipynb",
                "experiment/exp081/notebook/nb_train_r1.ipynb"]:
    print(f"\n=== Syntax check: {nb_path} ===")
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    n_ok, n_err = 0, 0
    for c in nb["cells"]:
        if c.get("cell_type") != "code": continue
        src = "".join(c.get("source", []))
        clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                          for ln in src.splitlines())
        try:
            ast.parse(clean)
            n_ok += 1
        except SyntaxError as e:
            n_err += 1
            print(f"  SyntaxError in cell {c.get('id','?')}: {e}")
            err_line = e.lineno or 1
            for i, ln in enumerate(clean.splitlines()[max(0, err_line-3):err_line+2]):
                marker = ">>> " if i + max(0, err_line-3) + 1 == err_line else "    "
                print(f"  {marker}{ln}")
    print(f"  Total: {n_ok} OK, {n_err} ERRORS")
