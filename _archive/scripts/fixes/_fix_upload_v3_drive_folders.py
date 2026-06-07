"""R2 upload: R1 from Drive + R2 from local, folder-separated structure.

Final Dataset structure (latest version):
  r1/
    ckpt_best_ns22.pth   ← from Drive (exp082/r1/)
    ckpt_best_macro.pth  ← from Drive
    ckpt_latest.pth      ← from Drive
    history.json         ← from Drive
  r2/
    ckpt_best_ns22.pth   ← from local OUT_DIR (after R2 train)
    ckpt_best_macro.pth  ← from local
    ckpt_latest.pth      ← from local
    history.json         ← from local

Advantages over Kaggle Dataset download:
- No network failure risk
- Faster (Drive copy vs Kaggle API DL)
- Cleaner folder structure
- Always works as long as Drive R1 ckpts exist (which they do after R1 train)

Note: infer NB は path 更新必要 (例: `ckpt_best_ns22.pth` → `r1/ckpt_best_ns22.pth`)
"""
import json
from pathlib import Path


R2_UPLOAD_DRIVE = '''# ============================================================
# Cell: Upload R2 weights to Kaggle Dataset (R1 from Drive + R2 from local)
# ============================================================
import tempfile, time
from kaggle.api.kaggle_api_extended import KaggleApi

# ★ Re-authenticate at upload start
print("Re-authenticating Kaggle API for upload...")
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

# R1 files location (Drive) — sibling of DRIVE_OUTPUT_DIR (R2 dir)
DRIVE_R1_DIR = DRIVE_OUTPUT_DIR.parent / "r1"

# R1 ckpts to copy from Drive (R1 stage)
TARGETS_R1 = [
    "ckpt_best_ns22.pth",
    "ckpt_best_macro.pth",
    "ckpt_latest.pth",
    "history.json",
]
# R2 ckpts to copy from local OUT_DIR (R2 stage)
TARGETS_R2 = [
    "ckpt_best_ns22.pth",
    "ckpt_best_macro.pth",
    "ckpt_latest.pth",
    "history.json",
]

print(f"\\nUpload target: {USER}/{SLUG}")
print(f"  R1 source: {DRIVE_R1_DIR}")
print(f"  R2 source: {OUT_DIR}")

with tempfile.TemporaryDirectory() as td:
    td = Path(td)

    # ============================================================
    # [1/3] Stage R1 files (Drive → temp_dir/r1/)
    # ============================================================
    r1_dst = td / "r1"
    r1_dst.mkdir(parents=True, exist_ok=True)
    print(f"\\n[1/3] Staging R1 files (Drive → temp/r1/)")
    n_r1 = 0
    r1_mb = 0.0
    if not DRIVE_R1_DIR.exists():
        print(f"  [WARN] DRIVE_R1_DIR not found: {DRIVE_R1_DIR}")
        print(f"  → R1 will NOT be preserved. R2 upload will proceed.")
    else:
        for fn in TARGETS_R1:
            src = DRIVE_R1_DIR / fn
            if not src.exists():
                print(f"  skip (missing on Drive): r1/{fn}")
                continue
            shutil.copy2(str(src), str(r1_dst / fn))
            sz_mb = src.stat().st_size / 1e6
            r1_mb += sz_mb
            n_r1 += 1
            print(f"  R1 staged: r1/{fn}  ({sz_mb:.1f}MB)")
    print(f"  R1 TOTAL: {n_r1} files, {r1_mb:.1f}MB")

    # ============================================================
    # [2/3] Stage R2 files (local OUT_DIR → temp_dir/r2/)
    # ============================================================
    r2_dst = td / "r2"
    r2_dst.mkdir(parents=True, exist_ok=True)
    print(f"\\n[2/3] Staging R2 files (local OUT_DIR → temp/r2/)")
    n_r2 = 0
    r2_mb = 0.0
    for fn in TARGETS_R2:
        src = OUT_DIR / fn
        if not src.exists():
            print(f"  skip (missing local): r2/{fn}")
            continue
        shutil.copy2(str(src), str(r2_dst / fn))
        sz_mb = src.stat().st_size / 1e6
        r2_mb += sz_mb
        n_r2 += 1
        print(f"  R2 staged: r2/{fn}  ({sz_mb:.1f}MB)")
    assert n_r2 > 0, "No R2 files to upload! Aborting."
    print(f"  R2 TOTAL: {n_r2} files, {r2_mb:.1f}MB")

    # Final staging summary
    print(f"\\n  Final staging tree:")
    for sub in ["r1", "r2"]:
        sub_dir = td / sub
        if sub_dir.exists():
            for f in sorted(sub_dir.iterdir()):
                if f.is_file():
                    sz_mb = f.stat().st_size / 1e6
                    print(f"    {sub}/{f.name}  ({sz_mb:.1f}MB)")
    total_mb = sum(f.stat().st_size for f in td.rglob("*") if f.is_file()) / 1e6
    print(f"  Grand total: {total_mb:.1f}MB")

    # ============================================================
    # [3/3] Upload (create_version → fallback create_new)
    # ============================================================
    meta = {
        "title": TITLE,
        "id": f"{USER}/{SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    version_notes = __VERSION_NOTES__
    uploaded = False
    last_err = None

    print(f"\\n[3/3] Uploading to Kaggle Dataset...")
    print(f"  version_notes: {version_notes}")
    try:
        t0 = time.time()
        api.dataset_create_version(folder=str(td),
                                    version_notes=version_notes,
                                    dir_mode="zip", quiet=False)
        print(f"\\n  OK Uploaded (new version, {time.time()-t0:.0f}s)")
        uploaded = True
    except Exception as e:
        msg = str(e)[:600]
        print(f"  create_version FAILED: type={type(e).__name__}")
        print(f"    msg: {msg}")
        last_err = e
        if "not found" in msg.lower() or "404" in msg or "Could not find" in msg:
            print(f"\\n  Dataset doesn\\'t exist → try create_new...")
            try:
                t0 = time.time()
                api.dataset_create_new(folder=str(td), public=False,
                                       dir_mode="zip", quiet=False)
                print(f"  OK Created (first time, {time.time()-t0:.0f}s)")
                uploaded = True
            except Exception as e2:
                msg2 = str(e2)[:600]
                print(f"  create_new FAILED: type={type(e2).__name__}")
                print(f"    msg: {msg2}")
                last_err = e2

    if uploaded:
        print(f"\\n[OK] Dataset URL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
        print(f"     Structure: r1/{{ckpt_*.pth, history.json}}, r2/{{ckpt_*.pth, history.json}}")
        print(f"     Both R1 + R2 visible in latest version")
    else:
        print(f"\\n[FAIL] Upload failed. Last error:")
        print(f"  {type(last_err).__name__}: {str(last_err)[:500]}")
        print(f"\\nFiles staged at {td} (will be cleaned up).")
'''


def make_upload_cell(slug, title, version_notes_expr):
    return (R2_UPLOAD_DRIVE
            .replace("__SLUG__", slug)
            .replace("__TITLE__", title)
            .replace("__VERSION_NOTES__", version_notes_expr))


# Apply to exp082 R2
P = Path("experiment/exp082/notebook/nb_train_r2.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))
cell_map = {c.get("id"): (i, c) for i, c in enumerate(nb["cells"])}

i, c = cell_map["upload_state"]
new_src = make_upload_cell(
    slug="birdclef2026-exp082-weights",
    title="birdclef2026 exp082 weights",
    version_notes_expr='f"R2 best_ns22={best_ns22:.4f}, best_macro={best_macro:.4f} (r1/ + r2/ folders)"',
)
c["source"] = new_src.splitlines(keepends=True)
P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"OK Fixed exp082 R2 upload — Drive R1 + local R2, folder structure")


# Also update exp081 R1 NB upload cell (R1 only, to r1/ subfolder for consistency)
R1_UPLOAD_DRIVE = '''# ============================================================
# Cell: Upload R1 weights to Kaggle Dataset (r1/ subfolder structure)
# ============================================================
import tempfile, time
from kaggle.api.kaggle_api_extended import KaggleApi

# ★ Re-authenticate at upload start (long training session may have expired auth)
print("Re-authenticating Kaggle API for upload...")
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

# R1 ckpts to upload (from local OUT_DIR after R1 train)
TARGETS_R1 = [
    "ckpt_best_ns22.pth",
    "ckpt_best_macro.pth",
    "ckpt_latest.pth",
    "history.json",
]

print(f"\\nUpload target: {USER}/{SLUG}")
print(f"  R1 source: {OUT_DIR}")

with tempfile.TemporaryDirectory() as td:
    td = Path(td)

    # Stage R1 files into r1/ subfolder
    r1_dst = td / "r1"
    r1_dst.mkdir(parents=True, exist_ok=True)
    print(f"\\nStaging R1 files (local OUT_DIR → temp/r1/)")
    n_r1 = 0
    r1_mb = 0.0
    for fn in TARGETS_R1:
        src = OUT_DIR / fn
        if not src.exists():
            print(f"  skip (missing): r1/{fn}")
            continue
        shutil.copy2(str(src), str(r1_dst / fn))
        sz_mb = src.stat().st_size / 1e6
        r1_mb += sz_mb
        n_r1 += 1
        print(f"  R1 staged: r1/{fn}  ({sz_mb:.1f}MB)")
    assert n_r1 > 0, "No R1 files to upload!"
    print(f"  R1 TOTAL: {n_r1} files, {r1_mb:.1f}MB")

    meta = {
        "title": TITLE,
        "id": f"{USER}/{SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    version_notes = __VERSION_NOTES__
    uploaded = False
    last_err = None

    print(f"\\nUploading to Kaggle Dataset...")
    try:
        t0 = time.time()
        api.dataset_create_version(folder=str(td),
                                    version_notes=version_notes,
                                    dir_mode="zip", quiet=False)
        print(f"  OK Uploaded (new version, {time.time()-t0:.0f}s)")
        uploaded = True
    except Exception as e:
        msg = str(e)[:600]
        print(f"  create_version FAILED: type={type(e).__name__}")
        print(f"    msg: {msg}")
        last_err = e
        if "not found" in msg.lower() or "404" in msg or "Could not find" in msg:
            print(f"\\n  Dataset doesn\\'t exist → try create_new...")
            try:
                t0 = time.time()
                api.dataset_create_new(folder=str(td), public=False,
                                       dir_mode="zip", quiet=False)
                print(f"  OK Created (first time, {time.time()-t0:.0f}s)")
                uploaded = True
            except Exception as e2:
                msg2 = str(e2)[:600]
                print(f"  create_new FAILED: type={type(e2).__name__}")
                print(f"    msg: {msg2}")
                last_err = e2

    if uploaded:
        print(f"\\n[OK] Dataset URL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
        print(f"     Structure: r1/{{ckpt_*.pth, history.json}}")
    else:
        print(f"\\n[FAIL] Upload failed. Last error:")
        print(f"  {type(last_err).__name__}: {str(last_err)[:500]}")
'''


def make_r1_upload_cell(slug, title, version_notes_expr):
    return (R1_UPLOAD_DRIVE
            .replace("__SLUG__", slug)
            .replace("__TITLE__", title)
            .replace("__VERSION_NOTES__", version_notes_expr))


# Apply to exp081 R1
P = Path("experiment/exp081/notebook/nb_train_r1.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))
cell_map = {c.get("id"): (i, c) for i, c in enumerate(nb["cells"])}

if "upload" in cell_map:
    i, c = cell_map["upload"]
    new_src = make_r1_upload_cell(
        slug="birdclef2026-exp081-weights",
        title="birdclef2026 exp081 weights",
        version_notes_expr='f"R1 best_ns22={best_ns22:.4f}, best_macro={best_macro:.4f} (r1/ folder)"',
    )
    c["source"] = new_src.splitlines(keepends=True)
    P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"OK Fixed exp081 R1 upload — r1/ folder structure")


# Syntax check
import ast
for nb_path in ["experiment/exp082/notebook/nb_train_r2.ipynb",
                "experiment/exp081/notebook/nb_train_r1.ipynb"]:
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
            print(f"  SyntaxError {nb_path} cell {c.get('id','?')}: {e}")
    print(f"{nb_path}: {n_ok} OK, {n_err} ERRORS")
