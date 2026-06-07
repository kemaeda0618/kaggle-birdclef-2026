"""Reinstate preserve-R1 step robustly for R2 upload cells.

R2 upload flow:
1. Re-authenticate
2. Stage R2 files (r2_ prefix) from local OUT_DIR
3. ★ Download existing Dataset files (R1 ckpts), copy non-r2_* to staging
4. Upload all together (R1 + R2) → both visible in latest version

Robust:
- If download R1 fails → WARN + continue with R2-only upload (with explicit notification)
- Verify file count + sizes after staging
- Print which R1 files are preserved
"""
import json
from pathlib import Path


ROBUST_R2_UPLOAD = '''# ============================================================
# Cell: Upload R2 weights to Kaggle Dataset (preserve R1 + robust)
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

# R2 ckpts to upload (r2_ prefix で別名保存)
TARGETS_R2 = __TARGETS_R2__

print(f"\\nUpload target: {USER}/{SLUG}")
print(f"  Source dir: {OUT_DIR}")

with tempfile.TemporaryDirectory() as td:
    td = Path(td)

    # ============================================================
    # [1/3] Stage R2 files (local → temp dir)
    # ============================================================
    print(f"\\n[1/3] Staging R2 files (local → temp dir)")
    n_r2 = 0
    r2_mb = 0.0
    for src_name, dst_name in TARGETS_R2:
        src = OUT_DIR / src_name
        if not src.exists():
            print(f"  skip (missing): {src_name}")
            continue
        shutil.copy2(str(src), str(td / dst_name))
        sz_mb = src.stat().st_size / 1e6
        r2_mb += sz_mb
        n_r2 += 1
        print(f"  R2 staged: {dst_name}  ({sz_mb:.1f}MB)")
    assert n_r2 > 0, "No R2 files to upload!"
    print(f"  R2 TOTAL: {n_r2} files, {r2_mb:.1f}MB")

    # ============================================================
    # [2/3] Preserve R1 files (download existing Dataset → temp dir)
    # ============================================================
    print(f"\\n[2/3] Downloading existing R1 files from {USER}/{SLUG}...")
    r1_preserved = False
    n_r1_preserved = 0
    r1_mb = 0.0
    try:
        tmp_dl = td / "_r1_existing"
        tmp_dl.mkdir(exist_ok=True)
        t0 = time.time()
        api.dataset_download_files(f"{USER}/{SLUG}", path=str(tmp_dl),
                                    unzip=True, quiet=False)
        dl_time = time.time() - t0
        print(f"  R1 download OK in {dl_time:.0f}s")

        # Copy non-r2_* files (R1 ckpts and history) to staging dir
        for f in tmp_dl.iterdir():
            if not f.is_file():
                continue
            # Skip r2_* (those are old R2, will be replaced)
            if f.name.startswith("r2_"):
                print(f"    skip old R2: {f.name}")
                continue
            # Skip if same name already staged (shouldn\\'t happen with r2_ prefix)
            if (td / f.name).exists():
                print(f"    skip already staged: {f.name}")
                continue
            shutil.copy2(str(f), str(td / f.name))
            sz_mb = f.stat().st_size / 1e6
            r1_mb += sz_mb
            n_r1_preserved += 1
            print(f"  R1 preserved: {f.name}  ({sz_mb:.1f}MB)")
        shutil.rmtree(tmp_dl)
        r1_preserved = True
    except Exception as e:
        msg = str(e)[:300]
        print(f"  [WARN] R1 download FAILED: {type(e).__name__}: {msg}")
        print(f"  → R2 upload will proceed, but R1 ckpts will only be in OLD version")
        print(f"     (download by pinning version=1 to access R1 files)")

    print(f"\\n  R1 preserved: {n_r1_preserved} files, {r1_mb:.1f}MB" if r1_preserved
          else "  R1 preserve: SKIPPED (will be in old version only)")

    # Final staging summary
    all_staged = sorted([f.name for f in td.iterdir() if f.is_file() and f.name != "dataset-metadata.json"])
    total_mb = sum((td / fn).stat().st_size for fn in all_staged) / 1e6
    print(f"\\n  Final staging: {len(all_staged)} files, {total_mb:.1f}MB total")
    for fn in all_staged:
        sz_mb = (td / fn).stat().st_size / 1e6
        print(f"    {fn}  ({sz_mb:.1f}MB)")

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
        if r1_preserved:
            print(f"     R1 + R2 both visible in latest version")
        else:
            print(f"     R2 only in latest version (R1 in older version, use version pin)")
    else:
        print(f"\\n[FAIL] Upload failed. Last error:")
        print(f"  {type(last_err).__name__}: {str(last_err)[:500]}")
        print(f"\\nFiles remain at {OUT_DIR}, manual upload via Kaggle UI possible")
'''


def make_r2_upload_cell(slug, title, targets_r2, version_notes_expr):
    targets_repr = "[\n" + ",\n".join(f'    ({repr(s)}, {repr(d)})' for s, d in targets_r2) + ",\n]"
    src = (ROBUST_R2_UPLOAD
           .replace("__SLUG__", slug)
           .replace("__TITLE__", title)
           .replace("__TARGETS_R2__", targets_repr)
           .replace("__VERSION_NOTES__", version_notes_expr))
    return src


# Apply to exp082 R2
P = Path("experiment/exp082/notebook/nb_train_r2.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))
cell_map = {c.get("id"): (i, c) for i, c in enumerate(nb["cells"])}

i, c = cell_map["upload_state"]
new_src = make_r2_upload_cell(
    slug="birdclef2026-exp082-weights",
    title="birdclef2026 exp082 weights",
    targets_r2=[
        ("ckpt_best_ns22.pth",  "r2_ckpt_best_ns22.pth"),
        ("ckpt_best_macro.pth", "r2_ckpt_best_macro.pth"),
        ("ckpt_latest.pth",     "r2_ckpt_latest.pth"),
        ("history.json",        "r2_history.json"),
    ],
    version_notes_expr='f"R2 best_ns22={best_ns22:.4f}, best_macro={best_macro:.4f}"',
)
c["source"] = new_src.splitlines(keepends=True)
P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"OK Fixed exp082 R2 upload cell — preserve R1 + robust pattern")

# Syntax check
import ast
nb = json.loads(P.read_text(encoding="utf-8"))
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
        print(f"  SyntaxError cell {c.get('id','?')}: {e}")
print(f"\nSyntax: {n_ok} OK, {n_err} ERRORS")
