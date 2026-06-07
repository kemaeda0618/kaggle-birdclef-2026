"""Apply same Drive R1 + local R2 + folder structure to exp081 R2 train NB."""
import json
from pathlib import Path

R2_UPLOAD_DRIVE = '''# ============================================================
# Cell: Upload R2 weights to Kaggle Dataset (R1 from Drive + R2 from local)
# ============================================================
import tempfile, time
from kaggle.api.kaggle_api_extended import KaggleApi

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

DRIVE_R1_DIR = DRIVE_OUTPUT_DIR.parent / "r1"

TARGETS_R1 = [
    "ckpt_best_ns22.pth", "ckpt_best_macro.pth", "ckpt_latest.pth", "history.json",
]
TARGETS_R2 = [
    "ckpt_best_ns22.pth", "ckpt_best_macro.pth", "ckpt_latest.pth", "history.json",
]

print(f"\\nUpload target: {USER}/{SLUG}")
print(f"  R1 source: {DRIVE_R1_DIR}")
print(f"  R2 source: {OUT_DIR}")

with tempfile.TemporaryDirectory() as td:
    td = Path(td)

    # [1/3] Stage R1 (Drive → temp/r1/)
    r1_dst = td / "r1"
    r1_dst.mkdir(parents=True, exist_ok=True)
    print(f"\\n[1/3] Staging R1 files")
    n_r1 = 0; r1_mb = 0.0
    if not DRIVE_R1_DIR.exists():
        print(f"  [WARN] DRIVE_R1_DIR not found: {DRIVE_R1_DIR}")
        print(f"  → R1 will NOT be preserved.")
    else:
        for fn in TARGETS_R1:
            src = DRIVE_R1_DIR / fn
            if not src.exists():
                print(f"  skip (missing on Drive): r1/{fn}")
                continue
            shutil.copy2(str(src), str(r1_dst / fn))
            sz_mb = src.stat().st_size / 1e6
            r1_mb += sz_mb; n_r1 += 1
            print(f"  R1 staged: r1/{fn}  ({sz_mb:.1f}MB)")
    print(f"  R1 TOTAL: {n_r1} files, {r1_mb:.1f}MB")

    # [2/3] Stage R2 (local → temp/r2/)
    r2_dst = td / "r2"
    r2_dst.mkdir(parents=True, exist_ok=True)
    print(f"\\n[2/3] Staging R2 files")
    n_r2 = 0; r2_mb = 0.0
    for fn in TARGETS_R2:
        src = OUT_DIR / fn
        if not src.exists():
            print(f"  skip (missing local): r2/{fn}")
            continue
        shutil.copy2(str(src), str(r2_dst / fn))
        sz_mb = src.stat().st_size / 1e6
        r2_mb += sz_mb; n_r2 += 1
        print(f"  R2 staged: r2/{fn}  ({sz_mb:.1f}MB)")
    assert n_r2 > 0, "No R2 files to upload!"
    print(f"  R2 TOTAL: {n_r2} files, {r2_mb:.1f}MB")

    print(f"\\n  Final staging tree:")
    for sub in ["r1", "r2"]:
        sub_dir = td / sub
        if sub_dir.exists():
            for f in sorted(sub_dir.iterdir()):
                if f.is_file():
                    sz_mb = f.stat().st_size / 1e6
                    print(f"    {sub}/{f.name}  ({sz_mb:.1f}MB)")

    # [3/3] Upload
    meta = {
        "title": TITLE,
        "id": f"{USER}/{SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    version_notes = __VERSION_NOTES__
    uploaded = False
    last_err = None

    print(f"\\n[3/3] Uploading...")
    try:
        t0 = time.time()
        api.dataset_create_version(folder=str(td),
                                    version_notes=version_notes,
                                    dir_mode="zip", quiet=False)
        print(f"\\n  OK Uploaded (new version, {time.time()-t0:.0f}s)")
        uploaded = True
    except Exception as e:
        msg = str(e)[:600]
        print(f"  create_version FAILED: type={type(e).__name__}: {msg}")
        last_err = e
        if "not found" in msg.lower() or "404" in msg or "Could not find" in msg:
            print(f"\\n  Try create_new...")
            try:
                t0 = time.time()
                api.dataset_create_new(folder=str(td), public=False,
                                       dir_mode="zip", quiet=False)
                print(f"  OK Created ({time.time()-t0:.0f}s)")
                uploaded = True
            except Exception as e2:
                msg2 = str(e2)[:600]
                print(f"  create_new FAILED: type={type(e2).__name__}: {msg2}")
                last_err = e2

    if uploaded:
        print(f"\\n[OK] Dataset URL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
        print(f"     Structure: r1/{{ckpt_*.pth, history.json}}, r2/{{ckpt_*.pth, history.json}}")
    else:
        print(f"\\n[FAIL] {type(last_err).__name__}: {str(last_err)[:500]}")
'''


def make_cell(slug, title, version_notes_expr):
    return (R2_UPLOAD_DRIVE
            .replace("__SLUG__", slug)
            .replace("__TITLE__", title)
            .replace("__VERSION_NOTES__", version_notes_expr))


P = Path("experiment/exp081/notebook/nb_train_r2.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

# Find upload cell (id might be upload_state or upload)
cell_map = {c.get("id"): (i, c) for i, c in enumerate(nb["cells"])}

upload_id = None
for cand in ["upload_state", "upload"]:
    if cand in cell_map:
        upload_id = cand
        break

# Also search by content
if upload_id is None:
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if "dataset_create_version" in src and "TARGETS" in src:
            upload_id = c.get("id")
            break

if upload_id is None:
    print("[ERR] Could not find upload cell in exp081 R2 NB")
else:
    print(f"Found upload cell: id={upload_id}")
    i, c = cell_map.get(upload_id) if upload_id in cell_map else (None, None)
    if c is None:
        # Find by id again from list
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("id") == upload_id:
                c = cell; i = idx; break

    if c is not None:
        new_src = make_cell(
            slug="birdclef2026-exp081-weights",
            title="birdclef2026 exp081 weights",
            version_notes_expr='f"R2 best_ns22={best_ns22:.4f}, best_macro={best_macro:.4f} (r1/ + r2/ folders)"',
        )
        c["source"] = new_src.splitlines(keepends=True)
        P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"OK Fixed exp081 R2 upload cell")

# Also check for orphan exp017 hardcoded cell
nb = json.loads(P.read_text(encoding="utf-8"))
to_delete = []
for i, c in enumerate(nb["cells"]):
    cid = c.get("id", "")
    src = "".join(c.get("source", []))
    if c.get("cell_type") != "code":
        continue
    if 'output" / "exp017"' in src and 'birdclef2026-exp017-weights' in src:
        print(f"  Cell {i} id={cid} contains exp017 hardcoded → DELETE")
        to_delete.append(i)
for i in reversed(to_delete):
    del nb["cells"][i]
P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

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
        ast.parse(clean); n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  SyntaxError cell {c.get('id','?')}: {e}")
print(f"exp081 R2 syntax: {n_ok} OK, {n_err} ERRORS")
