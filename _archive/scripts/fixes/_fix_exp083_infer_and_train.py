"""Fix:
1. exp083 R1 infer NB: switch from dataset_sources to kernel_sources (Dataset doesn't exist)
2. exp083 R1 train NB upload cell: replace with robust v3 pattern (no dataset_view)
"""
import json
from pathlib import Path
import shutil


# ============================================================
# Fix 1: exp083 R1 infer NB — kernel_sources path
# ============================================================
P = Path("experiment/exp083/notebook/nb_infer_r1.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

for c in nb["cells"]:
    if c.get("id") == "paths":
        new_src = '''# ============================================================
# Cell 2: Paths — locate competition data + R1 ckpt (kernel output)
# ============================================================
BASE = None
for p in [Path("/kaggle/input/competitions/birdclef-2026"),
          Path("/kaggle/input/birdclef-2026")]:
    if p.exists():
        BASE = p; break
assert BASE is not None, "BC2026 competition data not found"

TEST_DIR = BASE / "test_soundscapes"
TAXO_PATH = BASE / "taxonomy.csv"
SAMPLE_SUB_PATH = BASE / "sample_submission.csv"
print(f"BASE: {BASE}")
print(f"  test_soundscapes exists: {TEST_DIR.exists()}")

# Locate exp083 R1 ckpts (kernel output 経由、Dataset は未作成)
STATE_DIR = None
CANDIDATES = [
    # kernel output (R1 ckpts here)
    Path("/kaggle/input/birdclef2026-exp083-train-r1"),
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp083-train-r1"),
    # Dataset (fallback、もし将来作成された場合)
    Path("/kaggle/input/datasets/maekeso/exp083-state"),
    Path("/kaggle/input/exp083-state"),
]
for p in CANDIDATES:
    if p.exists():
        if any(p.rglob("ckpt_best_*.pth")) or any(p.rglob("ckpt_latest*.pth")):
            STATE_DIR = p; break

if STATE_DIR is None:
    for hit in Path("/kaggle/input").rglob("ckpt_best_ns22.pth"):
        STATE_DIR = hit.parent; break

assert STATE_DIR is not None, (
    "ckpt not found. Attach maekeso/birdclef2026-exp083-train-r1 as kernel_sources"
)
print(f"State dir: {STATE_DIR}")
for f in sorted(STATE_DIR.rglob("ckpt_*.pth"))[:6]:
    print(f"  {f.relative_to(STATE_DIR)!s}  {f.stat().st_size/1e6:.2f} MB")
'''
        c["source"] = new_src.splitlines(keepends=True)
        print(f"OK Fixed exp083 R1 infer NB cell 'paths' — kernel_sources path")
        break

P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# Update push script: kernel_sources instead of dataset_sources
push_path = Path("experiment/exp083/notebook/_push_nb_infer_r1.py")
push_src = '''"""Push exp083 R1 inference NB (CPU sub).

★ Uses kernel_sources (kernel output) instead of dataset_sources,
   because birdclef2026-exp083-state Dataset upload failed (api.dataset_view bug).
"""
import json, os, sys, io, tempfile, shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

if not os.environ.get("KAGGLE_API_TOKEN"):
    _kgat = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    os.environ["KAGGLE_API_TOKEN"] = _kgat

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB = Path(__file__).with_name("nb_infer_r1.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp083-infer-r1"
TITLE = "birdclef2026 exp083 infer r1"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": TITLE,
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": False,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [],
        # ★ kernel_sources: R1 ckpts は kernel output に存在
        "kernel_sources": [
            f"{USER}/birdclef2026-exp083-train-r1",
        ],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None))
    print("Version:", getattr(r, "version_number", None))
    err = getattr(r, "error", "")
    if err: print("Error:", err)
'''
push_path.write_text(push_src, encoding="utf-8")
print(f"OK Updated push script with kernel_sources")


# ============================================================
# Fix 2: exp083 R1 train NB upload cell — robust v3 pattern
# ============================================================
P = Path("experiment/exp083/notebook/nb_train_r1.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

ROBUST_R1_UPLOAD = '''# ============================================================
# Cell 12: Upload state to maekeso/exp083-state (robust v3 pattern)
# ============================================================
import tempfile, time
from kaggle.api.kaggle_api_extended import KaggleApi

# ★ Re-authenticate (long training session may have expired auth)
print("Re-authenticating Kaggle API for upload...")
KAGGLE_CFG = Path.home() / ".kaggle" / "kaggle.json"
if KAGGLE_CFG.exists():
    creds = json.loads(KAGGLE_CFG.read_text())
    if creds.get("key", "").startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = creds["key"]
api = KaggleApi(); api.authenticate()
print("  Re-auth OK")

DATASET_USER = "maekeso"
DATASET_SLUG = "exp083-state"
DATASET_TITLE = "exp083 training state"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    # Copy output artifacts
    n_copied = 0
    for f in Path("/kaggle/working").glob("*"):
        if not f.is_file(): continue
        if f.suffix not in {".pth", ".json", ".txt"}: continue
        shutil.copy(str(f), str(td / f.name))
        n_copied += 1
    print(f"Staged {n_copied} files for upload")
    for p in sorted(td.iterdir()):
        if p.is_file():
            print(f"  {p.name}  {p.stat().st_size/1e6:.2f} MB")

    meta = {
        "title": DATASET_TITLE,
        "id": f"{DATASET_USER}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    version_notes = f"epoch {trained_up_to}/{N_TOTAL_EPOCHS}"
    uploaded = False
    last_err = None

    # ★ Strategy: try create_version first (robust), fall through to create_new only on 404
    print(f"\\n[1/2] Try create_version...")
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
        if "not found" in msg.lower() or "404" in msg or "Could not find" in msg:
            print(f"\\n[2/2] Dataset doesn\\'t exist → try create_new...")
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
        print(f"\\n[OK] Dataset URL: https://www.kaggle.com/datasets/{DATASET_USER}/{DATASET_SLUG}")
    else:
        print(f"\\n[FAIL] Upload failed. Last error:")
        print(f"  {type(last_err).__name__}: {str(last_err)[:500]}")
        print(f"\\n  Files remain in /kaggle/working/ — accessible via kernel output")
        print(f"  Infer NB can use kernel_sources: 'maekeso/birdclef2026-exp083-train-r1'")
'''

for c in nb["cells"]:
    if c.get("id") == "upload_state":
        c["source"] = ROBUST_R1_UPLOAD.splitlines(keepends=True)
        print(f"OK Fixed exp083 R1 train NB cell 'upload_state' — robust v3 pattern")
        break
P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


# Syntax check
import ast
for nb_path in ["experiment/exp083/notebook/nb_infer_r1.ipynb",
                "experiment/exp083/notebook/nb_train_r1.ipynb"]:
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
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
            print(f"  SyntaxError {nb_path} cell {c.get('id','?')}: {e}")
    print(f"{nb_path}: {n_ok} OK, {n_err} ERRORS")
