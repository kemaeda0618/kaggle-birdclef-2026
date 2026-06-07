"""Colab 上で実行: exp081 R2 ckpts (Drive) を Kaggle Dataset に upload

★ コピペで Colab のセルに貼って実行してください。

前提:
- Drive に R1 ckpts: /content/drive/MyDrive/kaggle/birdclef2026/output/exp081/r1/
- Drive に R2 ckpts: /content/drive/MyDrive/kaggle/birdclef2026/output/exp081/r2/
- Drive に kaggle.json
- Drive 既にマウント済 or 自動マウント

Output:
- maekeso/birdclef2026-exp081-weights の新 version
  - r1/{ckpt_*.pth, history.json}  ← Drive R1
  - r2/{ckpt_*.pth, history.json}  ← Drive R2
"""
import os, json, shutil, tempfile, time
from pathlib import Path

# === 1. Drive mount ===
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except ImportError:
    print("[INFO] Not on Colab, assuming Drive paths are accessible")

# === 2. Paths ===
DRIVE_INPUT_DIR = Path("/content/drive/MyDrive/kaggle/birdclef2026")
DRIVE_R1_DIR = DRIVE_INPUT_DIR / "output" / "exp081" / "r1"
DRIVE_R2_DIR = DRIVE_INPUT_DIR / "output" / "exp081" / "r2"

assert DRIVE_R1_DIR.exists(), f"R1 dir missing: {DRIVE_R1_DIR}"
assert DRIVE_R2_DIR.exists(), f"R2 dir missing: {DRIVE_R2_DIR}"

print(f"R1 dir: {DRIVE_R1_DIR}")
for f in sorted(DRIVE_R1_DIR.iterdir()):
    if f.is_file():
        print(f"  {f.name} ({f.stat().st_size/1e6:.1f}MB)")
print(f"\nR2 dir: {DRIVE_R2_DIR}")
for f in sorted(DRIVE_R2_DIR.iterdir()):
    if f.is_file():
        print(f"  {f.name} ({f.stat().st_size/1e6:.1f}MB)")

# === 3. Kaggle auth ===
KJ_CANDIDATES = [
    DRIVE_INPUT_DIR / "kaggle.json",
    Path("/content/drive/MyDrive/kaggle.json"),
    Path.home() / ".kaggle" / "kaggle.json",
]
KJ = next((p for p in KJ_CANDIDATES if p.exists()), None)
assert KJ is not None, f"kaggle.json not found in {[str(p) for p in KJ_CANDIDATES]}"

KAGGLE_CFG = Path.home() / ".kaggle"
KAGGLE_CFG.mkdir(parents=True, exist_ok=True)
if KJ != KAGGLE_CFG / "kaggle.json":
    shutil.copy(str(KJ), str(KAGGLE_CFG / "kaggle.json"))
os.chmod(str(KAGGLE_CFG / "kaggle.json"), 0o600)
creds = json.loads(KJ.read_text())
if creds.get("key", "").startswith("KGAT_"):
    os.environ["KAGGLE_API_TOKEN"] = creds["key"]
print(f"\nkaggle.json: {KJ}")

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
print("Kaggle authenticated OK")

# === 4. Upload (r1/ + r2/ folder structure) ===
USER = "maekeso"
SLUG = "birdclef2026-exp081-weights"
TITLE = "birdclef2026 exp081 weights"

TARGETS_R1 = ["ckpt_best_ns22.pth", "ckpt_best_macro.pth", "ckpt_latest.pth", "history.json"]
TARGETS_R2 = ["ckpt_best_ns22.pth", "ckpt_best_macro.pth", "ckpt_latest.pth", "history.json"]

print(f"\nUpload target: {USER}/{SLUG}")

with tempfile.TemporaryDirectory() as td:
    td = Path(td)

    # [1/3] Stage R1
    r1_dst = td / "r1"
    r1_dst.mkdir(parents=True, exist_ok=True)
    print(f"\n[1/3] Staging R1")
    n_r1 = 0; r1_mb = 0.0
    for fn in TARGETS_R1:
        src = DRIVE_R1_DIR / fn
        if not src.exists():
            print(f"  skip: r1/{fn}"); continue
        shutil.copy2(str(src), str(r1_dst / fn))
        sz_mb = src.stat().st_size / 1e6
        r1_mb += sz_mb; n_r1 += 1
        print(f"  R1 staged: r1/{fn} ({sz_mb:.1f}MB)")
    print(f"  R1 TOTAL: {n_r1} files, {r1_mb:.1f}MB")

    # [2/3] Stage R2
    r2_dst = td / "r2"
    r2_dst.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/3] Staging R2")
    n_r2 = 0; r2_mb = 0.0
    for fn in TARGETS_R2:
        src = DRIVE_R2_DIR / fn
        if not src.exists():
            print(f"  skip: r2/{fn}"); continue
        shutil.copy2(str(src), str(r2_dst / fn))
        sz_mb = src.stat().st_size / 1e6
        r2_mb += sz_mb; n_r2 += 1
        print(f"  R2 staged: r2/{fn} ({sz_mb:.1f}MB)")
    assert n_r2 > 0, "No R2 files!"
    print(f"  R2 TOTAL: {n_r2} files, {r2_mb:.1f}MB")

    # Final staging summary
    print(f"\n  Final staging tree:")
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

    version_notes = "R1 + R2 manual upload (r1/+r2/ folders)"
    uploaded = False
    last_err = None

    print(f"\n[3/3] Uploading...")
    try:
        t0 = time.time()
        api.dataset_create_version(folder=str(td),
                                    version_notes=version_notes,
                                    dir_mode="zip", quiet=False)
        print(f"\n  OK Uploaded (new version, {time.time()-t0:.0f}s)")
        uploaded = True
    except Exception as e:
        msg = str(e)[:600]
        print(f"  create_version FAILED: {type(e).__name__}: {msg}")
        last_err = e
        if "not found" in msg.lower() or "404" in msg or "Could not find" in msg:
            print(f"\n  Try create_new...")
            try:
                t0 = time.time()
                api.dataset_create_new(folder=str(td), public=False, dir_mode="zip", quiet=False)
                print(f"  OK Created ({time.time()-t0:.0f}s)")
                uploaded = True
            except Exception as e2:
                msg2 = str(e2)[:600]
                print(f"  create_new FAILED: {type(e2).__name__}: {msg2}")
                last_err = e2

    if uploaded:
        print(f"\n[OK] Dataset URL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
        print(f"     Structure: r1/{{ckpt_*.pth, history.json}}, r2/{{ckpt_*.pth, history.json}}")
    else:
        print(f"\n[FAIL] {type(last_err).__name__}: {str(last_err)[:500]}")
