"""Standalone Kaggle Dataset upload for exp102 ckpts.

Robust to:
- dataset_list_files 403 (skip the check, try create_version directly)
- partial upload (fold 0/1 only OK; fold 2 added later via new version)
- KGAT_ token auth via ~/.kaggle/kaggle.json key field

Usage (Colab):
  # mount drive first
  !python _upload_exp102_standalone.py --src /content/drive/MyDrive/kaggle/birdclef2026/exp102

Usage (local, with files downloaded):
  python _upload_exp102_standalone.py --src ./exp102_ckpts

Files searched (in --src):
  fold{0,1,2}/r3/ckpt_best_ns22.pth
  fold{0,1,2}/r3/ckpt_best_macro.pth
  fold{0,1,2}/r3/history.json
Renamed to: r3_fold{k}_<original_name>.<ext>
"""
import argparse, io, json, os, shutil, sys, tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Root path containing fold{k}/r3/ ckpt dirs")
    p.add_argument("--folds", default="0,1,2", help="Comma-separated fold indices to include")
    p.add_argument("--user", default="maekeso")
    p.add_argument("--slug", default="birdclef2026-exp102-l1-3fold-r3p")
    p.add_argument("--title", default="birdclef2026 exp102 l1 3fold r3p")
    p.add_argument("--notes", default="exp102 ckpts (partial or full)")
    return p.parse_args()


def setup_auth():
    """Setup KGAT_ token via env var (avoids Basic auth 401)."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        print("[auth] KAGGLE_API_TOKEN already set")
        return
    kj = Path.home() / ".kaggle" / "kaggle.json"
    if not kj.exists():
        # Try Colab Drive default
        for cand in [Path("/content/drive/MyDrive/kaggle.json"),
                     Path("/root/.kaggle/kaggle.json")]:
            if cand.exists():
                kj.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(cand), str(kj))
                try: os.chmod(str(kj), 0o600)
                except Exception: pass
                print(f"[auth] copied {cand} -> {kj}")
                break
    assert kj.exists(), f"kaggle.json not found at {kj}"
    creds = json.loads(kj.read_text())
    key = creds["key"]
    if key.startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = key
        print(f"[auth] KGAT_ token set via env var (length={len(key)})")
    else:
        print("[auth] standard kaggle.json (non-KGAT), using SDK default")


def stage_files(src_root, folds, td):
    """Copy fold ckpts + history into td/, renamed as r3_foldK_<name>."""
    src_root = Path(src_root)
    assert src_root.exists(), f"src root not found: {src_root}"
    n_staged = 0
    found_by_fold = {}
    for k in folds:
        r3_dir = src_root / f"fold{k}" / "r3"
        files = []
        if r3_dir.exists():
            for fn in ["ckpt_best_ns22.pth", "ckpt_best_macro.pth", "history.json"]:
                src = r3_dir / fn
                if src.exists():
                    dst = td / f"r3_fold{k}_{fn}"
                    shutil.copy2(str(src), str(dst))
                    files.append(fn)
                    n_staged += 1
        found_by_fold[k] = files
    print(f"\n[stage] {n_staged} files staged from {src_root}")
    for k, fs in found_by_fold.items():
        marker = "[OK]" if fs else "[MISS]"
        print(f"  {marker} fold {k}: {fs if fs else '(no files)'}")
    return n_staged


def upload(td, user, slug, title, notes):
    """Try create_version first, fallback to create_new. Skip list_files check."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print(f"\n[auth-test] authenticated. user: {api.config_values.get('username','?')}")

    meta = {
        "title": title,
        "id": f"{user}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (td / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[meta] written {td/'dataset-metadata.json'}")

    # Try create_version (existing) first
    print("\n[upload] trying create_version (existing dataset)...")
    try:
        api.dataset_create_version(
            folder=str(td), version_notes=notes,
            dir_mode="zip", quiet=False)
        print(f"\n[OK] version uploaded")
        print(f"URL: https://www.kaggle.com/datasets/{user}/{slug}")
        return True
    except Exception as e:
        msg = str(e)[:400]
        print(f"[create_version FAILED] {type(e).__name__}: {msg}")

    # Fallback: create_new
    print("\n[upload] trying create_new...")
    try:
        api.dataset_create_new(
            folder=str(td), public=False,
            dir_mode="zip", quiet=False)
        print(f"\n[OK] new dataset created")
        print(f"URL: https://www.kaggle.com/datasets/{user}/{slug}")
        return True
    except Exception as e:
        msg = str(e)[:400]
        print(f"[create_new FAILED] {type(e).__name__}: {msg}")
        return False


def main():
    args = parse_args()
    folds = [int(x.strip()) for x in args.folds.split(",")]
    print(f"=== exp102 standalone upload ===")
    print(f"src:   {args.src}")
    print(f"folds: {folds}")
    print(f"id:    {args.user}/{args.slug}")

    setup_auth()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        n = stage_files(args.src, folds, td)
        if n == 0:
            print("[ABORT] no files staged")
            sys.exit(1)
        ok = upload(td, args.user, args.slug, args.title, args.notes)
        sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
