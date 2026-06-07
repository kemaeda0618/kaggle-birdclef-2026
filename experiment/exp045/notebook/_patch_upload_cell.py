"""Replace upload cell with robust try-version-first + strong verification pattern."""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp045\notebook\nb_train_r3_l1_filtered.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

NEW_UPLOAD = '''# ============================================================
# Cell 10: Upload R3 ckpts to Kaggle Dataset (robust try-version-first pattern)
# ============================================================
import tempfile

KJ = next((p for p in [DRIVE_INPUT_DIR / "kaggle.json",
                        Path("/content/drive/MyDrive/kaggle.json")] if p.exists()), None)
if KJ is not None:
    KAGGLE_CFG = Path.home() / ".kaggle"
    KAGGLE_CFG.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(KJ), str(KAGGLE_CFG / "kaggle.json"))
    os.chmod(str(KAGGLE_CFG / "kaggle.json"), 0o600)
    creds = json.loads(KJ.read_text())
    if creds.get("key", "").startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = creds["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
print("kaggle re-auth OK")

USER  = "maekeso"
SLUG  = "birdclef2026-exp045-l1-filtered"
TITLE = "birdclef2026 exp045 l1 filtered ablation"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    n_staged = 0
    for fold_k in FOLDS:
        # ★ exp045: filter ablation R3 ckpt のみ upload
        DRIVE_R3_DIR = DRIVE_EXP_DIR / f"fold{fold_k}" / "r3"
        for fn in ["ckpt_best_ns22.pth", "ckpt_best_macro.pth", "history.json"]:
            src = DRIVE_R3_DIR / fn
            if src.exists():
                dst = td / f"r3_fold{fold_k}_{fn}"
                shutil.copy2(str(src), str(dst))
                n_staged += 1
                print(f"  staged: {dst.name} ({src.stat().st_size/1e6:.1f} MB)")
    print(f"Staged {n_staged} files total")
    assert n_staged > 0, "No files to upload — check DRIVE_EXP_DIR / fold paths"

    meta = {"title": TITLE, "id": f"{USER}/{SLUG}",
            "licenses": [{"name": "CC0-1.0"}]}
    (td / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    summary = " ; ".join([f"f{k}_r3={v.get('r2_best_ns22','?')}" for k, v in sorted(fold_results.items())])
    version_notes = f"exp045 l1 chunk-filter ablation (filter={CHUNK_FILTER_MAX_PROB}) | {summary}"[:498]

    # ★ robust: try-version-first, fallback to create_new on failure (no pre-check!)
    # pre-check (dataset_view/list_files) is unreliable for auth quirks/rate limits
    upload_method = None
    upload_error = None
    try:
        print(f"\\nAttempting dataset_create_version for {USER}/{SLUG} ...")
        api.dataset_create_version(folder=str(td), version_notes=version_notes,
                                    dir_mode="zip", quiet=False)
        upload_method = "version_up"
        print(f"OK uploaded as new version")
    except Exception as e:
        upload_error = str(e)[:300]
        print(f"  version_create failed: {upload_error}")
        print(f"  → fallback to dataset_create_new...")
        try:
            api.dataset_create_new(folder=str(td), public=False,
                                    dir_mode="zip", quiet=False)
            upload_method = "create_new"
            print(f"OK created new dataset")
        except Exception as e2:
            upload_method = "FAILED"
            print(f"  create_new ALSO failed: {str(e2)[:300]}")

    # ★ strong verification: file count check (catches silent fail)
    print(f"\\nVerifying upload (method={upload_method})...")
    try:
        import time
        time.sleep(5)  # give Kaggle time to register
        files_after = api.dataset_list_files(f"{USER}/{SLUG}").files
        n_uploaded = len(files_after)
        print(f"  remote files: {n_uploaded}")
        for f in files_after[:10]:
            print(f"    - {f.name}")
        if n_uploaded < n_staged:
            print(f"  ⚠ WARN: only {n_uploaded} files visible, staged {n_staged}")
            print(f"  ⚠ Possible silent fail. Re-run upload cell or check Kaggle UI")
        else:
            print(f"  ✓ verified: {n_uploaded} >= {n_staged} (staged)")
    except Exception as e:
        print(f"  [VERIFY FAIL] {str(e)[:300]}")
        print(f"  → Cannot list files. Check Kaggle UI: https://www.kaggle.com/datasets/{USER}/{SLUG}")

    print(f"\\nURL: https://www.kaggle.com/datasets/{USER}/{SLUG}")
    print(f"Upload method: {upload_method}")
    if upload_error:
        print(f"Initial error (recovered if method != FAILED): {upload_error}")
'''

for c in nb["cells"]:
    if c.get("id") == "upload":
        c["source"] = NEW_UPLOAD.splitlines(keepends=True)
        print("[upload] OK replaced with robust pattern")
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB}")
