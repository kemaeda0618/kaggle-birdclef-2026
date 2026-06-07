"""Add auto-download fallback for labels.csv via kaggle API.

labels.csv is confirmed in rishikeshjani/perch-onnx-for-birdclef-2026 (312KB).
User's Drive has the ONNX but not labels.csv → auto-download if missing.
"""
import json
from pathlib import Path

NB = Path(__file__).with_name("nb_train_ali.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

# Insert auto-download BEFORE the assertion
OLD = '''    assert _bc_labels_path is not None, (
        f"Perch labels.csv not found. Searched: {len(_bc_labels_candidates)} candidates + glob.\\n"
        f"Please ensure labels.csv exists. It is part of Perch v2 SavedModel "
        f"(`google/bird-vocalization-classifier` Kaggle model, under `assets/labels.csv`) "
        f"or perch-onnx Kaggle dataset (`rishikeshjani/perch-onnx-for-birdclef-2026`)."
    )'''

NEW = '''    # === Auto-download labels.csv from Kaggle if missing ===
    if _bc_labels_path is None:
        print("[ALI KD] Attempting auto-download from Kaggle dataset...")
        _target_dir = LOCAL_DATA / "perch-onnx"
        _target_dir.mkdir(parents=True, exist_ok=True)
        _target_path = _target_dir / "labels.csv"
        try:
            # Try via Kaggle API (needs kaggle.json credentials)
            import subprocess as _sp
            _result = _sp.run(
                ["kaggle", "datasets", "download",
                 "-d", "rishikeshjani/perch-onnx-for-birdclef-2026",
                 "-f", "labels.csv",
                 "-p", str(_target_dir),
                 "--unzip"],
                capture_output=True, text=True, timeout=60,
            )
            print(f"[ALI KD] kaggle cli stdout: {_result.stdout[:300]}")
            if _result.returncode != 0:
                print(f"[ALI KD] kaggle cli stderr: {_result.stderr[:300]}")
            if _target_path.exists():
                _bc_labels_path = _target_path
                print(f"[ALI KD] Downloaded successfully: {_bc_labels_path}")
        except Exception as _e:
            print(f"[ALI KD] kaggle cli failed: {_e}")

        # Fallback: direct opendatasets / urllib (no kaggle.json needed)
        if _bc_labels_path is None:
            try:
                import urllib.request as _ur
                # Public Kaggle dataset direct URL (may need API token)
                _url = "https://www.kaggle.com/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/download?datasetVersionNumber=latest&fileName=labels.csv"
                print(f"[ALI KD] Trying urllib download from: {_url}")
                _ur.urlretrieve(_url, _target_path)
                if _target_path.exists() and _target_path.stat().st_size > 1000:
                    _bc_labels_path = _target_path
                    print(f"[ALI KD] Downloaded via urllib: {_bc_labels_path}")
                else:
                    print("[ALI KD] urllib download produced empty/invalid file")
                    if _target_path.exists(): _target_path.unlink()
            except Exception as _e:
                print(f"[ALI KD] urllib failed: {_e}")

    assert _bc_labels_path is not None, (
        f"Perch labels.csv not found and auto-download failed.\\n"
        f"Manual fix: run in Colab cell:\\n"
        f"  !pip install -q kaggle\\n"
        f"  !mkdir -p ~/.kaggle && cp /content/drive/MyDrive/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json\\n"
        f"  !kaggle datasets download -d rishikeshjani/perch-onnx-for-birdclef-2026 -f labels.csv -p /content/data/perch-onnx/ --unzip\\n"
        f"Then re-run this cell."
    )'''

found = False
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if OLD in src:
        new_src = src.replace(OLD, NEW)
        c["source"] = new_src.splitlines(keepends=True)
        found = True
        print("[OK] Auto-download fallback added")
        break

if not found:
    print("[MISS] OLD pattern not found")

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {NB}")

# Compile check
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
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
        print(f"  [FAIL] cell {i}: {e}")
print(f"Result: {n_ok} OK, {n_err} ERRORS")
