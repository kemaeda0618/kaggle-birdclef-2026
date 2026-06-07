"""Expand labels.csv search with auto-discovery via glob."""
import json
from pathlib import Path

NB = Path(__file__).with_name("nb_train_ali.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

OLD = '''    # Find Perch labels.csv (alongside perch ONNX file in same dataset)
    _bc_labels_candidates = [
        LOCAL_DATA / "perch-onnx" / "labels.csv",
        LOCAL_DATA / "perch-onnx" / "assets" / "labels.csv",
        Path("/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/labels.csv"),
        Path("/kaggle/input/rishikeshjani/perch-onnx-for-birdclef-2026/labels.csv"),
    ]
    _bc_labels_path = next((p for p in _bc_labels_candidates if p.exists()), None)
    assert _bc_labels_path is not None, f"Perch labels.csv not found in: {_bc_labels_candidates}"'''

NEW = '''    # Find Perch labels.csv — many possible locations (TF SavedModel assets/, ONNX dir, etc.)
    import glob as _glob
    _bc_labels_candidates = [
        # ONNX dataset (alongside .onnx)
        LOCAL_DATA / "perch-onnx" / "labels.csv",
        LOCAL_DATA / "perch-onnx" / "assets" / "labels.csv",
        # Perch v2 TF SavedModel — assets/labels.csv is standard location
        LOCAL_DATA / "perch_v2_cpu" / "assets" / "labels.csv",
        LOCAL_DATA / "perch_v2_cpu" / "1" / "assets" / "labels.csv",
        LOCAL_DATA / "perch_v2" / "assets" / "labels.csv",
        LOCAL_DATA / "perch_v2" / "1" / "assets" / "labels.csv",
        LOCAL_DATA / "perch-v2" / "assets" / "labels.csv",
        LOCAL_DATA / "google_perch_v2" / "assets" / "labels.csv",
        LOCAL_DATA / "google" / "perch_v2_cpu" / "1" / "assets" / "labels.csv",
        LOCAL_DATA / "bird-vocalization-classifier" / "1" / "assets" / "labels.csv",
        # Kaggle environments
        Path("/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/labels.csv"),
        Path("/kaggle/input/rishikeshjani/perch-onnx-for-birdclef-2026/labels.csv"),
        Path("/kaggle/input/perch-v2/1/assets/labels.csv"),
        Path("/kaggle/input/bird-vocalization-classifier-tensorflow2-perch_v2_cpu/1/assets/labels.csv"),
    ]
    _bc_labels_path = next((p for p in _bc_labels_candidates if p.exists()), None)

    # If not found in candidates, auto-discover via glob (search Drive + common locations)
    if _bc_labels_path is None:
        print("[ALI KD] labels.csv not in candidate paths, scanning Drive...")
        _search_roots = [str(LOCAL_DATA), "/content/data", "/content/drive/MyDrive/kaggle"]
        _found = []
        for _root in _search_roots:
            if Path(_root).exists():
                _found.extend(_glob.glob(f"{_root}/**/labels.csv", recursive=True))
        if _found:
            # Prefer paths containing "perch"
            _perch_paths = [p for p in _found if "perch" in p.lower()]
            _bc_labels_path = Path(_perch_paths[0] if _perch_paths else _found[0])
            print(f"[ALI KD] Auto-discovered: {_bc_labels_path}")
        else:
            print("[ALI KD] No labels.csv anywhere. Showing perch dirs:")
            for _root in _search_roots:
                if Path(_root).exists():
                    for _d in Path(_root).rglob("*perch*"):
                        if _d.is_dir():
                            print(f"  {_d}")

    assert _bc_labels_path is not None, (
        f"Perch labels.csv not found. Searched: {len(_bc_labels_candidates)} candidates + glob.\\n"
        f"Please ensure labels.csv exists. It is part of Perch v2 SavedModel "
        f"(`google/bird-vocalization-classifier` Kaggle model, under `assets/labels.csv`) "
        f"or perch-onnx Kaggle dataset (`rishikeshjani/perch-onnx-for-birdclef-2026`)."
    )'''

found = False
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if OLD in src:
        new_src = src.replace(OLD, NEW)
        c["source"] = new_src.splitlines(keepends=True)
        found = True
        print("[OK] Expanded labels.csv search with auto-discovery")
        break

if not found:
    print("[MISS] Old pattern not found")

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
