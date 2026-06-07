"""Fix lbl_to_bc.csv loading issue.

Problem: lbl_to_bc.csv doesn't exist as a standalone file.
Solution: Build mapping dynamically from taxonomy.csv + Perch labels.csv (in perch-onnx dataset).

Path: labels.csv is alongside perch_v2_no_dft.onnx in the perch-onnx dataset.
"""
import json, io, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path(__file__).with_name("nb_train_ali.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

# Replace the old (broken) lbl_to_bc.csv loading code with dynamic mapping
OLD = '''# === exp094: Mapped/Unmapped species indices (for Ali KL distillation) ===
if USE_ALI_KD:
    _lbl2bc_path_candidates = [
        PERCH_META_DIR / "lbl_to_bc.csv",
        LOCAL_DATA / "perch-meta-bc2026" / "lbl_to_bc.csv",
    ]
    _lbl2bc_path = next((p for p in _lbl2bc_path_candidates if p.exists()), None)
    assert _lbl2bc_path is not None, f"lbl_to_bc.csv not found in: {_lbl2bc_path_candidates}"
    print(f"[ALI KD] lbl_to_bc.csv: {_lbl2bc_path}")
    _lbl2bc_df = pd.read_csv(_lbl2bc_path)
    # Expect columns: primary_label, bc_index (or label, bc_idx)
    if "primary_label" in _lbl2bc_df.columns and "bc_index" in _lbl2bc_df.columns:
        _lbl2bc = _lbl2bc_df.set_index("primary_label")["bc_index"]
    else:
        _lbl2bc = _lbl2bc_df.set_index(_lbl2bc_df.columns[0])[_lbl2bc_df.columns[1]]
    NO_LABEL = -1
    # PRIMARY_LABELS will be set later (after taxonomy load) - we build BC_INDICES then
    print(f"[ALI KD] T_DISTILL={T_DISTILL}, ALPHA_LOGIT={ALPHA_LOGIT}")'''

NEW = '''# === exp094: Mapped/Unmapped species indices (for Ali KL distillation) ===
if USE_ALI_KD:
    # Find Perch labels.csv (alongside perch ONNX file in same dataset)
    _bc_labels_candidates = [
        LOCAL_DATA / "perch-onnx" / "labels.csv",
        LOCAL_DATA / "perch-onnx" / "assets" / "labels.csv",
        Path("/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/labels.csv"),
        Path("/kaggle/input/rishikeshjani/perch-onnx-for-birdclef-2026/labels.csv"),
    ]
    _bc_labels_path = next((p for p in _bc_labels_candidates if p.exists()), None)
    assert _bc_labels_path is not None, f"Perch labels.csv not found in: {_bc_labels_candidates}"
    print(f"[ALI KD] Perch labels.csv: {_bc_labels_path}")
    NO_LABEL = -1
    # Mapping will be built after taxonomy load (using scientific_name merge)
    print(f"[ALI KD] T_DISTILL={T_DISTILL}, ALPHA_LOGIT={ALPHA_LOGIT}")'''

# Also need to update the BC_INDICES construction (which expects _lbl2bc)
OLD_BC = '''# === exp094: Build BC_INDICES / MAPPED_POS after PRIMARY_LABELS is defined ===
if USE_ALI_KD:
    BC_INDICES = np.array([
        int(_lbl2bc.loc[c]) if c in _lbl2bc.index else NO_LABEL
        for c in PRIMARY_LABELS
    ], dtype=np.int32)
    MAPPED_MASK = BC_INDICES != NO_LABEL
    MAPPED_POS_NP = np.where(MAPPED_MASK)[0].astype(np.int64)
    MAPPED_BC_IDX_NP = BC_INDICES[MAPPED_MASK].astype(np.int64)
    MAPPED_POS = torch.from_numpy(MAPPED_POS_NP).long().to(device)
    MAPPED_BC_IDX = torch.from_numpy(MAPPED_BC_IDX_NP).long().to(device)
    print(f"[ALI KD] Mapped: {MAPPED_MASK.sum()} / {NUM_CLASSES} species have a Perch logit")
    print(f"[ALI KD] MAPPED_POS shape: {MAPPED_POS.shape}, MAPPED_BC_IDX shape: {MAPPED_BC_IDX.shape}")'''

NEW_BC = '''# === exp094: Build BC_INDICES / MAPPED_POS after PRIMARY_LABELS + taxonomy are defined ===
if USE_ALI_KD:
    # Load Perch labels.csv (its order = Perch's BC index)
    _bc_labels = (pd.read_csv(_bc_labels_path)
                  .reset_index()
                  .rename(columns={"index": "bc_index"}))
    # Find scientific_name column (might be "inat2024_fsd50k" or "scientific_name")
    _sci_col = None
    for _c in ["scientific_name", "inat2024_fsd50k", "name", "species"]:
        if _c in _bc_labels.columns:
            _sci_col = _c
            break
    assert _sci_col is not None, f"scientific_name column not found in {_bc_labels.columns.tolist()}"
    if _sci_col != "scientific_name":
        _bc_labels = _bc_labels.rename(columns={_sci_col: "scientific_name"})

    # Load BC2026 taxonomy (has primary_label + scientific_name for 234 species)
    _taxo = pd.read_csv(TAXO_PATH)
    # Merge on scientific_name (left join keeps all 234 species)
    _mapping = _taxo.merge(_bc_labels[["bc_index", "scientific_name"]],
                            on="scientific_name", how="left")
    _mapping["bc_index"] = _mapping["bc_index"].fillna(NO_LABEL).astype(int)
    _lbl2bc = _mapping.set_index("primary_label")["bc_index"]

    BC_INDICES = np.array([
        int(_lbl2bc.loc[c]) if c in _lbl2bc.index else NO_LABEL
        for c in PRIMARY_LABELS
    ], dtype=np.int32)
    MAPPED_MASK = BC_INDICES != NO_LABEL
    MAPPED_POS_NP = np.where(MAPPED_MASK)[0].astype(np.int64)
    MAPPED_BC_IDX_NP = BC_INDICES[MAPPED_MASK].astype(np.int64)
    MAPPED_POS = torch.from_numpy(MAPPED_POS_NP).long().to(device)
    MAPPED_BC_IDX = torch.from_numpy(MAPPED_BC_IDX_NP).long().to(device)
    print(f"[ALI KD] Mapped: {MAPPED_MASK.sum()} / {NUM_CLASSES} species have a Perch logit")
    print(f"[ALI KD] MAPPED_POS shape: {MAPPED_POS.shape}, MAPPED_BC_IDX shape: {MAPPED_BC_IDX.shape}")'''

patches = [(OLD, NEW), (OLD_BC, NEW_BC)]

print(f"Applying {len(patches)} patches...")
for old, new in patches:
    applied = False
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if old in src:
            new_src = src.replace(old, new)
            c["source"] = new_src.splitlines(keepends=True)
            applied = True
            print(f"  [OK] patch applied ({len(old)} chars)")
            break
    if not applied:
        print(f"  [MISS] {old[:100]!r}")

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nSaved: {NB}")

# Verify
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
print("\n=== Verify ===")
checks = [
    ("lbl_to_bc.csv reference REMOVED", "lbl_to_bc.csv" not in all_src),
    ("Perch labels.csv path candidates", '_bc_labels_candidates' in all_src),
    ("Dynamic mapping via merge", "_taxo.merge(_bc_labels" in all_src),
    ("scientific_name column logic", '_sci_col' in all_src),
    ("BC_INDICES construction", "BC_INDICES = np.array(" in all_src),
    ("MAPPED_POS torch tensor", "MAPPED_POS = torch.from_numpy" in all_src),
    # Sanity
    ("USE_ALI_KD intact", "USE_ALI_KD     = True" in all_src),
    ("T_DISTILL = 1.0 intact", "T_DISTILL      = 1.0" in all_src),
    ("ALPHA_LOGIT = 0.1 intact", "ALPHA_LOGIT    = 0.1" in all_src),
]
for name, ok in checks:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")

# Compile
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
