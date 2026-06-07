"""exp067: exp048 + Cross-Dialect (Pantanal proximity) post-process calibration.

戦略:
  exp048 (LB 0.950) の blend 後、TopN PP の前に Pantanal proximity calibration 挿入
  per-species: train.csv lat/lon から Pantanal 内 ratio 計算
  blend_3d を mild boost (ALPHA=0.10)

Path: blend cell 内に calibration 段階を inject
Pantanal: 緯度 (-21.6, -16.5)、経度 (-57.6, -55.9)
"""
import json
from pathlib import Path

SRC = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp048\notebook\nb_blend_topn1.ipynb")
OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp067\notebook\nb_blend_dialect.ipynb")

nb = json.load(open(SRC, encoding="utf-8"))


def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}


# === Cross-Dialect (Pantanal proximity) calibration cell ===
# Insert BEFORE TopN PP in blend cell
dialect_src = r"""# === Cross-Dialect Pantanal proximity calibration ===
# 各 species の train.csv lat/lon から Pantanal 内 ratio を計算
# blend_3d (rank space) を per-species で boost
# Reference: arXiv:2509.22317 (Cross-Dialect Bird Species Recognition)

# Pantanal coordinates (per CLAUDE.md):
PANTANAL_LAT = (-21.6, -16.5)
PANTANAL_LON = (-57.6, -55.9)
ALPHA_DIALECT = 0.10   # mild boost (exp023 wet/dry prior 失敗を考慮、軽め)

# Load train.csv lat/lon
TRAIN_CSV_PATH = BASE / "train.csv"
if TRAIN_CSV_PATH.exists():
    _train_csv = pd.read_csv(TRAIN_CSV_PATH)
    print(f"train.csv: {len(_train_csv)} rows for dialect calibration")

    # Per-species Pantanal proximity (0-1)
    _species_factor = np.ones(len(PRIMARY_LABELS), dtype=np.float32)
    for _i, _sp in enumerate(PRIMARY_LABELS):
        _sp_rows = _train_csv[_train_csv["primary_label"] == _sp]
        if len(_sp_rows) == 0:
            _species_factor[_i] = 1.0  # neutral if no data
            continue
        _in_pantanal = (
            _sp_rows["latitude"].between(*PANTANAL_LAT) &
            _sp_rows["longitude"].between(*PANTANAL_LON)
        ).sum()
        _ratio = _in_pantanal / len(_sp_rows)
        # factor: 1.0 + ALPHA * ratio (1.0 = neutral、ratio 1.0 で +10%)
        _species_factor[_i] = 1.0 + ALPHA_DIALECT * float(_ratio)

    # Apply calibration BEFORE TopN PP
    # blend_3d already computed above; calibrate it
    print(f"  Pantanal-native species (factor > 1.05): {(_species_factor > 1.05).sum()}")
    print(f"  Neutral species (factor == 1.0): {(_species_factor == 1.0).sum()}")
    print(f"  Mean factor: {_species_factor.mean():.4f}")
    print(f"  Max factor: {_species_factor.max():.4f}")

    # Apply to probs_blend (shape: n_files, n_windows, n_classes)
    # Broadcasting: factor (n_classes,) → (1, 1, n_classes)
    _factor_3d = _species_factor[None, None, :]
    probs_blend = probs_blend * _factor_3d
    # Clip (rank space [0, 1])
    probs_blend = np.clip(probs_blend, 0.0, 1.0).astype(np.float32)
    print(f"  After dialect calibration: mean={probs_blend.mean():.4f}, max={probs_blend.max():.4f}")
else:
    print(f"  ⚠ train.csv not found, skipping dialect calibration")
"""

# Modify blend cell: insert dialect calibration after blend_3d construction, before TopN PP
for c in nb["cells"]:
    if c.get("id") == "blend":
        src = "".join(c["source"])
        # Insertion: after `probs_blend = blend_flat.reshape(probs_exp010.shape)` line
        # Before `preds_array = probs_blend.reshape(-1, N_CLASSES)` line
        marker = "preds_array = probs_blend.reshape(-1, N_CLASSES)"
        idx = src.find(marker)
        if idx < 0:
            print(f"  WARN: marker not found: {marker}")
            continue
        # Insert dialect calibration BEFORE this line
        new_src = src[:idx] + dialect_src + "\n" + src[idx:]
        c["source"] = new_src.splitlines(keepends=True)
        print(f"  dialect calibration inserted before TopN PP")
        break

# Update hdr cell
for c in nb["cells"]:
    if c.get("id") == "hdr":
        new_hdr = """# exp067: exp048 + Cross-Dialect (Pantanal proximity) calibration

**Base**: exp048 (LB 0.950)
**追加**: Pantanal proximity per-species boost (ALPHA=0.10 mild)
**Reference**: arXiv:2509.22317 (Cross-Dialect Bird Species Recognition)

**Logic**:
  per-species: train.csv lat/lon で Pantanal 内 ratio 計算
  factor = 1.0 + 0.10 * ratio (1.0=neutral、1.10=full Pantanal native)
  blend_3d (rank space) × factor で boost

**期待 LB**: 0.950-0.955 (+0.000-0.005 mild)
**Risk**: exp023 wet/dry prior 失敗 lesson、ALPHA 軽めで安全
"""
        c["source"] = new_hdr.splitlines(keepends=True)
        break

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Wrote {OUT}")
print(f"  cells: {len(nb['cells'])}")
