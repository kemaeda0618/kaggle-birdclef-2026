"""Patch nb_infer_analysis.ipynb to fix V1 bugs:
1. EMBED_IDX bug (overwritten by spatial_embed) - perch_setup cell
2. Perch label mapping (use Perch model's assets/labels.csv instead of perch-meta) - mapping cell
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "nb_infer_analysis.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# Replacement source for `perch_setup` cell — fix EMBED_IDX logic
new_perch_setup = '''# ============================================================
# Cell 3: Perch v2 ONNX setup (GPU) - V2 FIX
# ============================================================
# Locate Perch ONNX
PERCH_PATH = None
for cand in [
    Path("/kaggle/input/datasets/tuckerarrants/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx"),
    Path("/kaggle/input/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx"),
]:
    if cand.exists():
        PERCH_PATH = cand; break
if PERCH_PATH is None:
    for hit in Path("/kaggle/input").rglob("perch_v2*.onnx"):
        PERCH_PATH = hit; break
assert PERCH_PATH is not None, "Perch ONNX not found"
print(f"Perch: {PERCH_PATH}")

providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
session = ort.InferenceSession(str(PERCH_PATH), providers=providers)
input_name = session.get_inputs()[0].name
print(f"Perch input: {input_name}, providers: {session.get_providers()}")
for i, o in enumerate(session.get_outputs()):
    print(f"  output[{i}] {o.name} shape={o.shape}")

# ★ V2 FIX: Use FIRST 2D match for embedding (output[0] = pooled 1536-d)
#   Don't overwrite with spatial_embedding (4D)
EMBED_IDX = None
LABEL_IDX = None
for i, o in enumerate(session.get_outputs()):
    shape = o.shape
    if shape and EMBED_IDX is None and len(shape) == 2 and shape[-1] == 1536:
        EMBED_IDX = i
    if shape and LABEL_IDX is None and len(shape) == 2 and shape[-1] > 5000:
        LABEL_IDX = i
print(f"EMBED_IDX={EMBED_IDX}, LABEL_IDX={LABEL_IDX}")
assert EMBED_IDX is not None and LABEL_IDX is not None

def perch_infer(audio_batch, batch_size=64):
    """Run Perch inference on audio batch.
    Returns: embeddings (N, 1536), scores (N, num_perch_classes)
    """
    N = audio_batch.shape[0]
    embeddings, scores = [], []
    for i in range(0, N, batch_size):
        batch = audio_batch[i:i+batch_size].astype(np.float32)
        out = session.run(None, {input_name: batch})
        embeddings.append(out[EMBED_IDX])
        scores.append(out[LABEL_IDX])
    return np.concatenate(embeddings), np.concatenate(scores)

# Test
dummy = np.random.randn(2, SR * 5).astype(np.float32) * 0.01
emb, sc = perch_infer(dummy)
print(f"Test: emb {emb.shape}, scores {sc.shape}")
assert emb.shape[-1] == 1536, f"Embedding wrong shape: {emb.shape}"
'''

# Replacement source for `mapping` cell — use Perch model's assets/labels.csv
new_mapping = '''# ============================================================
# Cell 4: Perch label → BC2026 mapping (from Perch v2 TF model's assets/labels.csv)
# ============================================================
# Locate Perch v2 TF model dir (provides assets/labels.csv via model_sources)
MODEL_DIR = None
for cand in [
    Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"),
    Path("/kaggle/input/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"),
]:
    if cand.exists():
        MODEL_DIR = cand; break
if MODEL_DIR is None:
    # rglob fallback for labels.csv near a Perch model
    for hit in Path("/kaggle/input").rglob("labels.csv"):
        if "perch" in str(hit).lower() or "bird-vocal" in str(hit).lower():
            MODEL_DIR = hit.parent.parent
            break
assert MODEL_DIR is not None, "Perch v2 TF model dir not found"
print(f"Perch MODEL_DIR: {MODEL_DIR}")

LABELS_CSV = MODEL_DIR / "assets" / "labels.csv"
assert LABELS_CSV.exists(), f"labels.csv not found at {LABELS_CSV}"
bc_labels = pd.read_csv(LABELS_CSV)
bc_labels = bc_labels.reset_index().rename(columns={"index": "perch_class_idx"})
print(f"Perch labels.csv: {len(bc_labels)} classes")
print(f"  columns: {list(bc_labels.columns)[:8]}")
print(f"  sample: {bc_labels.head(2).to_dict('records')}")

# Determine join key — try scientific_name first, then ebird2021
join_col = None
for c in ["scientific_name", "ebird2021", "ebird_code", "code", "label"]:
    if c in bc_labels.columns and c in taxonomy_df.columns:
        join_col = c
        break
if join_col is None:
    # Try fuzzy: find a string column in bc_labels that has overlap with taxonomy
    for c in bc_labels.columns:
        if bc_labels[c].dtype == object:
            overlap = set(bc_labels[c].dropna().astype(str)) & set(taxonomy_df["scientific_name"].dropna().astype(str))
            if len(overlap) > 50:
                # rename to scientific_name to use
                bc_labels = bc_labels.rename(columns={c: "scientific_name"})
                join_col = "scientific_name"
                break
assert join_col is not None, f"No matching column found. bc_labels: {list(bc_labels.columns)}, taxonomy: {list(taxonomy_df.columns)}"
print(f"Join column: {join_col}")

NO_LABEL = len(bc_labels)
tax_with_perch = taxonomy_df[["primary_label", "scientific_name"]].merge(
    bc_labels[["perch_class_idx", join_col]],
    on=join_col if join_col in taxonomy_df.columns else "scientific_name",
    how="left",
)
tax_with_perch["perch_class_idx"] = tax_with_perch["perch_class_idx"].fillna(NO_LABEL).astype(int)

# Build BC_INDICES in PRIMARY_LABELS order
bc_idx_map = tax_with_perch.set_index("primary_label")["perch_class_idx"].to_dict()
BC_INDICES = np.array([bc_idx_map.get(lbl, NO_LABEL) for lbl in PRIMARY_LABELS], dtype=np.int32)
MAPPED_MASK = BC_INDICES != NO_LABEL
MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)
print(f"Mapped: {MAPPED_MASK.sum()} / {N_CLASSES} species have Perch logit")

def perch_to_bc2026(perch_scores):
    """Convert Perch (N, num_perch) logits → BC2026 (N, 234) sigmoid probs.
    Unmapped species → sigmoid(-50) ≈ 0 (NOT 0.5)."""
    bc_logits = np.full((perch_scores.shape[0], N_CLASSES), -50.0, dtype=np.float32)
    bc_logits[:, MAPPED_POS] = perch_scores[:, MAPPED_BC_IDX]
    return 1.0 / (1.0 + np.exp(-np.clip(bc_logits, -50, 50)))

test_bc = perch_to_bc2026(sc)
print(f"BC mapped test: shape={test_bc.shape}, mean={test_bc.mean():.4f}, max={test_bc.max():.4f}")
print(f"  Non-zero species (pred > 0.01): {(test_bc.max(axis=0) > 0.01).sum()}")
'''

# Apply
for c in nb["cells"]:
    if c.get("id") == "perch_setup":
        c["source"] = new_perch_setup.splitlines(keepends=True)
        print("[OK] Patched perch_setup cell")
    elif c.get("id") == "mapping":
        c["source"] = new_mapping.splitlines(keepends=True)
        print("[OK] Patched mapping cell")

# Save
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {NB_PATH}")

# Compile-check
import ast
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
        print(f"  [FAIL] cell {i} (id={c.get('id','?')}): {e}")
print(f"Result: {n_ok} OK, {n_err} ERRORS")
