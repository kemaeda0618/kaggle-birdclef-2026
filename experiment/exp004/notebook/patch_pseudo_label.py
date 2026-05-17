"""
Patch submission.ipynb to load pseudo-label models in submit mode.

Changes:
- Cell [3]: Add `import pickle`
- Cells [8], [10], [16], [17]: Wrap in `if MODE == "train":` guard
- Cell [18]: In submit mode, load pseudo-label models instead of training

Usage: python patch_pseudo_label.py
"""
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
NB_PATH = HERE / "submission.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)


def get_source(cell):
    return "".join(cell["source"])


def set_source(cell, text):
    """Set cell source as list of lines (ipynb format)."""
    lines = text.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)
    cell["source"] = result


def indent(text, spaces=4):
    """Indent all lines of text by given number of spaces."""
    prefix = " " * spaces
    lines = text.split("\n")
    result = []
    for line in lines:
        if line.strip():  # non-empty line
            result.append(prefix + line)
        else:
            result.append("")  # keep blank lines blank
    return "\n".join(result)


def wrap_in_train_guard(cell, skip_msg="using pre-trained pseudo-label models"):
    """Wrap cell content in `if MODE == "train":` guard."""
    original = get_source(cell)
    # Extract the first comment line as cell label
    lines = original.split("\n")
    header = lines[0] if lines[0].startswith("#") else ""
    body = "\n".join(lines[1:]) if header else original

    if header:
        wrapped = f'{header}\nif MODE == "train":\n{indent(body)}\nelse:\n    print("Skipped — {skip_msg}")'
    else:
        wrapped = f'if MODE == "train":\n{indent(original)}\nelse:\n    print("Skipped — {skip_msg}")'
    set_source(cell, wrapped)


# ==============================================================
# 1. Cell [3]: Add `import pickle` after tqdm import
# ==============================================================
src3 = get_source(nb["cells"][3])
src3 = src3.replace(
    "from tqdm.auto import tqdm",
    "from tqdm.auto import tqdm\nimport pickle",
)
set_source(nb["cells"][3], src3)
print("[3] Added `import pickle`")

# ==============================================================
# 2. Cell [8]: Wrap in train guard (Load Perch cache)
# ==============================================================
wrap_in_train_guard(nb["cells"][8], "train_soundscapes cache not needed (pseudo-label models)")
print("[8] Wrapped in train guard")

# ==============================================================
# 3. Cell [10]: Wrap in train guard (OOF base/prior)
# ==============================================================
wrap_in_train_guard(nb["cells"][10], "OOF not needed (pseudo-label models)")
print("[10] Wrapped in train guard")

# ==============================================================
# 4. Cell [16]: Wrap in train guard (Fit prior tables)
# ==============================================================
wrap_in_train_guard(nb["cells"][16], "prior tables loaded from pseudo-label aux")
print("[16] Wrapped in train guard")

# ==============================================================
# 5. Cell [17]: Wrap in train guard (Fit scaler + PCA)
# ==============================================================
wrap_in_train_guard(nb["cells"][17], "scaler/PCA loaded from pseudo-label aux")
print("[17] Wrapped in train guard")

# ==============================================================
# 6. Cell [18]: Train vs Load pseudo-label models
# ==============================================================
original_train = get_source(nb["cells"][18])

load_code = '''\
# Load pre-trained pseudo-label models
import pickle

PSEUDO_DIR = Path("/kaggle/input/exp004-pseudo-label")
if not PSEUDO_DIR.exists():
    PSEUDO_DIR = Path("/kaggle/input/datasets/maekeso/exp004-pseudo-label")
print(f"Loading pseudo-label models from: {PSEUDO_DIR}")

# 1. Load ProtoSSM weights
ssm_cfg = CFG["proto_ssm"]
n_families, class_to_family, fam_to_idx = build_family_mapping(taxonomy, PRIMARY_LABELS)

model = ProtoSSM(
    d_input=1536,
    d_model=ssm_cfg["d_model"],
    d_state=ssm_cfg["d_state"],
    n_ssm_layers=ssm_cfg["n_ssm_layers"],
    n_classes=N_CLASSES,
    n_windows=N_WINDOWS,
    dropout=ssm_cfg["dropout"],
).to(DEVICE)
model.init_family_head(n_families, class_to_family)

checkpoint = torch.load(PSEUDO_DIR / "pseudo_label_proto_ssm.pt", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded ProtoSSM: {model.count_parameters():,} params")

# 2. Load MLP probes
with open(PSEUDO_DIR / "pseudo_label_probes.pkl", "rb") as f:
    probe_models = pickle.load(f)
print(f"Loaded {len(probe_models)} MLP probes")

# 3. Load auxiliary (scaler, PCA, prior tables)
with open(PSEUDO_DIR / "pseudo_label_aux.pkl", "rb") as f:
    aux = pickle.load(f)
emb_scaler = aux["emb_scaler"]
emb_pca = aux["emb_pca"]
final_prior_tables = aux["prior_tables"]
print(f"Loaded aux: PCA dim={emb_pca.n_components_}")

train_time = 0
train_history = None
print("All pseudo-label models loaded.")'''

new_cell18 = f'if MODE == "train":\n{indent(original_train)}\nelse:\n{indent(load_code)}'
set_source(nb["cells"][18], new_cell18)
print("[18] Added pseudo-label loading branch")

# ==============================================================
# Save
# ==============================================================
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nPatched: {NB_PATH}")
print(f"Size: {NB_PATH.stat().st_size:,} bytes")
