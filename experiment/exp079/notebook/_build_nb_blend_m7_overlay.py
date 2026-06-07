"""Build exp079: exp048 base + M7 overlay on Amphibia+Insecta species.

Strategy:
  - Start from exp048 nb_blend_topn1.ipynb (LB 0.950)
  - Add M7 inference cells (b0, 20s, 63 sub class)
  - Apply per-species overlay: for Amphibia+Insecta species, blend M7 with exp048 50/50
  - Save as new NB

Required kernel_sources (in addition to exp048's):
  - maekeso/birdclef2026-exp076-m7-train (M7 .pth ckpts)
"""
import json
from pathlib import Path

SRC_NB = Path("experiment/exp048/notebook/nb_blend_topn1.ipynb")
DST_NB = Path(__file__).with_name("nb_blend_exp048_m7.ipynb")

nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

# === Define new cells to insert AFTER cell 12 (final blend + submission) ===

def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "source": src.splitlines(keepends=True) if src else [],
            "outputs": [], "execution_count": None}

new_cells = []

# Cell M7-A: Model arch + species list
new_cells.append(code_cell("""# === M7 overlay: model arch + species list ===
print("\\n" + "="*60)
print("=== M7 OVERLAY START ===")
print("="*60)

# M7 spec (Babych b0 sub class 63)
M7_BACKBONE = "tf_efficientnet_b0.ns_jft_in1k"
M7_WINDOW_SEC = 20
M7_WINDOW_SAMPLES = SR * M7_WINDOW_SEC
M7_N_MELS = 224
M7_N_FFT = 4096
M7_HOP = 1252
M7_FMIN = 0
M7_FMAX = 16000
M7_TOP_DB = 80
M7_BEST_FOLD = 4  # Will pick best fold by val_ns22 from .pth metadata

# Locate M7 ckpts (kernel_sources)
M7_CKPT_DIR = None
for _p in ["/kaggle/input/birdclef2026-exp076-m7-train",
           "/kaggle/input/notebooks/maekeso/birdclef2026-exp076-m7-train"]:
    if Path(_p).exists():
        M7_CKPT_DIR = Path(_p); break
assert M7_CKPT_DIR is not None, "M7 train NB output not attached"
print(f"M7 ckpt dir: {M7_CKPT_DIR}")

m7_pth_paths = sorted(M7_CKPT_DIR.glob("m7_fold*_ckpt_best.pth"))
assert len(m7_pth_paths) > 0, "No M7 ckpts found"
print(f"  Found {len(m7_pth_paths)} M7 ckpts")

# Load species list (63 Amphibia + Insecta)
species63_path = M7_CKPT_DIR / "m7_species_63.json"
if species63_path.exists():
    with open(species63_path) as f:
        SPECIES_63 = json.load(f)
    print(f"  Loaded 63 species list from m7_species_63.json")
else:
    # Fallback: derive from taxonomy
    taxo = pd.read_csv(BASE / "taxonomy.csv")
    amph_ins = taxo[taxo["class_name"].isin(["Amphibia", "Insecta"])].reset_index(drop=True)
    SPECIES_63 = amph_ins["primary_label"].astype(str).tolist()
    print(f"  Derived 63 species from taxonomy")
assert len(SPECIES_63) == 63, f"Expected 63 species, got {len(SPECIES_63)}"

# Map 63 species → 234-class indices
m7_to_234_idx = np.array([PRIMARY_LABELS.index(sp) for sp in SPECIES_63], dtype=np.int32)
print(f"  M7 → 234 index mapping: {len(m7_to_234_idx)} indices")
print(f"  First 5 mappings: {[(s, m7_to_234_idx[i]) for i, s in enumerate(SPECIES_63[:5])]}")
"""))

# Cell M7-B: Model definition (CLEFClassifierSED for b0)
new_cells.append(code_cell("""# === M7 architecture (CLEFClassifierSED b0) ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as TT
import timm


def _m7_gem_freq(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)


class _M7GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return _m7_gem_freq(x, p=self.p, eps=self.eps)


class _M7AttHead(nn.Module):
    def __init__(self, in_chans, p=0.5, num_class=63, hidden_dim=512):
        super().__init__()
        self.pooling = _M7GeMFreq()
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),
            nn.Linear(in_chans, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(hidden_dim, num_class, kernel_size=1, bias=True)
        self.fix_scale = nn.Conv1d(hidden_dim, num_class, kernel_size=1, bias=True)
    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)
        feat = self.dense_layers(feat).permute(0, 2, 1)
        framewise_logit = self.fix_scale(feat)
        return {"framewise_logit": framewise_logit}


class _M7NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        norm_max = torch.amax(Xstd, dim=(1, 2), keepdim=True)
        norm_min = torch.amin(Xstd, dim=(1, 2), keepdim=True)
        return (Xstd - norm_min) / (norm_max - norm_min + self.eps)


class _M7SpecFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            TT.MelSpectrogram(sample_rate=SR, normalized=True, n_fft=M7_N_FFT,
                              hop_length=M7_HOP, win_length=M7_N_FFT,
                              f_max=M7_FMAX, n_mels=M7_N_MELS, f_min=M7_FMIN),
            TT.AmplitudeToDB(top_db=M7_TOP_DB),
        )
        self.norm = _M7NormalizeMelSpec()
    def forward(self, x):
        return self.norm(self.feature_extractor(x))


class _M7CLEFClassifierSED(nn.Module):
    def __init__(self, backbone_name=M7_BACKBONE, num_classes=63):
        super().__init__()
        self.mel_spectr_generator = _M7SpecFeatureExtractor()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, features_only=True,
            in_chans=3, drop_path_rate=0.0,
        )
        backbone_dim = self.backbone.feature_info.channels()[-1]
        self.head = _M7AttHead(in_chans=backbone_dim, num_class=num_classes)
    def forward(self, wav, return_framewise=False):
        spec = self.mel_spectr_generator(wav)
        spec3 = torch.stack([spec, spec, spec], 1)
        feat = self.backbone(spec3)[-1]
        head_output = self.head(feat)
        framewise_logit = head_output["framewise_logit"]
        clip_logit = framewise_logit.max(dim=-1).values
        if return_framewise:
            return clip_logit, framewise_logit.permute(0, 2, 1)
        return clip_logit


print("OK M7 architecture defined")
"""))

# Cell M7-C: Load best M7 ckpt + run inference using audio_cache (extract 20s from 60s)
new_cells.append(code_cell("""# === M7 ckpt load + inference (uses audio_cache built by NB4 stage) ===
import re

# Pick best M7 fold by val_ns22
m7_best_score = -1
m7_best_pth = None
m7_fold_scores = []
for p in m7_pth_paths:
    try:
        st = torch.load(str(p), map_location="cpu", weights_only=False)
        score = st.get("val_ns22", st.get("best_ns22", -1))
        m_fold = re.search(r"fold(\\d+)", p.name)
        fi = int(m_fold.group(1)) if m_fold else -1
        m7_fold_scores.append((fi, score, p))
        if score > m7_best_score:
            m7_best_score = score
            m7_best_pth = p
    except Exception as e:
        print(f"  {p.name}: load failed ({e})")

print(f"M7 fold scores:")
for fi, sc, p in sorted(m7_fold_scores):
    marker = " ★" if p == m7_best_pth else ""
    print(f"  fold {fi}: val_ns22={sc:.4f}{marker}")
print(f"\\nBest M7 ckpt: {m7_best_pth.name} (val_ns22={m7_best_score:.4f})")

# Load M7 model
m7_model = _M7CLEFClassifierSED()
m7_state = torch.load(str(m7_best_pth), map_location="cpu", weights_only=False)
m7_model.load_state_dict(m7_state.get("model_state", m7_state), strict=False)
m7_model.eval()
print(f"  M7 model: {sum(p.numel() for p in m7_model.parameters())/1e6:.1f}M params")


def _extract_20s_chunks_from_60s(wav_60s):
    \"\"\"Extract 12 chunks of 20s from 60s audio, aligned to 5s test row_id positions.\"\"\"
    chunks = np.zeros((N_WINDOWS, M7_WINDOW_SAMPLES), dtype=np.float32)
    target_60s = SR * 60
    if len(wav_60s) < target_60s:
        wav_60s = np.pad(wav_60s, (0, target_60s - len(wav_60s)))
    elif len(wav_60s) > target_60s:
        wav_60s = wav_60s[:target_60s]
    for ci in range(N_WINDOWS):
        center_sec = (ci + 0.5) * 5
        start_sec = max(0, center_sec - M7_WINDOW_SEC / 2)
        if start_sec + M7_WINDOW_SEC > 60:
            start_sec = 60 - M7_WINDOW_SEC
        start = int(start_sec * SR)
        end = start + M7_WINDOW_SAMPLES
        ch = wav_60s[start:end]
        if len(ch) < M7_WINDOW_SAMPLES:
            ch = np.pad(ch, (0, M7_WINDOW_SAMPLES - len(ch)))
        # Amplitude normalize
        m = np.abs(ch).max()
        if m > 0: ch = ch / m
        chunks[ci] = ch
    return chunks


# Inference using audio_cache (60s raw per file)
import time as _t
t0 = _t.time()
M7_BATCH = 4   # small for memory (20s mel × 3 channels)
probs_m7 = []  # (N_files, 12, 63)

with torch.no_grad():
    for fi, raw_60s in enumerate(audio_cache):
        chunks_20s = _extract_20s_chunks_from_60s(raw_60s.astype(np.float32))
        chunks_t = torch.from_numpy(chunks_20s)
        file_preds = np.zeros((N_WINDOWS, 63), dtype=np.float32)
        for s in range(0, N_WINDOWS, M7_BATCH):
            batch = chunks_t[s:s+M7_BATCH]
            clip_logit, framewise = m7_model(batch, return_framewise=True)
            frame_max = framewise.max(dim=1).values
            p_clip = torch.sigmoid(clip_logit).numpy()
            p_fmax = torch.sigmoid(frame_max).numpy()
            p_blend = 0.5 * p_clip + 0.5 * p_fmax
            file_preds[s:s+len(p_blend)] = p_blend
        probs_m7.append(file_preds)
        if (fi + 1) % 50 == 0 or fi == len(audio_cache) - 1:
            print(f"  M7 [{fi+1}/{len(audio_cache)}] {(_t.time()-t0):.0f}s")

probs_m7 = np.stack(probs_m7).astype(np.float32)
print(f"\\nM7 inference done: {probs_m7.shape} in {(_t.time()-t0)/60:.1f}min")
print(f"  Stats: mean={probs_m7.mean():.4f}, max={probs_m7.max():.4f}")
"""))

# Cell M7-D: Map 63 → 234, apply per-species overlay, overwrite submission
new_cells.append(code_cell("""# === M7 → 234 mapping + per-species overlay + new submission ===
# probs_m7 shape: (N_files, 12, 63)
# Apply rank-pct over flat (n_files * 12) chunks for fair blend with rank-based existing pipeline
probs_m7_flat = probs_m7.reshape(-1, 63)  # (n_chunks, 63)
rank_m7 = pd.DataFrame(probs_m7_flat).rank(axis=0, pct=True).to_numpy().astype(np.float32)

print(f"\\n=== Per-species overlay on Amphibia+Insecta (63 species) ===")
print(f"Original submission shape: {submission.shape}")
print(f"M7 rank shape: {rank_m7.shape}")

# Get existing predictions in same row_id order as audio_cache
# all_row_ids is the order audio_cache was iterated
# submission rows are in sample_sub order — need to map by row_id

# Build row_id → m7_rank mapping
m7_row_to_rank = {}
for fi, raw_id in enumerate(all_row_ids):
    chunk_idx = fi % N_WINDOWS
    file_idx = fi // N_WINDOWS
    m7_row_to_rank[raw_id] = rank_m7[fi]   # (63,)

# Apply overlay: for each row in submission, blend Amphibia+Insecta species 50/50
OVERLAY_WEIGHT = 0.5  # M7 weight (0=no overlay, 1=replace with M7)
print(f"  Overlay weight (M7): {OVERLAY_WEIGHT}")

# Get current Amphibia+Insecta columns from submission
amph_ins_col_names = SPECIES_63
amph_ins_col_idx_in_234 = m7_to_234_idx  # 63 indices

# Existing submission preds for those species (use rank-pct from existing)
existing_preds = submission[amph_ins_col_names].to_numpy().astype(np.float32)  # (n_rows, 63)
existing_ranks = pd.DataFrame(existing_preds).rank(axis=0, pct=True).to_numpy().astype(np.float32)

# M7 ranks aligned to submission order
m7_aligned = np.zeros((len(submission), 63), dtype=np.float32)
for i, rid in enumerate(submission["row_id"].values):
    if rid in m7_row_to_rank:
        m7_aligned[i] = m7_row_to_rank[rid]
    # else: zeros (no M7 prediction — shouldn't happen if all_row_ids covers test)

# Blend in rank space: 0.5 * existing_rank + 0.5 * m7_rank
blended_rank = (1.0 - OVERLAY_WEIGHT) * existing_ranks + OVERLAY_WEIGHT * m7_aligned

# Replace those columns in submission
for col_i, col_name in enumerate(amph_ins_col_names):
    submission[col_name] = blended_rank[:, col_i]

print(f"  Overlay applied to {len(amph_ins_col_names)} species columns")

# Re-save submission.csv (overwrite the existing one)
submission.to_csv("submission.csv", index=False)
print(f"\\nNew submission saved (M7 overlay applied)")
print(f"  Shape: {submission.shape}")
print(f"  Mean pred: {submission[PRIMARY_LABELS].values.mean():.6f}")
print(f"  Max pred:  {submission[PRIMARY_LABELS].values.max():.6f}")
print(submission.head(3))
print(f"\\n=== M7 OVERLAY DONE ===")
"""))

# Append new cells after Cell 12 (final cell)
nb["cells"].extend(new_cells)

# Add cell IDs (required for some NB versions)
for i, c in enumerate(nb["cells"]):
    if "id" not in c:
        c["id"] = f"cell-{i}"

# Save
DST_NB.parent.mkdir(parents=True, exist_ok=True)
DST_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Saved: {DST_NB}")
print(f"Total cells: {len(nb['cells'])} (original {len(nb['cells'])-len(new_cells)} + {len(new_cells)} new)")
