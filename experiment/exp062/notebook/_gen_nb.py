"""Generate exp062: exp048 + exp032 R2 (Babych b3) Aves-only 15% blend.

Pattern same as exp056 (Stage 2A v1 Aves-only) which worked, but uses exp032 R2 model.
exp032 R2 (LB 0.887, Babych init) has different Aves inductive bias than other streams.
"""
import json
from pathlib import Path

SRC = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp048\notebook\nb_blend_topn1.ipynb")
OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp062\notebook\nb_blend_aves_exp032.ipynb")

nb = json.load(open(SRC, encoding="utf-8"))


def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}


# === exp032 R2 inference + Aves-only blend (insert before blend cell, replace blend) ===
# Strategy: insert as new cell before existing 'blend', so probs_e32 is computed
# BEFORE the existing blend cell. Then modify blend cell to:
#   - compute base 3-way blend → probs_blend (existing path)
#   - then apply Aves-only blend with probs_e32 → modify probs_blend
#   - then TopN PP (existing path)

# But existing 'blend' cell does both blend AND TopN PP. So we'll:
#   1. Insert e32-infer cell before blend
#   2. Modify blend cell to do Aves-only merge BEFORE TopN PP

e32_infer_src = r"""# === exp032 R2 (tf_efficientnet_b3 + Babych mel + SED head) inference ===
# 4th stream for exp062 Aves-only blend: Babych init brings different inductive bias for Aves
# ckpt: maekeso/birdclef2026-exp032-weights (r2_ckpt_best_ns22.pth, LB 0.887)

# ---- Locate exp032 R2 ckpt ----
E32_STATE_DIR = None
for _p in [
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp032-weights"),
    Path("/kaggle/input/birdclef2026-exp032-weights"),
]:
    if _p.exists() and (any(_p.rglob("r2_ckpt_best_ns22.pth"))
                        or any(_p.rglob("ckpt_best_ns22.pth"))):
        E32_STATE_DIR = _p; break
if E32_STATE_DIR is None:
    for _hit in Path("/kaggle/input").rglob("r2_ckpt_best_ns22.pth"):
        # skip exp020 R2 ckpts (5-fold pattern)
        if "fold" not in _hit.name:
            E32_STATE_DIR = _hit.parent; break
assert E32_STATE_DIR is not None, "exp062: exp032 R2 ckpt not found"
print(f"exp032 R2 ckpt dir: {E32_STATE_DIR}")

E32_CKPT = None
for _name in ["r2_ckpt_best_ns22.pth", "ckpt_best_ns22.pth"]:
    _hits = [p for p in E32_STATE_DIR.rglob(_name) if "fold" not in p.name]
    if _hits:
        E32_CKPT = _hits[0]; break
assert E32_CKPT is not None
print(f"  ckpt: {E32_CKPT.name} ({E32_CKPT.stat().st_size/1e6:.1f}MB)")

# ---- exp032 config (Babych b3 mel params) ----
E32_BACKBONE = "tf_efficientnet_b3.ns_jft_in1k"
E32_N_FFT  = 4096
E32_HOP    = 312
E32_N_MELS = 224
E32_FMIN   = 0
E32_FMAX   = 16000
E32_TRAIN_SAMPLES = SR * 5
E32_USE_DISTILL = True
E32_PERCH_DIM = 1536


class _E32MelTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=E32_N_FFT, hop_length=E32_HOP,
            n_mels=E32_N_MELS, f_min=E32_FMIN, f_max=E32_FMAX, power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, x):
        return self.db(self.mel_spec(x))


class _E32GeMFreq(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class _E32DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, fm):
        return self.proj(fm.mean(dim=[2, 3]))


class _E32SED(nn.Module):
    def __init__(self, backbone_name=E32_BACKBONE, num_classes=N_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = E32_TRAIN_SAMPLES // E32_HOP + 1
            dummy = torch.randn(1, 1, E32_N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = _E32GeMFreq(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        if E32_USE_DISTILL:
            self.distill_head = _E32DistillHead(self.backbone_dim, E32_PERCH_DIM)
    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = h.detach() if E32_USE_DISTILL else h
        h_cls = self.gem_freq(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise_logits = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise_logits, dim=2)
        if return_framewise:
            return clip_logits, framewise_logits.permute(0, 2, 1)
        return clip_logits


# ---- Load ckpt ----
try:
    _state = torch.load(str(E32_CKPT), map_location="cpu", weights_only=False)
except TypeError:
    _state = torch.load(str(E32_CKPT), map_location="cpu")
_sd = _state.get("model_state", _state.get("state_dict", _state))
print(f"  ckpt val_ns22={_state.get('best_ns22', _state.get('val_ns22', float('nan'))):.4f}")

e32_model = _E32SED().to(torch.device("cpu"))
e32_model.load_state_dict(_sd, strict=False)
e32_model.eval()
e32_mel_tf = _E32MelTF().to(torch.device("cpu"))
print(f"  e32 params: {sum(p.numel() for p in e32_model.parameters())/1e6:.1f}M")

# ---- Inference using audio_cache ----
t0 = time.time()
probs_e32 = []
with torch.no_grad():
    for _fi, _raw_60s in enumerate(audio_cache):
        _chunks = _raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
        _wav_t = torch.from_numpy(_chunks).unsqueeze(1)
        _mel = e32_mel_tf(_wav_t)
        for _i in range(_mel.size(0)):
            _mel[_i] = (_mel[_i] - _mel[_i].mean()) / (_mel[_i].std() + 1e-6)
        _clip, _frame = e32_model(_mel, return_framewise=True)
        _frame_max = _frame.max(dim=1).values
        _p_clip = torch.sigmoid(_clip).numpy().astype(np.float32)
        _p_frame = torch.sigmoid(_frame_max).numpy().astype(np.float32)
        _p_mean = 0.5 * _p_clip + 0.5 * _p_frame
        _p_smooth = gaussian_filter1d(_p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
        probs_e32.append(_p_smooth)
        if (_fi + 1) % 10 == 0 or _fi == len(audio_cache) - 1:
            print(f"  e32 [{_fi+1}/{len(audio_cache)}] {time.time()-t0:.0f}s")

probs_e32 = np.stack(probs_e32).astype(np.float32)
print(f"exp032 R2 done: {probs_e32.shape} in {time.time()-t0:.0f}s")
del e32_model, e32_mel_tf
gc.collect()
"""

# --- Modified blend cell (Aves-only merge before TopN) ---
blend_aves_src = r"""# === Stage C: 3-way rank blend + Aves-only exp032 R2 merge + TopN N=1 PP ===
# exp062: base = exp048 (NB4 + Tucker + exp029) + exp032 R2 Aves-only 15%

# (1) 3-way base blend (matches exp048)
BLEND_W_E10  = 0.30
BLEND_W_SED  = 0.40
BLEND_W_E17  = 0.30
USE_SED_PRE_AVG = False
assert abs(BLEND_W_E10 + BLEND_W_SED + BLEND_W_E17 - 1.0) < 1e-6

flat_e10 = probs_exp010.reshape(-1, probs_exp010.shape[-1])
flat_sed = probs_sed.reshape(-1, probs_sed.shape[-1])
flat_e17 = probs_e17.reshape(-1, probs_e17.shape[-1])

rank_e10 = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_sed = pd.DataFrame(flat_sed).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_e17 = pd.DataFrame(flat_e17).rank(axis=0, pct=True).to_numpy().astype(np.float32)

blend_flat = (BLEND_W_E10 * rank_e10
              + BLEND_W_SED * rank_sed
              + BLEND_W_E17 * rank_e17)

n_files = probs_exp010.shape[0]
n_windows = probs_exp010.shape[1]
n_classes = probs_exp010.shape[2]
blend_3d = blend_flat.reshape(n_files, n_windows, n_classes)
print(f"3-way blend: NB4={BLEND_W_E10:.2f} Tucker={BLEND_W_SED:.2f} e29={BLEND_W_E17:.2f}")

# (2) ★ Aves-only blend with exp032 R2 ★
# Convert exp032 sigmoid space → rank space for fair blend with base rank
flat_e32 = probs_e32.reshape(-1, n_classes)
rank_e32 = pd.DataFrame(flat_e32).rank(axis=0, pct=True).to_numpy().astype(np.float32)
e32_3d = rank_e32.reshape(n_files, n_windows, n_classes)

# Aves mask
AVES_MASK = np.array([LABEL_TO_CLASS.get(lbl) == "Aves" for lbl in PRIMARY_LABELS], dtype=bool)
print(f"Aves species count: {AVES_MASK.sum()} (expected 162)")

E32_AVES_WEIGHT = 0.15
print(f"Applying Aves-only exp032 R2 blend (weight={E32_AVES_WEIGHT})...")
blend_3d_new = blend_3d.copy()
blend_3d_new[:, :, AVES_MASK] = (
    (1 - E32_AVES_WEIGHT) * blend_3d[:, :, AVES_MASK]
    + E32_AVES_WEIGHT * e32_3d[:, :, AVES_MASK]
)
blend_3d = blend_3d_new
print(f"  after Aves merge: mean={blend_3d.mean():.4f}")

# (3) TopN N=1 PP (paper 256 spec)
FCS_TOP_K = 1
FCS_POWER = 1.0
file_topk_mean = np.zeros((n_files, n_classes), dtype=np.float32)
for fi in range(n_files):
    sorted_per_class = np.sort(blend_3d[fi], axis=0)[::-1]
    file_topk_mean[fi] = sorted_per_class[:FCS_TOP_K].mean(axis=0)
file_mult = np.power(file_topk_mean, FCS_POWER).astype(np.float32)
calibrated = blend_3d * file_mult[:, None, :]

# (4) Sonotype mirror
MIRROR_PAIRS = [
    ('s01', 's03'), ('s02', 's03'), ('s05', 's06'),
    ('s07', 's08'), ('s09', 's12'), ('s10', 's11'),
    ('s13', 's14'), ('s23', 's24'),
]
sonotype_to_idx = {}
for idx, label in enumerate(PRIMARY_LABELS):
    if isinstance(label, str) and label.startswith('s') and label[1:].isdigit():
        sonotype_to_idx[label] = idx

for a, b in MIRROR_PAIRS:
    if a in sonotype_to_idx and b in sonotype_to_idx:
        ia, ib = sonotype_to_idx[a], sonotype_to_idx[b]
        mirrored = np.maximum(calibrated[:, :, ia], calibrated[:, :, ib])
        calibrated[:, :, ia] = mirrored
        calibrated[:, :, ib] = mirrored

# (5) Build submission
flat_pred = calibrated.reshape(-1, n_classes)
sub = pd.DataFrame(flat_pred, columns=PRIMARY_LABELS)
sub.insert(0, "row_id", row_ids)
sub.to_csv("submission.csv", index=False)
print(f"submission.csv: {sub.shape}")
print(sub.head(3))
"""

# --- Modify NB ---
new_cells = []
inserted = False
for c in nb["cells"]:
    if c.get("id") == "blend" and not inserted:
        new_cells.append(code_cell("e32-infer", e32_infer_src))
        inserted = True
    if c.get("id") == "blend":
        new_cells.append(code_cell("blend", blend_aves_src))
    else:
        new_cells.append(c)

# Update hdr
for c in new_cells:
    if c.get("id") == "hdr":
        new_hdr = """# exp062 — Aves-only exp032 R2 blend on exp048

**Base**: exp048 (LB 0.950) — 3-way blend (NB4 + Tucker + exp029) + TopN N=1 PP

**新規追加**: exp032 R2 (LB 0.887, Babych b3 init) を **Aves species のみに 15% weight で blend**
- non-Aves species: exp048 のまま (drag 回避)
- Aves 162 species: 0.85 × base + 0.15 × exp032 R2 (rank space)

**exp032 R2 の特性**:
- Babych BC25 1位 init からの transfer
- tf_efficientnet_b3 + Babych mel (N_FFT=4096, hop=312, N_MELS=224)
- 異なる inductive bias = diversity 源

**Expected LB**: 0.949-0.954 (Aves 寄与 +0.001-0.005、drag risk あり)

**CPU**: ~70-80 min
- exp048 既存 3-way: ~60-70 min
- + exp032 R2 b3 SED: ~10-15 min
"""
        c["source"] = new_hdr.splitlines(keepends=True)
        break

nb["cells"] = new_cells

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Wrote {OUT}")
print(f"  cells: {len(new_cells)}")
for c in new_cells:
    cid = c.get("id", "?")
    src = "".join(c["source"])
    print(f"    {c['cell_type']:8s} id={cid:25s} lines={len(src.splitlines())}")
