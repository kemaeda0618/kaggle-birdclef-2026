"""Generate exp061: exp048 base + exp058 (effv2_b0 SED) inline 4-way blend.

Approach: copy exp048 NB, insert e58-infer cell before blend, modify blend to be 4-way.
exp058 ckpt: maekeso/birdclef2026-exp058-effv2b0-combined (effv2b0_combined_best.pth)
"""
import json
from pathlib import Path

SRC = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp048\notebook\nb_blend_topn1.ipynb")
OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp061\notebook\nb_blend_inline.ipynb")

nb = json.load(open(SRC, encoding="utf-8"))


def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}


# --- New cell: exp058 (effv2_b0 + SED) inference ---
e58_src = r"""# === exp058 (effv2_b0 + SED head) inline inference ===
# 4th stream for exp061: trained via OverfitOracle path (XC pseudo + BC2026 hard combined)
# ckpt: maekeso/birdclef2026-exp058-effv2b0-combined

# ---- Locate exp058 ckpt ----
E58_STATE_DIR = None
for _p in [
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp058-effv2b0-combined"),
    Path("/kaggle/input/birdclef2026-exp058-effv2b0-combined"),
]:
    if _p.exists() and any(_p.rglob("effv2b0_combined_best.pth")):
        E58_STATE_DIR = _p; break
if E58_STATE_DIR is None:
    for _hit in Path("/kaggle/input").rglob("effv2b0_combined_best.pth"):
        E58_STATE_DIR = _hit.parent; break
assert E58_STATE_DIR is not None, "exp061: exp058 effv2_b0 ckpt not found"

E58_CKPT = next(E58_STATE_DIR.rglob("effv2b0_combined_best.pth"), None)
assert E58_CKPT is not None
print(f"exp058 ckpt: {E58_CKPT} ({E58_CKPT.stat().st_size/1e6:.1f} MB)")

# ---- exp058 config (must match NB2 training) ----
E58_BACKBONE = "tf_efficientnetv2_b0.in1k"
E58_N_MELS = 256
E58_N_FFT  = 2048
E58_HOP    = 512
E58_FMIN   = 20
E58_FMAX   = 16000
E58_TRAIN_SAMPLES = SR * 5
E58_HIDDEN = 512
E58_IN_CHANS = 1


class _E58MelTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=E58_N_FFT, hop_length=E58_HOP,
            n_mels=E58_N_MELS, f_min=E58_FMIN, f_max=E58_FMAX, power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, x):
        return self.db(self.mel_spec(x))


class _E58GeMFreq(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class _E58SED(nn.Module):
    def __init__(self, backbone_name=E58_BACKBONE, num_classes=N_CLASSES,
                 drop_path_rate=0.1, hidden_dim=E58_HIDDEN, in_chans=E58_IN_CHANS):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=in_chans,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = E58_TRAIN_SAMPLES // E58_HOP + 1
            dummy = torch.randn(1, in_chans, E58_N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = _E58GeMFreq(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = self.gem_freq(h)
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
    _state = torch.load(str(E58_CKPT), map_location="cpu", weights_only=False)
except TypeError:
    _state = torch.load(str(E58_CKPT), map_location="cpu")
# NB2 saves: {"state_dict": ..., "val_ns22": ..., "val_macro": ..., "ep": ..., "cfg": ...}
_sd = _state.get("state_dict", _state)
print(f"  exp058 ckpt ep={_state.get('ep')}, val_ns22={_state.get('val_ns22', float('nan')):.4f}, "
      f"val_macro={_state.get('val_macro', float('nan')):.4f}")

e58_model = _E58SED().to(torch.device("cpu"))
miss, unexp = e58_model.load_state_dict(_sd, strict=False)
print(f"  load: missing={len(miss)}, unexpected={len(unexp)}")
e58_model.eval()
e58_mel_tf = _E58MelTF().to(torch.device("cpu"))
print(f"  e58 params: {sum(p.numel() for p in e58_model.parameters())/1e6:.1f}M")

# ---- Inference using audio_cache ----
t0 = time.time()
probs_e58 = []
with torch.no_grad():
    for _fi, _raw_60s in enumerate(audio_cache):
        _chunks = _raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
        _wav_t = torch.from_numpy(_chunks).unsqueeze(1)  # (12, 1, 160000)
        _mel = e58_mel_tf(_wav_t)
        for _i in range(_mel.size(0)):
            _mel[_i] = (_mel[_i] - _mel[_i].mean()) / (_mel[_i].std() + 1e-6)
        _clip, _frame = e58_model(_mel, return_framewise=True)
        _frame_max = _frame.max(dim=1).values
        _p_clip = torch.sigmoid(_clip).numpy().astype(np.float32)
        _p_frame = torch.sigmoid(_frame_max).numpy().astype(np.float32)
        _p_mean = 0.5 * _p_clip + 0.5 * _p_frame  # (12, N_CLASSES)
        _p_smooth = gaussian_filter1d(_p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
        probs_e58.append(_p_smooth)
        if (_fi + 1) % 10 == 0 or _fi == len(audio_cache) - 1:
            print(f"  e58 [{_fi+1}/{len(audio_cache)}] {time.time()-t0:.0f}s")

probs_e58 = np.stack(probs_e58).astype(np.float32)
print(f"exp058 done: {probs_e58.shape} in {time.time()-t0:.0f}s")
"""

# --- 4-way blend ---
blend_4way_src = r"""# === Stage C: 4-way rank blend (NB4 + Tucker + exp029 + exp058) + TopN N=1 PP ===
# exp061 ratio: exp058 as 4th stream
BLEND_W_E10  = 0.27   # NB4 (was 0.30)
BLEND_W_SED  = 0.36   # Tucker (was 0.40)
BLEND_W_E17  = 0.27   # exp029 (was 0.30)
BLEND_W_E58  = 0.10   # ★ exp058 effv2_b0 + XC pseudo trained (new)

assert abs(BLEND_W_E10 + BLEND_W_SED + BLEND_W_E17 + BLEND_W_E58 - 1.0) < 1e-6

flat_e10 = probs_exp010.reshape(-1, probs_exp010.shape[-1])
flat_sed = probs_sed.reshape(-1, probs_sed.shape[-1])
flat_e17 = probs_e17.reshape(-1, probs_e17.shape[-1])
flat_e58 = probs_e58.reshape(-1, probs_e58.shape[-1])

rank_e10 = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_sed = pd.DataFrame(flat_sed).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_e17 = pd.DataFrame(flat_e17).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_e58 = pd.DataFrame(flat_e58).rank(axis=0, pct=True).to_numpy().astype(np.float32)

blend_flat = (BLEND_W_E10 * rank_e10
              + BLEND_W_SED * rank_sed
              + BLEND_W_E17 * rank_e17
              + BLEND_W_E58 * rank_e58)
print(f"4-way blend: NB4={BLEND_W_E10:.2f} Tucker={BLEND_W_SED:.2f} e29={BLEND_W_E17:.2f} e58={BLEND_W_E58:.2f}")

# === TopN N=1 PP (paper 256 spec) ===
FCS_TOP_K = 1
FCS_POWER = 1.0

n_files = probs_exp010.shape[0]
n_windows = probs_exp010.shape[1]
n_classes = probs_exp010.shape[2]

blend_3d = blend_flat.reshape(n_files, n_windows, n_classes)
file_topk_mean = np.zeros((n_files, n_classes), dtype=np.float32)
for fi in range(n_files):
    sorted_per_class = np.sort(blend_3d[fi], axis=0)[::-1]
    file_topk_mean[fi] = sorted_per_class[:FCS_TOP_K].mean(axis=0)
file_mult = np.power(file_topk_mean, FCS_POWER).astype(np.float32)
calibrated = blend_3d * file_mult[:, None, :]

# === Sonotype mirror ===
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

# === Build submission ===
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
        new_cells.append(code_cell("e58-infer", e58_src))
        inserted = True
    if c.get("id") == "blend":
        new_cells.append(code_cell("blend", blend_4way_src))
    else:
        new_cells.append(c)

# Update hdr
for c in new_cells:
    if c.get("id") == "hdr":
        new_hdr = """# exp061 — Inline 4-way Blend (exp048 base + exp058 effv2_b0)

**Path**: exp048 (3-way + TopN N=1) + exp058 effv2_b0 SED inline = 4-way

**Streams**:
- NB4 (HGNetV2 + ProtoSSM + MLP): w=0.27
- Tucker SED 5-fold:                w=0.36
- exp029 R3 (eca_nfnet_l1):         w=0.27
- ★ exp058 (effv2_b0 + XC pseudo combined): w=0.10

**exp058 path**: OverfitOracle 1-stage combined training
- BC2026 train_audio (hard) + XC pseudo (frame-level from exp020 R2 5-fold teacher)
- effv2_b0 + SED head
- val_macro ~0.94+ expected

**Expected LB**: 0.952-0.954 (+0.002-0.004 from 0.950)

**CPU budget**: ~75-90 min
- exp048 3-way: ~60-70 min
- + exp058 effv2_b0 SED inference: ~10-15 min
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
