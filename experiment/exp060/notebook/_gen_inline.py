"""Generate exp060 v2 (inline 4-way blend): exp048 base + exp020 R2 5-fold ONNX inline.

Approach: copy exp048 NB, insert e20-infer cell before blend, modify blend to be 4-way.
"""
import json
from pathlib import Path

SRC = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp048\notebook\nb_blend_topn1.ipynb")
OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp060\notebook\nb_blend_inline.ipynb")

nb = json.load(open(SRC, encoding="utf-8"))


def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}


# --- New cell: exp020 R2 5-fold ONNX inference ---
e20_src = r"""# === exp020 R2 5-fold (eca_nfnet_l0 + SED) ONNX inference ===
# 4th stream for exp060: 5-fold ensemble ONNX from maekeso/birdclef2026-exp020-weights-5fold
# Reuses audio_cache (60s per file, 12 chunks of 5s).
import re
import onnxruntime as ort

# ---- Locate exp020 R2 ckpts ----
E20_STATE_DIR = None
for _p in [
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp020-weights-5fold"),
    Path("/kaggle/input/birdclef2026-exp020-weights-5fold"),
]:
    if _p.exists() and any(_p.rglob("r2_fold*_ckpt_best_ns22.pth")):
        E20_STATE_DIR = _p; break
if E20_STATE_DIR is None:
    for _hit in Path("/kaggle/input").rglob("r2_fold0_ckpt_best_ns22.pth"):
        E20_STATE_DIR = _hit.parent; break
assert E20_STATE_DIR is not None, "exp060: exp020 R2 ckpts not found"
print(f"exp020 R2 ckpt dir: {E20_STATE_DIR}")

# ---- exp020 config (must match training) ----
E20_BACKBONE = "eca_nfnet_l0"
E20_N_MELS   = 256
E20_N_FFT    = 2048
E20_HOP      = 512
E20_FMIN     = 20
E20_FMAX     = 16000
E20_TRAIN_SAMPLES = SR * 5
E20_USE_DISTILL = True
E20_PERCH_DIM = 1536


class _E20MelTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=E20_N_FFT, hop_length=E20_HOP,
            n_mels=E20_N_MELS, f_min=E20_FMIN, f_max=E20_FMAX, power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    def forward(self, x):
        return self.db(self.mel_spec(x))


class _E20GeMFreq(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class _E20DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, fm):
        return self.proj(fm.mean(dim=[2, 3]))


class _E20SED(nn.Module):
    def __init__(self, backbone_name=E20_BACKBONE, num_classes=N_CLASSES,
                 drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = E20_TRAIN_SAMPLES // E20_HOP + 1
            dummy = torch.randn(1, 1, E20_N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = _E20GeMFreq(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        if E20_USE_DISTILL:
            self.distill_head = _E20DistillHead(self.backbone_dim, E20_PERCH_DIM)
    def forward(self, x, return_framewise=False):
        h = self.backbone(x)
        h_cls = h.detach() if E20_USE_DISTILL else h
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


# ---- ONNX export wrapper ----
class _E20ONNX(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, mel):
        return self.model(mel, return_framewise=True)


# ---- Export 5 folds to ONNX and load sessions ----
E20_ONNX_DIR = Path("/tmp/e20_onnx")
E20_ONNX_DIR.mkdir(exist_ok=True)

e20_mel_tf = _E20MelTF()
e20_ckpts = sorted(E20_STATE_DIR.rglob("r2_fold*_ckpt_best_ns22.pth"))
print(f"Found {len(e20_ckpts)} exp020 R2 fold ckpts")

# build dummy mel for export
_dummy_wav = torch.randn(1, 1, SR * 5)
_dummy_mel = e20_mel_tf(_dummy_wav)
_dummy_mel = (_dummy_mel - _dummy_mel.mean()) / (_dummy_mel.std() + 1e-6)

e20_sessions = []
_t_exp = time.time()
for ck in e20_ckpts:
    fid = int(re.search(r"fold(\d+)", ck.name).group(1))
    try:
        st = torch.load(str(ck), map_location="cpu", weights_only=False)
    except TypeError:
        st = torch.load(str(ck), map_location="cpu")
    m = _E20SED(); m.load_state_dict(st["model_state"], strict=False); m.eval()
    onnx_p = E20_ONNX_DIR / f"e20_fold{fid}.onnx"
    with torch.no_grad():
        torch.onnx.export(_E20ONNX(m), _dummy_mel, str(onnx_p),
                          opset_version=17,
                          input_names=["mel"], output_names=["clip", "framewise"],
                          dynamic_axes={"mel": {0: "batch"},
                                        "clip": {0: "batch"},
                                        "framewise": {0: "batch"}},
                          do_constant_folding=True, dynamo=False)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_p), sess_options=opts, providers=["CPUExecutionProvider"])
    e20_sessions.append(sess)
    del m, st
    gc.collect()
print(f"5 fold ONNX exported & loaded in {time.time()-_t_exp:.1f}s")

# ---- Inference on audio_cache ----
t0 = time.time()
probs_e20 = []  # (N_files, 12, N_CLASSES)
for fi, raw_60s in enumerate(audio_cache):
    chunks = raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
    wav_t = torch.from_numpy(chunks).unsqueeze(1)  # (12, 1, 160000)
    mel = e20_mel_tf(wav_t)
    for i in range(mel.size(0)):
        mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
    mel_np = mel.numpy()
    # 5-fold ensemble (sigmoid space avg)
    accum_clip = None
    accum_frame = None
    for sess in e20_sessions:
        out = sess.run(["clip", "framewise"], {"mel": mel_np})
        p_clip = 1.0 / (1.0 + np.exp(-np.clip(out[0], -50, 50)))
        p_frame = 1.0 / (1.0 + np.exp(-np.clip(out[1], -50, 50)))
        p_frame_max = p_frame.max(axis=1)
        if accum_clip is None:
            accum_clip = p_clip; accum_frame = p_frame_max
        else:
            accum_clip = accum_clip + p_clip
            accum_frame = accum_frame + p_frame_max
    accum_clip /= len(e20_sessions)
    accum_frame /= len(e20_sessions)
    # clip + frame max blend (Tucker recipe)
    p_mean = 0.5 * accum_clip + 0.5 * accum_frame
    p_smooth = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
    probs_e20.append(p_smooth)
    if (fi + 1) % 10 == 0 or fi == len(audio_cache) - 1:
        print(f"  e20 [{fi+1}/{len(audio_cache)}] {time.time()-t0:.0f}s")

probs_e20 = np.stack(probs_e20).astype(np.float32)
print(f"exp020 R2 done: {probs_e20.shape} in {time.time()-t0:.0f}s")
del e20_sessions
gc.collect()
"""

# --- New blend cell (4-way) ---
blend_4way_src = r"""# === Stage C: 4-way rank blend (NB4 + Tucker + exp029 + exp020 R2) + TopN N=1 PP ===
# exp060 v2 ratio: 4-way weighting
BLEND_W_E10  = 0.27   # NB4 (was 0.30 in exp048)
BLEND_W_SED  = 0.36   # Tucker (was 0.40)
BLEND_W_E17  = 0.27   # exp029 (was 0.30)
BLEND_W_E20  = 0.10   # ★ exp020 R2 4th stream (new)

assert abs(BLEND_W_E10 + BLEND_W_SED + BLEND_W_E17 + BLEND_W_E20 - 1.0) < 1e-6, \
    f"blend weights must sum to 1.0, got {BLEND_W_E10 + BLEND_W_SED + BLEND_W_E17 + BLEND_W_E20}"

flat_e10 = probs_exp010.reshape(-1, probs_exp010.shape[-1])
flat_sed = probs_sed.reshape(-1, probs_sed.shape[-1])
flat_e17 = probs_e17.reshape(-1, probs_e17.shape[-1])
flat_e20 = probs_e20.reshape(-1, probs_e20.shape[-1])

# rank-based 4-way blend (matches exp048 path)
rank_e10 = pd.DataFrame(flat_e10).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_sed = pd.DataFrame(flat_sed).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_e17 = pd.DataFrame(flat_e17).rank(axis=0, pct=True).to_numpy().astype(np.float32)
rank_e20 = pd.DataFrame(flat_e20).rank(axis=0, pct=True).to_numpy().astype(np.float32)

blend_flat = (BLEND_W_E10 * rank_e10
              + BLEND_W_SED * rank_sed
              + BLEND_W_E17 * rank_e17
              + BLEND_W_E20 * rank_e20)
print(f"4-way blend: NB4={BLEND_W_E10:.2f} Tucker={BLEND_W_SED:.2f} e29={BLEND_W_E17:.2f} e20={BLEND_W_E20:.2f}")

# === File-level Calibration with Sonotype mirror (FCS, paper 256 TopN N=1) ===
# Same as exp048: FCS_TOP_K = 1, FCS_POWER = 1.0
FCS_TOP_K = 1
FCS_POWER = 1.0
print(f"FCS spec: TOP_K={FCS_TOP_K}, POWER={FCS_POWER}")

n_files = probs_exp010.shape[0]
n_windows = probs_exp010.shape[1]
n_classes = probs_exp010.shape[2]

blend_3d = blend_flat.reshape(n_files, n_windows, n_classes)

# Per-file top-K mean (across windows) per class, raised to POWER
# This is the Sonotype mirror / paper 256 FCS recipe
file_topk_mean = np.zeros((n_files, n_classes), dtype=np.float32)
for fi in range(n_files):
    sorted_per_class = np.sort(blend_3d[fi], axis=0)[::-1]  # (n_windows, n_classes), descending
    file_topk_mean[fi] = sorted_per_class[:FCS_TOP_K].mean(axis=0)

# Apply: blend_3d[fi, wi, c] *= (file_topk_mean[fi, c] ** POWER)
file_mult = np.power(file_topk_mean, FCS_POWER).astype(np.float32)  # (n_files, n_classes)
calibrated = blend_3d * file_mult[:, None, :]
print(f"calibrated shape: {calibrated.shape}, mean: {calibrated.mean():.4f}, max: {calibrated.max():.4f}")

# === Sonotype mirror groups ===
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
# Insert e20-infer cell before blend
new_cells = []
inserted = False
for c in nb["cells"]:
    if c.get("id") == "blend" and not inserted:
        # Insert e20-infer just before blend
        new_cells.append(code_cell("e20-infer", e20_src))
        inserted = True
    # Replace blend with 4-way version
    if c.get("id") == "blend":
        new_cells.append(code_cell("blend", blend_4way_src))
    else:
        new_cells.append(c)
assert inserted, "blend cell not found"

# Update hdr
for c in new_cells:
    if c.get("id") == "hdr":
        new_hdr = """# exp060 v2 — Inline 4-way Blend (exp048 base + exp020 R2 5-fold)

**Path**: exp048 (3-way blend + TopN N=1) + exp020 R2 5-fold ONNX inline = 4-way

**Streams**:
- NB4 (HGNetV2 + ProtoSSM + MLP): w=0.27 (was 0.30 in exp048)
- Tucker SED 5-fold:                w=0.36 (was 0.40)
- exp029 R3 (eca_nfnet_l1):         w=0.27 (was 0.30)
- ★ exp020 R2 5-fold (NEW):        w=0.10

**Expected LB**: 0.951-0.953 (+0.001-0.003 from 0.950)

**CPU budget**: ~85-95 min (tight, monitor)
- exp048 3-way: ~60-70 min
- + exp020 R2 5-fold ONNX: ~15-25 min
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
