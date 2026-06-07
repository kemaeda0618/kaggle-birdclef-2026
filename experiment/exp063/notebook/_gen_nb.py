"""Generate exp063: exp048 + exp020 R2 5-fold Aves-only 15% blend.

Pattern: exp062 path (Aves-only blend before TopN PP) but uses exp020 R2 5-fold (LB 0.915)
instead of exp032 R2 b3 (LB 0.887). Stronger 4th stream for Aves.
"""
import json
from pathlib import Path

SRC = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp048\notebook\nb_blend_topn1.ipynb")
OUT = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp063\notebook\nb_blend_aves_exp020.ipynb")

nb = json.load(open(SRC, encoding="utf-8"))


def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}


# === exp020 R2 5-fold ONNX inference (same as exp060) ===
e20_infer_src = r"""# === exp020 R2 5-fold (eca_nfnet_l0 + SED) ONNX inference ===
# 4th stream for exp063 Aves-only blend: LB 0.915 single, strong Aves
import re
import onnxruntime as ort

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
assert E20_STATE_DIR is not None, "exp063: exp020 R2 ckpts not found"
print(f"exp020 R2 ckpt dir: {E20_STATE_DIR}")

E20_BACKBONE = "eca_nfnet_l0"
E20_N_MELS, E20_N_FFT, E20_HOP, E20_FMIN, E20_FMAX = 256, 2048, 512, 20, 16000
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
    def forward(self, x): return self.db(self.mel_spec(x))


class _E20GeMFreq(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init))); self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class _E20DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, fm): return self.proj(fm.mean(dim=[2, 3]))


class _E20SED(nn.Module):
    def __init__(self, num_classes=N_CLASSES, drop_path_rate=0.1, hidden_dim=512):
        super().__init__()
        self.backbone = timm.create_model(
            E20_BACKBONE, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = E20_TRAIN_SAMPLES // E20_HOP + 1
            dummy = torch.randn(1, 1, E20_N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]
        self.gem_freq = _E20GeMFreq(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25), nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Dropout(0.5),
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


class _E20ONNX(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, mel): return self.model(mel, return_framewise=True)


E20_ONNX_DIR = Path("/tmp/e20_onnx")
E20_ONNX_DIR.mkdir(exist_ok=True)
e20_mel_tf = _E20MelTF()
e20_ckpts = sorted(E20_STATE_DIR.rglob("r2_fold*_ckpt_best_ns22.pth"))
print(f"Found {len(e20_ckpts)} exp020 R2 fold ckpts")

_dummy_wav = torch.randn(1, 1, SR * 5)
_dummy_mel = e20_mel_tf(_dummy_wav)
_dummy_mel = (_dummy_mel - _dummy_mel.mean()) / (_dummy_mel.std() + 1e-6)

e20_sessions = []
_t = time.time()
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
                          dynamic_axes={"mel": {0: "batch"}, "clip": {0: "batch"}, "framewise": {0: "batch"}},
                          do_constant_folding=True, dynamo=False)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_p), sess_options=opts, providers=["CPUExecutionProvider"])
    e20_sessions.append(sess)
    del m, st
    gc.collect()
print(f"5-fold ONNX exported & loaded in {time.time()-_t:.1f}s")

# Inference
t0 = time.time()
probs_e20 = []
for fi, raw_60s in enumerate(audio_cache):
    chunks = raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
    wav_t = torch.from_numpy(chunks).unsqueeze(1)
    mel = e20_mel_tf(wav_t)
    for i in range(mel.size(0)):
        mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
    mel_np = mel.numpy()
    accum_clip = None; accum_frame = None
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
    accum_clip /= len(e20_sessions); accum_frame /= len(e20_sessions)
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

# === Aves-only blend (same pattern as exp062) ===
blend_aves_src = r"""# === Stage C: 3-way base + Aves-only exp020 R2 5-fold + TopN N=1 PP ===
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

blend_flat = (BLEND_W_E10 * rank_e10 + BLEND_W_SED * rank_sed + BLEND_W_E17 * rank_e17)

n_files = probs_exp010.shape[0]
n_windows = probs_exp010.shape[1]
n_classes = probs_exp010.shape[2]
blend_3d = blend_flat.reshape(n_files, n_windows, n_classes)
print(f"3-way base blend: NB4={BLEND_W_E10:.2f} Tucker={BLEND_W_SED:.2f} e29={BLEND_W_E17:.2f}")

# ★ Aves-only exp020 R2 blend ★
flat_e20 = probs_e20.reshape(-1, n_classes)
rank_e20 = pd.DataFrame(flat_e20).rank(axis=0, pct=True).to_numpy().astype(np.float32)
e20_3d = rank_e20.reshape(n_files, n_windows, n_classes)

AVES_MASK = np.array([LABEL_TO_CLASS.get(lbl) == "Aves" for lbl in PRIMARY_LABELS], dtype=bool)
print(f"Aves species: {AVES_MASK.sum()} (expected 162)")

E20_AVES_WEIGHT = 0.15
print(f"Aves-only exp020 R2 blend (weight={E20_AVES_WEIGHT})...")
blend_3d_new = blend_3d.copy()
blend_3d_new[:, :, AVES_MASK] = (
    (1 - E20_AVES_WEIGHT) * blend_3d[:, :, AVES_MASK]
    + E20_AVES_WEIGHT * e20_3d[:, :, AVES_MASK]
)
blend_3d = blend_3d_new
print(f"  after Aves merge: mean={blend_3d.mean():.4f}")

# TopN N=1 PP
FCS_TOP_K = 1
FCS_POWER = 1.0
file_topk_mean = np.zeros((n_files, n_classes), dtype=np.float32)
for fi in range(n_files):
    sorted_per_class = np.sort(blend_3d[fi], axis=0)[::-1]
    file_topk_mean[fi] = sorted_per_class[:FCS_TOP_K].mean(axis=0)
file_mult = np.power(file_topk_mean, FCS_POWER).astype(np.float32)
calibrated = blend_3d * file_mult[:, None, :]

# Sonotype mirror
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

flat_pred = calibrated.reshape(-1, n_classes)
sub = pd.DataFrame(flat_pred, columns=PRIMARY_LABELS)
sub.insert(0, "row_id", row_ids)
sub.to_csv("submission.csv", index=False)
print(f"submission.csv: {sub.shape}")
print(sub.head(3))
"""

# Apply EMB_DIR path fix to config cell too
OLD_BLOCK = 'EMB_DIR = Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp010-nb1-embedding")'
NEW_BLOCK = '''_emb_candidates = [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp010-nb1-embedding"),
    Path("/kaggle/input/maekeso/birdclef2026-exp010-nb1-embedding"),
    Path("/kaggle/input/birdclef2026-exp010-nb1-embedding"),
]
EMB_DIR = next((p for p in _emb_candidates if p.exists()), _emb_candidates[0])
if not EMB_DIR.exists():
    for _hit in Path("/kaggle/input").rglob("soundscape_embeddings.npz"):
        EMB_DIR = _hit.parent; break'''

# Modify cells
new_cells = []
inserted = False
for c in nb["cells"]:
    if c.get("id") == "config":
        src = "".join(c["source"])
        if OLD_BLOCK in src:
            src = src.replace(OLD_BLOCK, NEW_BLOCK)
            c["source"] = src.splitlines(keepends=True)
    if c.get("id") == "blend" and not inserted:
        new_cells.append(code_cell("e20-infer", e20_infer_src))
        inserted = True
    if c.get("id") == "blend":
        new_cells.append(code_cell("blend", blend_aves_src))
    else:
        new_cells.append(c)

# Update hdr
for c in new_cells:
    if c.get("id") == "hdr":
        c["source"] = """# exp063 — exp020 R2 5-fold Aves-only 15% blend on exp048

**Base**: exp048 (LB 0.950) — 3-way blend + TopN N=1 PP

**新規追加**: exp020 R2 5-fold (LB 0.915, eca_nfnet_l0 SED) を **Aves species のみに 15% weight で blend**
- non-Aves species: exp048 のまま (drag 回避)
- Aves 162 species: 0.85 × base + 0.15 × exp020 R2 (rank space)

**vs exp060 (uniform 10%)**:
  exp060: 全 234 species で uniform weight
  exp063: Aves species のみ weight 15% (non-Aves 影響ゼロ)
  → Aves が真に強い model なら exp063 が優位

**vs exp062 (exp032 R2 Aves-only)**:
  exp062: exp032 R2 (LB 0.887、Babych b3 init)
  exp063: exp020 R2 5-fold (LB 0.915、より強い ingredient)

**Expected LB**: 0.951-0.955 (Aves 強化、+0.001-0.005)
**CPU**: ~85-95 min (cap ギリギリ)
""".splitlines(keepends=True)
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
