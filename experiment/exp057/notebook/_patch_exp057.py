"""Patch exp057: 5-way blend = exp048 (3 streams) + Stage 2A (all species 8%).

Note: True 5-way requires another stream (exp037 v313 or similar) which has no
easily-mountable ckpt Dataset. So practical "5-way attempt" = 4-way with Stage 2A
at uniform low weight (8%) on all species.

Different from exp056 (Aves-only 15%) — this tests all-species low-weight diversity.
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp057\notebook\nb_blend_5way.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

# Stage 2A inference + uniform 8% blend
STAGE2A_CELL_SRC = '''# ============================================================
# Stage 2A inference + uniform all-species blend at 8% (exp057)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchaudio
import soundfile as sf
import librosa

STAGE2A_CKPT_CANDIDATES = [
    Path("/kaggle/input/birdclef2026-exp053-stage2a-effv2s/stage2a_best.pth"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp053-stage2a-effv2s/stage2a_best.pth"),
]
STAGE2A_CKPT = next((p for p in STAGE2A_CKPT_CANDIDATES if p.exists()), None)
assert STAGE2A_CKPT is not None, f"Stage 2A ckpt not found in {STAGE2A_CKPT_CANDIDATES}"
print(f"Stage 2A ckpt: {STAGE2A_CKPT}")

S2A_SR = 32000
S2A_CHUNK_LEN = 32000 * 5
S2A_N_MELS = 128
S2A_N_FFT = 2048
S2A_HOP = 512

class S2AMelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=S2A_SR, n_fft=S2A_N_FFT, hop_length=S2A_HOP,
            n_mels=S2A_N_MELS, f_min=20, f_max=16000,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)
    def forward(self, wav):
        mel = self.mel(wav); mel = self.db(mel)
        mel = torch.clamp(mel, -80.0, 0.0); mel = (mel + 40.0) / 40.0
        return mel

class S2ASEDHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.att = nn.Linear(in_dim, n_classes)
        self.cla = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        att = torch.tanh(self.att(x))
        cla = self.cla(x)
        norm_att = F.softmax(att, dim=1)
        return (norm_att * cla).sum(dim=1)

class S2ASEDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s.in21k_ft_in1k", pretrained=False, in_chans=3,
            num_classes=0, global_pool="",
        )
        feat_dim = self.backbone.num_features
        self.head = S2ASEDHead(feat_dim, N_CLASSES)
    def forward(self, mel):
        x = mel.unsqueeze(1).repeat(1, 3, 1, 1)
        feat = self.backbone(x).mean(dim=2).transpose(1, 2)
        return self.head(feat)

s2a_model = S2ASEDModel()
_s2a_ckpt = torch.load(STAGE2A_CKPT, map_location="cpu", weights_only=False)
_state = _s2a_ckpt.get("state_dict", _s2a_ckpt)
_state = {k.replace("module.", ""): v for k, v in _state.items()}
s2a_model.load_state_dict(_state, strict=False)
s2a_model.eval()
s2a_mel_ex = S2AMelExtractor()
print(f"Stage 2A model loaded, val_ns22: {_s2a_ckpt.get('val_ns22', 'n/a')}")

# Run inference on test_files
probs_stage2a = np.zeros_like(probs_blend, dtype=np.float32)
print(f"Running Stage 2A on {probs_blend.shape[0]} files...")
_t0_s2a = time.time()
torch.set_grad_enabled(False)
torch.set_num_threads(4)

for fi, test_file in enumerate(test_files):
    try:
        wav, sr = sf.read(str(test_file), dtype="float32")
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != S2A_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=S2A_SR)
        n_chunks = min(N_WINDOWS, max(1, len(wav) // S2A_CHUNK_LEN))
        chunks = []
        for ci in range(n_chunks):
            seg = wav[ci*S2A_CHUNK_LEN:(ci+1)*S2A_CHUNK_LEN]
            if len(seg) < S2A_CHUNK_LEN:
                seg = np.pad(seg, (0, S2A_CHUNK_LEN - len(seg)))
            chunks.append(seg)
        if len(chunks) < N_WINDOWS:
            chunks += [np.zeros(S2A_CHUNK_LEN, dtype=np.float32)] * (N_WINDOWS - len(chunks))
        batch = torch.from_numpy(np.stack(chunks)).float()
        mel = s2a_mel_ex(batch)
        logit = s2a_model(mel)
        prob = torch.sigmoid(logit).numpy()
        probs_stage2a[fi] = prob[:N_WINDOWS]
    except Exception as e:
        print(f"  [s2a err] {test_file.name}: {str(e)[:80]}")
    if (fi + 1) % 50 == 0:
        elapsed = time.time() - _t0_s2a
        rate = (fi + 1) / elapsed
        eta = (probs_blend.shape[0] - fi - 1) / rate
        print(f"  [{fi+1}/{probs_blend.shape[0]}] {rate*60:.1f}/min ETA {eta/60:.1f}min")

print(f"Stage 2A done in {(time.time()-_t0_s2a)/60:.1f}min")

# Uniform all-species blend at 8% Stage 2A
S2A_WEIGHT = 0.08
print(f"Applying uniform Stage 2A blend at {S2A_WEIGHT*100:.0f}%...")
probs_blend = (1 - S2A_WEIGHT) * probs_blend + S2A_WEIGHT * probs_stage2a
print(f"  probs_blend after Stage 2A merge: mean={probs_blend.mean():.4f}")

# Cleanup
del s2a_model, s2a_mel_ex, _s2a_ckpt, _state
import gc; gc.collect()
'''

inserted = False
new_cells = []
for c in nb["cells"]:
    if c.get("cell_type") == "code":
        src = "".join(c["source"])
        if "preds_array = probs_blend.reshape" in src and not inserted:
            new_cells.append({
                "cell_type": "code",
                "id": "stage2a_blend",
                "metadata": {},
                "source": STAGE2A_CELL_SRC.splitlines(keepends=True),
                "execution_count": None, "outputs": [],
            })
            inserted = True
            print("[exp057] Stage 2A cell inserted")
    new_cells.append(c)

if inserted:
    nb["cells"] = new_cells

# Update hdr
for c in nb["cells"]:
    if c.get("id") == "hdr":
        src = "".join(c["source"])
        if "Stage 2A uniform 8%" not in src:
            prefix = """# exp057: 4/5-way blend with Stage 2A (all species 8%)

**Base**: exp048 (LB 0.950)
**新規変更**: Stage 2A を **全 234 species に uniform 8% weight** で blend (4-way)
- 全 species: 0.92 × exp048 + 0.08 × Stage 2A
- (注: 真の 5-way には 5 番目 stream が必要、現状 4-way で diversity 補強)

**vs exp056 (Aves-only 15%)**:
- exp057: all species 8%
- exp056: Aves only 15%
- ablation で blend strategy 比較

**期待 LB**: 0.948-0.952 (Stage 2A drag リスクあり、特に non-Aves)

---

"""
            c["source"] = (prefix + src).splitlines(keepends=True)
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB}")
