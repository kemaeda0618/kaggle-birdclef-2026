"""Fix insertion: move Stage 2A code INSIDE blend cell, between probs_blend def and preds_array use.

For exp056: Aves-only 15% blend
For exp057: uniform 8% blend
"""
import json
from pathlib import Path

EXP_CONFIGS = [
    {
        "nb": Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp056\notebook\nb_blend_aves_stage2a.ipynb"),
        "merge_code": """
# === Aves-only Stage 2A blend (exp056) ===
AVES_MASK = np.array([LABEL_TO_CLASS.get(lbl) == "Aves" for lbl in PRIMARY_LABELS], dtype=bool)
print(f"  Aves species: {AVES_MASK.sum()} (expected 162)")
S2A_AVES_WEIGHT = 0.15
print(f"  Applying Aves-only Stage 2A blend at {S2A_AVES_WEIGHT*100:.0f}% on Aves species only...")
probs_blend_new = probs_blend.copy()
probs_blend_new[:, :, AVES_MASK] = (
    (1 - S2A_AVES_WEIGHT) * probs_blend[:, :, AVES_MASK] +
    S2A_AVES_WEIGHT * probs_stage2a[:, :, AVES_MASK]
)
probs_blend = probs_blend_new
print(f"  probs_blend after Aves merge: mean={probs_blend.mean():.4f}, max={probs_blend.max():.4f}")
""",
    },
    {
        "nb": Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp057\notebook\nb_blend_5way.ipynb"),
        "merge_code": """
# === Uniform Stage 2A blend at 8% (exp057) ===
S2A_WEIGHT = 0.08
print(f"  Applying uniform Stage 2A blend at {S2A_WEIGHT*100:.0f}%...")
probs_blend = (1 - S2A_WEIGHT) * probs_blend + S2A_WEIGHT * probs_stage2a
print(f"  probs_blend after Stage 2A merge: mean={probs_blend.mean():.4f}, max={probs_blend.max():.4f}")
""",
    },
]

# Stage 2A inference code (same for both, no merge at end)
STAGE2A_INFER_CODE = """
# === Stage 2A inference (run after probs_blend computed) ===
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchaudio

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

class S2AMelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=S2A_SR, n_fft=2048, hop_length=512,
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
        att = torch.tanh(self.att(x)); cla = self.cla(x)
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
        print(f"  [{fi+1}/{probs_blend.shape[0]}] {rate*60:.1f}/min")

print(f"Stage 2A inference done in {(time.time()-_t0_s2a)/60:.1f}min")
del s2a_model, s2a_mel_ex, _s2a_ckpt, _state
import gc; gc.collect()
"""

for cfg in EXP_CONFIGS:
    NB = cfg["nb"]
    merge_code = cfg["merge_code"]
    nb = json.load(open(NB, encoding="utf-8"))
    print(f"\n=== {NB.name} ===")

    # Step 1: Remove the bad standalone stage2a_blend cell (if exists)
    nb["cells"] = [c for c in nb["cells"] if c.get("id") != "stage2a_blend"]

    # Step 2: Find blend cell with both `probs_blend = ` and `preds_array = `
    # Insert Stage 2A inference + merge BEFORE `preds_array = probs_blend.reshape(...)`
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"])
        if "probs_blend = blend_flat.reshape" in src and "preds_array = probs_blend.reshape" in src:
            # Inject Stage 2A inference + merge before preds_array line
            insertion = STAGE2A_INFER_CODE + merge_code
            target = "preds_array = probs_blend.reshape(-1, N_CLASSES)"
            if target in src and "STAGE2A_CKPT" not in src:
                src = src.replace(target, insertion + "\n\n" + target)
                c["source"] = src.splitlines(keepends=True)
                print(f"  [OK] Stage 2A inserted before preds_array (in same cell)")
                break

    json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  Saved {NB}")
