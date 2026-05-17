import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
import torchaudio.transforms as T
import os, time, glob
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

START = time.time()
print(f'PyTorch {torch.__version__}')

# ══════════════════════════════════════════════════════════════
# CONFIG & PATHS
# ══════════════════════════════════════════════════════════════
@dataclass
class Config:
    sr: int = 32_000
    chunk_duration: float = 5.0
    target_size: tuple = (256, 256)
    n_mels: int = 256
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 0
    fmax: int = 16_000
    top_db: float = 80.0
    power: float = 2.0
    norm: str = "slaney"
    mel_scale: str = "htk"
    backbone: str = "tf_efficientnet_b0.ns_jft_in1k"
    num_classes: int = 234
    in_channels: int = 3
    dropout: float = 0.1
    drop_path_rate: float = 0.0
    gem_p_init: float = 3.0

    @property
    def chunk_samples(self) -> int:
        return int(self.sr * self.chunk_duration)

cfg = Config()

DATA_ROOT = "/kaggle/input/competitions/birdclef-2026"
TEST_DIR = os.path.join(DATA_ROOT, "test_soundscapes")
TRAIN_SC_DIR = os.path.join(DATA_ROOT, "train_soundscapes")

# Try multiple possible mount paths for the checkpoint
CHECKPOINT_CANDIDATES = [
    "/kaggle/input/exp006/weights/best_fold0.pth",
    "/kaggle/input/datasets/maekeso/exp006/weights/best_fold0.pth",
]
CHECKPOINT = None
for p in CHECKPOINT_CANDIDATES:
    if os.path.isfile(p):
        CHECKPOINT = p
        break
if CHECKPOINT is None:
    import subprocess
    print("=== /kaggle/input contents ===")
    result = subprocess.run(["find", "/kaggle/input", "-maxdepth", "3", "-type", "f"],
                            capture_output=True, text=True, timeout=10)
    print(result.stdout[:3000])
    raise FileNotFoundError(f"Checkpoint not found in any of: {CHECKPOINT_CANDIDATES}")

print(f"Checkpoint: {CHECKPOINT}")

sub_df = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"), nrows=1)
SPECIES = list(sub_df.columns[1:])
print(f"Species: {len(SPECIES)}")

# ══════════════════════════════════════════════════════════════
# MODEL DEFINITION (must match training)
# ══════════════════════════════════════════════════════════════
class GEMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p_init))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)

class AttentionSEDHead(nn.Module):
    def __init__(self, feat_dim, num_classes, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.att_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)
        self.cls_conv = nn.Conv1d(feat_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.fc(x.permute(0, 2, 1)).permute(0, 2, 1)
        att = F.softmax(torch.tanh(self.att_conv(x)), dim=-1)
        cls = self.cls_conv(x)
        clipwise_logit = (att * cls).sum(dim=-1)
        return {
            "clipwise_prob": torch.sigmoid(clipwise_logit),
            "segmentwise_logit": cls.permute(0, 2, 1),
        }

class SEDModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=False,
            in_chans=cfg.in_channels, features_only=False,
            global_pool="", num_classes=0,
            drop_path_rate=cfg.drop_path_rate,
        )
        self.gem_pool = GEMFreqPool(p_init=cfg.gem_p_init)
        self.head = AttentionSEDHead(self.backbone.num_features,
                                     cfg.num_classes, cfg.dropout)

    def forward(self, x):
        return self.head(self.gem_pool(self.backbone(x)))

# ══════════════════════════════════════════════════════════════
# MEL TRANSFORM
# ══════════════════════════════════════════════════════════════
class MelSpectrogramTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, f_min=cfg.fmin, f_max=cfg.fmax,
            power=cfg.power, norm=cfg.norm, mel_scale=cfg.mel_scale,
        )
        self.db = T.AmplitudeToDB(stype="power", top_db=cfg.top_db)
        self.resize = torchvision.transforms.Resize(cfg.target_size, antialias=True)

    @torch.no_grad()
    def forward(self, waveforms):
        mel = self.db(self.mel(waveforms))
        mel = self.resize(mel)
        B = mel.shape[0]
        mel_flat = mel.reshape(B, -1)
        mel_min = mel_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        mel_max = mel_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
        return mel.unsqueeze(1).repeat(1, 3, 1, 1)

mel_transform = MelSpectrogramTransform(cfg)
mel_transform.eval()

# ══════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════
model = SEDModel(cfg)
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
epoch = ckpt.get('epoch', '?')
val_auc = ckpt.get('metrics', {}).get('macro_auc', '?')
print(f'Loaded: epoch={epoch}, val_auc={val_auc}')

# ══════════════════════════════════════════════════════════════
# FIND TEST FILES
# ══════════════════════════════════════════════════════════════
test_files = sorted(glob.glob(os.path.join(TEST_DIR, '*.ogg')))
if len(test_files) == 0:
    print('No test soundscapes found, using train_soundscapes as fallback')
    test_files = sorted(glob.glob(os.path.join(TRAIN_SC_DIR, '*.ogg')))[:8]
print(f'Test files: {len(test_files)}')

# ══════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════
def load_soundscape(path):
    stem = Path(path).stem
    y, _ = librosa.load(path, sr=cfg.sr, mono=True)
    return y, stem

print('Loading audio files...')
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as executor:
    audio_results = list(executor.map(load_soundscape, test_files))
print(f'  Loaded {len(audio_results)} files in {time.time()-t0:.1f}s')

CHUNK = cfg.chunk_samples
BATCH_SIZE = 32
all_row_ids = []
all_preds = []

print('Running inference...')
t0 = time.time()

for audio, stem in audio_results:
    n_chunks = max(1, len(audio) // CHUNK)
    padded_len = n_chunks * CHUNK
    if len(audio) < padded_len:
        audio = np.pad(audio, (0, padded_len - len(audio)))
    else:
        audio = audio[:padded_len]

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak

    chunks_tensor = torch.from_numpy(audio.reshape(n_chunks, CHUNK)).float()

    # Batched inference to avoid OOM on long soundscapes
    all_chunk_probs = []
    with torch.no_grad():
        for bi in range(0, n_chunks, BATCH_SIZE):
            batch = chunks_tensor[bi:bi+BATCH_SIZE]
            mel = mel_transform(batch)
            probs_batch = model(mel)['clipwise_prob'].numpy()
            all_chunk_probs.append(probs_batch)
    probs = np.concatenate(all_chunk_probs, axis=0)

    for i in range(n_chunks):
        end_sec = (i + 1) * int(cfg.chunk_duration)
        all_row_ids.append(f'{stem}_{end_sec}')
        all_preds.append(probs[i])

elapsed = time.time() - t0
print(f'  Inference done: {len(all_row_ids)} predictions in {elapsed:.1f}s')

# ══════════════════════════════════════════════════════════════
# BUILD SUBMISSION
# ══════════════════════════════════════════════════════════════
preds_array = np.stack(all_preds)
submission = pd.DataFrame(preds_array, columns=SPECIES)
submission.insert(0, 'row_id', all_row_ids)

sample_sub = pd.read_csv(os.path.join(DATA_ROOT, 'sample_submission.csv'))
expected_ids = set(sample_sub['row_id'])
our_ids = set(submission['row_id'])

missing = expected_ids - our_ids
if missing:
    print(f'WARNING: {len(missing)} missing row_ids - filling with zeros')
    missing_df = pd.DataFrame({'row_id': list(missing)})
    for sp in SPECIES:
        missing_df[sp] = 0.0
    submission = pd.concat([submission, missing_df], ignore_index=True)

extra = our_ids - expected_ids
if extra:
    print(f'Dropping {len(extra)} extra row_ids')
    submission = submission[submission['row_id'].isin(expected_ids)]

submission = submission.set_index('row_id').loc[sample_sub['row_id']].reset_index()
submission.to_csv('submission.csv', index=False)

total_time = time.time() - START
print(f'\nSubmission saved: {submission.shape}')
print(f'Total time: {total_time:.0f}s ({total_time/60:.1f} min)')
print(f'Mean prediction: {submission[SPECIES].values.mean():.6f}')
print(f'Max prediction:  {submission[SPECIES].values.max():.6f}')
print(submission.head())