"""Build Kaggle NB: exp083 R1 pseudo gen on train_soundscapes.

- Load exp083 R1 ckpt from Kaggle Dataset
- Infer on 10,658 train_soundscapes (12 chunks × 5s each = ~127K rows)
- Save pseudo_labels.csv
- Upload to birdclef2026-exp083-weights Dataset (preserving existing files)
"""
import json
import shutil
from pathlib import Path

OUT_DIR = Path("experiment/exp083/notebook")

# ============================================================
# Build pseudo gen NB
# ============================================================
cells = []

# Cell 0: header
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# exp083 R1 Pseudo Generation (Kaggle T4x2)\n",
        "\n",
        "## Purpose\n",
        "exp083 R1 (convnext_pico, 5s Hi-freq mel) を全 train_soundscapes に推論し、\n",
        "**R2 訓練の pseudo_sc source** として使う soft label CSV を生成。\n",
        "\n",
        "## Spec\n",
        "- Window: 5s × 12 per file (60s / 5s = 12)\n",
        "- Mel: Hi-freq (N_FFT=4096, HOP=512, N_MELS=256, FMIN=20)\n",
        "- Backbone: convnext_pico.d1_in1k (R1 と同 spec)\n",
        "- Smoothing: Gaussian sigma=0.65 across 12 windows\n",
        "- Power transform: γ=1.2 (soft label sharpening)\n",
        "\n",
        "## Output\n",
        "- `/kaggle/working/pseudo_labels.csv` (127,896 rows × 237 cols)\n",
        "- Upload to `maekeso/birdclef2026-exp083-weights/r1_pseudo/pseudo_labels.csv`\n",
        "\n",
        "## Required input\n",
        "- `birdclef-2026` (competition、train_soundscapes)\n",
        "- `maekeso/birdclef2026-exp083-weights` (R1 ckpts)\n",
        "\n",
        "## Compute\n",
        "- Kaggle T4x2 GPU (~10-15 min for 10,658 files)\n",
    ]
})

# Cell 1: setup
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 1: Setup\n",
        "# ============================================================\n",
        "import os, time, json, gc, glob\n",
        "from pathlib import Path\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torchaudio\n",
        "import timm\n",
        "from scipy.ndimage import gaussian_filter1d\n",
        "import soundfile as sf\n",
        "import librosa\n",
        "from tqdm.auto import tqdm\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f'Device: {device}')\n",
        "if torch.cuda.is_available():\n",
        "    print(f'GPU: {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}')\n",
        "\n",
        "from kaggle.api.kaggle_api_extended import KaggleApi\n",
        "api = KaggleApi(); api.authenticate()\n",
        "print('Kaggle auth OK')\n",
    ]
})

# Cell 2: Paths + Config
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 2: Paths + Config\n",
        "# ============================================================\n",
        "# Competition\n",
        "BASE = None\n",
        "for p in [Path('/kaggle/input/competitions/birdclef-2026'),\n",
        "          Path('/kaggle/input/birdclef-2026')]:\n",
        "    if p.exists():\n",
        "        BASE = p; break\n",
        "assert BASE is not None\n",
        "TS_DIR = BASE / 'train_soundscapes'\n",
        "SAMPLE_SUB_PATH = BASE / 'sample_submission.csv'\n",
        "print(f'TS_DIR: {TS_DIR} (exists: {TS_DIR.exists()})')\n",
        "\n",
        "# R1 ckpt (Kaggle Dataset attach)\n",
        "DATASET_SLUG = 'birdclef2026-exp083-weights'\n",
        "DATASET_ROOTS = [\n",
        "    Path('/kaggle/input/datasets/maekeso') / DATASET_SLUG,   # ★ Kaggle dataset_sources mount\n",
        "    Path('/kaggle/input') / DATASET_SLUG,                     # fallback\n",
        "]\n",
        "DATASET_ROOT = None\n",
        "for p in DATASET_ROOTS:\n",
        "    if p.exists():\n",
        "        DATASET_ROOT = p; break\n",
        "assert DATASET_ROOT is not None, f'Dataset not mounted: {DATASET_ROOTS}'\n",
        "print(f'DATASET_ROOT: {DATASET_ROOT}')\n",
        "\n",
        "# Find R1 best ckpt\n",
        "R1_CKPT = None\n",
        "for cand in ['ckpt_best_ns22.pth', 'ckpt_best_macro.pth', 'ckpt_latest.pth']:\n",
        "    p = DATASET_ROOT / cand\n",
        "    if p.exists():\n",
        "        R1_CKPT = p; break\n",
        "if R1_CKPT is None:\n",
        "    # rglob fallback\n",
        "    for f in DATASET_ROOT.rglob('ckpt_best_ns22.pth'):\n",
        "        R1_CKPT = f; break\n",
        "assert R1_CKPT is not None, 'R1 ckpt not found'\n",
        "print(f'R1 ckpt: {R1_CKPT} ({R1_CKPT.stat().st_size/1e6:.1f}MB)')\n",
        "\n",
        "# Config (must match exp083 R1 training)\n",
        "SR = 32000\n",
        "TRAIN_DURATION = 5\n",
        "TRAIN_SAMPLES  = SR * TRAIN_DURATION   # 160000\n",
        "N_FFT          = 4096   # ★ exp083 Hi-freq\n",
        "HOP_LENGTH     = 512    # 5s で 313 frames\n",
        "N_MELS         = 256\n",
        "FMIN           = 20\n",
        "FMAX           = 16000\n",
        "WIN_LENGTH     = N_FFT\n",
        "NUM_CLASSES    = 234\n",
        "BACKBONE = 'convnext_pico.d1_in1k'\n",
        "USE_PERCH_DISTILL = True\n",
        "PERCH_EMBED_DIM = 1536\n",
        "\n",
        "# Pseudo gen config (memory standard)\n",
        "N_WINDOWS    = 12    # 60s / 5s = 12\n",
        "GAUSS_SIGMA  = 0.65\n",
        "POWER_GAMMA  = 1.2\n",
        "\n",
        "# Labels (from sample_submission column order)\n",
        "sample_sub = pd.read_csv(SAMPLE_SUB_PATH)\n",
        "PRIMARY_LABELS = sample_sub.columns[1:].tolist()\n",
        "assert len(PRIMARY_LABELS) == NUM_CLASSES\n",
        "print(f'Labels: {len(PRIMARY_LABELS)} species')\n",
        "\n",
        "OUT_DIR = Path('/kaggle/working')\n",
        "OUT_DIR.mkdir(parents=True, exist_ok=True)\n",
    ]
})

# Cell 3: Model class
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 3: Model class (must match exp083 R1 training)\n",
        "# ============================================================\n",
        "class MelSpecTransform(nn.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "        self.mel_spec = torchaudio.transforms.MelSpectrogram(\n",
        "            sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,\n",
        "            win_length=WIN_LENGTH,\n",
        "            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,\n",
        "        )\n",
        "        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)\n",
        "    def forward(self, waveform):\n",
        "        return self.db_transform(self.mel_spec(waveform))\n",
        "\n",
        "\n",
        "class GeMFreqPool(nn.Module):\n",
        "    def __init__(self, p_init=3.0, eps=1e-6):\n",
        "        super().__init__()\n",
        "        self.p = nn.Parameter(torch.tensor(float(p_init)))\n",
        "        self.eps = eps\n",
        "    def forward(self, x):\n",
        "        p = self.p.clamp(min=1.0)\n",
        "        x = x.clamp(min=self.eps).pow(p)\n",
        "        x = x.mean(dim=2)\n",
        "        return x.pow(1.0 / p)\n",
        "\n",
        "\n",
        "class DistillHead(nn.Module):\n",
        "    def __init__(self, backbone_dim, embed_dim=1536):\n",
        "        super().__init__()\n",
        "        self.proj = nn.Linear(backbone_dim, embed_dim)\n",
        "    def forward(self, feature_map):\n",
        "        return self.proj(feature_map.mean(dim=[2, 3]))\n",
        "\n",
        "\n",
        "class BirdSEDModel(nn.Module):\n",
        "    def __init__(self, backbone_name=BACKBONE, num_classes=NUM_CLASSES,\n",
        "                 drop_path_rate=0.0, hidden_dim=512):\n",
        "        super().__init__()\n",
        "        self.backbone = timm.create_model(\n",
        "            backbone_name, pretrained=False, in_chans=1,\n",
        "            num_classes=0, global_pool='', drop_path_rate=drop_path_rate,\n",
        "        )\n",
        "        with torch.no_grad():\n",
        "            n_tf = TRAIN_SAMPLES // HOP_LENGTH + 1\n",
        "            dummy = torch.randn(1, 1, N_MELS, n_tf)\n",
        "            feat = self.backbone(dummy)\n",
        "            self.backbone_dim = feat.shape[1]\n",
        "        self.gem_freq = GeMFreqPool(p_init=3.0)\n",
        "        self.dense = nn.Sequential(\n",
        "            nn.Dropout(0.25),\n",
        "            nn.Linear(self.backbone_dim, hidden_dim),\n",
        "            nn.ReLU(inplace=True),\n",
        "            nn.Dropout(0.5),\n",
        "        )\n",
        "        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)\n",
        "        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)\n",
        "        if USE_PERCH_DISTILL:\n",
        "            self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)\n",
        "\n",
        "    def forward(self, x, return_framewise=False):\n",
        "        h = self.backbone(x)\n",
        "        h_cls = self.gem_freq(h)\n",
        "        h_cls = h_cls.permute(0, 2, 1)\n",
        "        h_cls = self.dense(h_cls)\n",
        "        h_cls = h_cls.permute(0, 2, 1)\n",
        "        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)\n",
        "        framewise = self.cla(h_cls)\n",
        "        clip_logits = torch.sum(norm_att * framewise, dim=2)\n",
        "        if return_framewise:\n",
        "            return clip_logits, framewise.permute(0, 2, 1)\n",
        "        return clip_logits\n",
        "\n",
        "print('OK model class ready')\n",
    ]
})

# Cell 4: Load R1 ckpt
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 4: Load R1 ckpt\n",
        "# ============================================================\n",
        "state = torch.load(str(R1_CKPT), map_location='cpu', weights_only=False)\n",
        "print(f'  epoch: {state.get(\"epoch\", \"?\")}')\n",
        "print(f'  best_ns22:  {state.get(\"best_ns22\", float(\"nan\")):.4f}')\n",
        "print(f'  best_macro: {state.get(\"best_macro\", float(\"nan\")):.4f}')\n",
        "\n",
        "model = BirdSEDModel().to(device)\n",
        "msg = model.load_state_dict(state['model_state'], strict=False)\n",
        "model.eval()\n",
        "model = model.to(memory_format=torch.channels_last)\n",
        "print(f'  Load: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}')\n",
        "print(f'  Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')\n",
        "\n",
        "mel_tf = MelSpecTransform().to(device)\n",
        "print('OK ckpt + mel transform ready')\n",
    ]
})

# Cell 5: Pseudo gen loop
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 5: Pseudo generation on all train_soundscapes\n",
        "# ============================================================\n",
        "CHUNK_N = SR * TRAIN_DURATION\n",
        "\n",
        "def load_60s(path):\n",
        "    '''Load 60s audio, reshape into (N_WINDOWS, CHUNK_N) chunks.'''\n",
        "    try:\n",
        "        wav, sr = sf.read(str(path), dtype='float32', always_2d=False)\n",
        "        if wav.ndim > 1: wav = wav.mean(axis=1)\n",
        "        if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)\n",
        "    except Exception as e:\n",
        "        print(f'WARN load fail {path.name}: {e}')\n",
        "        return np.zeros((N_WINDOWS, CHUNK_N), dtype=np.float32)\n",
        "    target = N_WINDOWS * CHUNK_N\n",
        "    if len(wav) < target: wav = np.pad(wav, (0, target - len(wav)))\n",
        "    elif len(wav) > target: wav = wav[:target]\n",
        "    return wav.astype(np.float32).reshape(N_WINDOWS, CHUNK_N)\n",
        "\n",
        "sc_files = sorted(glob.glob(str(TS_DIR / '*.ogg')))\n",
        "print(f'\\ntrain_soundscapes: {len(sc_files)} files')\n",
        "assert len(sc_files) > 0\n",
        "\n",
        "all_filenames, all_start_secs, all_end_secs, all_probs = [], [], [], []\n",
        "t0 = time.time()\n",
        "\n",
        "with torch.no_grad():\n",
        "    for fi, fpath in enumerate(tqdm(sc_files, desc='pseudo', mininterval=5.0)):\n",
        "        stem = Path(fpath).stem\n",
        "        chunks = load_60s(Path(fpath))\n",
        "        \n",
        "        # To tensor (B=12, 1, 160000)\n",
        "        wav_t = torch.from_numpy(chunks).unsqueeze(1).to(device)\n",
        "        \n",
        "        # Mel + per-chunk normalize\n",
        "        mel = mel_tf(wav_t)\n",
        "        for i in range(mel.size(0)):\n",
        "            mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)\n",
        "        mel = mel.to(memory_format=torch.channels_last)\n",
        "        \n",
        "        # Forward (no autocast for stability)\n",
        "        clip_logits, framewise = model(mel, return_framewise=True)\n",
        "        frame_max = framewise.max(dim=1).values\n",
        "        p_clip = torch.sigmoid(clip_logits).float().cpu().numpy()\n",
        "        p_fmax = torch.sigmoid(frame_max).float().cpu().numpy()\n",
        "        probs_file = 0.5 * p_clip + 0.5 * p_fmax    # (12, 234)\n",
        "        \n",
        "        # Gaussian smoothing across 12 windows\n",
        "        probs_file = gaussian_filter1d(probs_file, sigma=GAUSS_SIGMA, axis=0,\n",
        "                                        mode='nearest').astype(np.float32)\n",
        "        \n",
        "        all_probs.append(probs_file)\n",
        "        for wi in range(N_WINDOWS):\n",
        "            all_filenames.append(stem)\n",
        "            all_start_secs.append(wi * TRAIN_DURATION)\n",
        "            all_end_secs.append((wi + 1) * TRAIN_DURATION)\n",
        "\n",
        "prob_mat = np.concatenate(all_probs, axis=0).astype(np.float32)\n",
        "filenames_arr = np.array(all_filenames)\n",
        "start_secs_arr = np.array(all_start_secs, dtype=np.float32)\n",
        "end_secs_arr   = np.array(all_end_secs,   dtype=np.float32)\n",
        "print(f'\\nInference: shape={prob_mat.shape} in {(time.time()-t0)/60:.1f} min')\n",
        "print(f'  Pre-PT mean={prob_mat.mean():.4f}, max={prob_mat.max():.4f}, 99%ile={np.percentile(prob_mat, 99):.4f}')\n",
    ]
})

# Cell 6: Power transform + save CSV
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 6: Power transform γ=1.2 + save pseudo CSV\n",
        "# ============================================================\n",
        "# Power transform (soft label sharpening)\n",
        "prob_mat = np.power(prob_mat, POWER_GAMMA).astype(np.float32)\n",
        "print(f'Post-PT γ={POWER_GAMMA}: mean={prob_mat.mean():.4f}, max={prob_mat.max():.4f}, 99%ile={np.percentile(prob_mat, 99):.4f}')\n",
        "\n",
        "row_max = prob_mat.max(axis=1)\n",
        "print(f'row_max stats: 50%ile={np.percentile(row_max, 50):.4f}, 99%ile={np.percentile(row_max, 99):.4f}')\n",
        "print(f'  >0.5 cells: {(prob_mat > 0.5).sum()}, >0.7: {(prob_mat > 0.7).sum()}, >0.9: {(prob_mat > 0.9).sum()}')\n",
        "\n",
        "# Build DataFrame\n",
        "df = pd.DataFrame(prob_mat, columns=PRIMARY_LABELS)\n",
        "df.insert(0, 'filename',  filenames_arr)\n",
        "df.insert(1, 'start_sec', start_secs_arr)\n",
        "df.insert(2, 'end_sec',   end_secs_arr)\n",
        "\n",
        "# Save\n",
        "csv_path = OUT_DIR / 'pseudo_labels.csv'\n",
        "df.to_csv(csv_path, index=False)\n",
        "print(f'\\nSaved: {csv_path} ({csv_path.stat().st_size/1024/1024:.1f}MB)')\n",
        "print(f'  shape: {df.shape}, files: {df[\"filename\"].nunique()}')\n",
        "print(f'  example head:')\n",
        "print(df.iloc[:3, :8])\n",
        "\n",
        "# Meta file\n",
        "meta = {\n",
        "    'backbone': BACKBONE,\n",
        "    'round': 'R1',\n",
        "    'ckpt': str(R1_CKPT.name),\n",
        "    'epoch': state.get('epoch'),\n",
        "    'best_ns22': state.get('best_ns22'),\n",
        "    'best_macro': state.get('best_macro'),\n",
        "    'n_files': int(df['filename'].nunique()),\n",
        "    'n_rows': int(len(df)),\n",
        "    'gauss_sigma': GAUSS_SIGMA,\n",
        "    'power_gamma': POWER_GAMMA,\n",
        "    'n_mels': N_MELS, 'n_fft': N_FFT, 'hop_length': HOP_LENGTH,\n",
        "    'fmin': FMIN, 'fmax': FMAX,\n",
        "}\n",
        "meta_path = OUT_DIR / 'pseudo_meta.json'\n",
        "meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')\n",
        "print(f'meta saved: {meta_path}')\n",
        "\n",
        "# Free memory\n",
        "del prob_mat, df, all_probs, all_filenames, model\n",
        "gc.collect(); torch.cuda.empty_cache()\n",
    ]
})

# Cell 7: Upload to Kaggle Dataset (preserve existing + add r1_pseudo/)
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 7: Upload pseudo CSV to Kaggle Dataset (preserve existing files)\n",
        "# ============================================================\n",
        "import tempfile\n",
        "import shutil\n",
        "\n",
        "USER = 'maekeso'\n",
        "SLUG = DATASET_SLUG      # birdclef2026-exp083-weights\n",
        "TITLE = SLUG.replace('-', ' ')\n",
        "SUBFOLDER = 'r1_pseudo'\n",
        "\n",
        "print(f'Upload target: {USER}/{SLUG} → {SUBFOLDER}/')\n",
        "\n",
        "with tempfile.TemporaryDirectory() as td:\n",
        "    td = Path(td)\n",
        "    \n",
        "    # [1/2] Preserve existing files (from attached Dataset)\n",
        "    print(f'\\n[1/2] Preserving existing files...')\n",
        "    n_preserved = 0\n",
        "    for f in DATASET_ROOT.rglob('*'):\n",
        "        if not f.is_file(): continue\n",
        "        if f.name.startswith('__'): continue   # skip metadata files\n",
        "        rel = f.relative_to(DATASET_ROOT)\n",
        "        dst = td / rel\n",
        "        dst.parent.mkdir(parents=True, exist_ok=True)\n",
        "        shutil.copy2(str(f), str(dst))\n",
        "        n_preserved += 1\n",
        "    print(f'  Preserved {n_preserved} existing files')\n",
        "    \n",
        "    # [2/2] Add pseudo files in r1_pseudo/ subfolder\n",
        "    print(f'\\n[2/2] Adding pseudo CSV + meta to {SUBFOLDER}/')\n",
        "    sub_dst = td / SUBFOLDER\n",
        "    sub_dst.mkdir(parents=True, exist_ok=True)\n",
        "    shutil.copy2(str(csv_path), str(sub_dst / 'pseudo_labels.csv'))\n",
        "    shutil.copy2(str(meta_path), str(sub_dst / 'pseudo_meta.json'))\n",
        "    print(f'  Added: {SUBFOLDER}/pseudo_labels.csv ({csv_path.stat().st_size/1024/1024:.1f}MB)')\n",
        "    print(f'  Added: {SUBFOLDER}/pseudo_meta.json')\n",
        "    \n",
        "    # Metadata\n",
        "    meta_dict = {\n",
        "        'title': TITLE,\n",
        "        'id': f'{USER}/{SLUG}',\n",
        "        'licenses': [{'name': 'CC0-1.0'}],\n",
        "    }\n",
        "    (td / 'dataset-metadata.json').write_text(json.dumps(meta_dict, indent=2), encoding='utf-8')\n",
        "    \n",
        "    # Upload\n",
        "    version_notes = f'Added R1 pseudo_labels.csv ({SUBFOLDER}/)'\n",
        "    print(f'\\nUploading: {version_notes}')\n",
        "    try:\n",
        "        t0 = time.time()\n",
        "        api.dataset_create_version(\n",
        "            folder=str(td), version_notes=version_notes,\n",
        "            dir_mode='zip', quiet=False,\n",
        "        )\n",
        "        print(f'\\nOK Uploaded ({time.time()-t0:.0f}s)')\n",
        "        print(f'URL: https://www.kaggle.com/datasets/{USER}/{SLUG}')\n",
        "    except Exception as e:\n",
        "        msg = str(e)[:600]\n",
        "        print(f'FAILED: {type(e).__name__}: {msg}')\n",
        "        # Pseudo CSV still in /kaggle/working/ for manual DL\n",
        "        print(f'\\nManual fallback: download {csv_path} from kernel output')\n",
    ]
})

# Build NB
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

nb_path = OUT_DIR / "nb_pseudo_gen_r1.ipynb"
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"OK Built {nb_path} ({len(cells)} cells)")

# Build push script
push_src = '''"""Push exp083 R1 pseudo gen NB to Kaggle (T4x2 + internet)."""
import json, os, sys, io, tempfile, shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

if not os.environ.get("KAGGLE_API_TOKEN"):
    creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    if creds.get("key", "").startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = creds["key"]

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB = Path(__file__).with_name("nb_pseudo_gen_r1.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-exp083-pseudo-gen-r1"
TITLE = "birdclef2026 exp083 pseudo gen r1"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": TITLE,
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "machine_shape": "NvidiaTeslaT4",   # T4x2 GPU for pseudo inference speed
        "enable_internet": True,            # for Kaggle Dataset upload
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            f"{USER}/birdclef2026-exp083-weights",
        ],
        "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None))
    print("Version:", getattr(r, "version_number", None))
'''

push_path = OUT_DIR / "_push_nb_pseudo_gen_r1.py"
push_path.write_text(push_src, encoding="utf-8")
print(f"OK Built {push_path}")

# Syntax check
import ast
nb = json.loads(nb_path.read_text(encoding="utf-8"))
n_ok, n_err = 0, 0
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                      for ln in src.splitlines())
    try:
        ast.parse(clean); n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  SyntaxError: {e}")
print(f"\nSyntax: {n_ok} OK, {n_err} ERRORS")
