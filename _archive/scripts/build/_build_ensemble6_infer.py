"""Build 6-model ensemble inference NB with speed monitoring.

Models:
1. exp017 R2 (5s, eca_nfnet_l0, Tucker mel) — OV
2. exp081 R2 (10s sliding, eca_nfnet_l0, Tucker mel) — OV
3. exp082 R2 (20s sliding, eca_nfnet_l0, Babych mel) — OV
4. exp015 R2 (5s, convnext_pico, Tucker mel) — OV
5. exp083 R1 (5s, convnext_pico, Hi-freq mel) — PyTorch CPU (no OV yet)
6. exp084 R1 (5s, convnext_pico, Hi-time mel) — PyTorch CPU (no OV yet)

Each model: blend clip + frame_max in logit space, then aggregate.
Final: blend across models in logit space, smooth, sigmoid.

Speed measurement: per-50-files rate + ETA + per-model timing breakdown.
"""
import json
from pathlib import Path

OUT_DIR = Path("experiment/ensemble6/notebook")
OUT_DIR.mkdir(parents=True, exist_ok=True)

cells = []

# ---- Cell 0: Header ----
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 6-model Ensemble Inference (speed measurement)\n",
        "\n",
        "## Models\n",
        "| Model | Backbone | Window | Mel | Format | Weight |\n",
        "|---|---|---|---|---|---|\n",
        "| exp017 R2 | eca_nfnet_l0 | 5s | Tucker | OV | 0.25 |\n",
        "| exp081 R2 | eca_nfnet_l0 | 10s sliding | Tucker | OV | 0.20 |\n",
        "| exp082 R2 | eca_nfnet_l0 | 20s sliding | Babych | OV | 0.10 |\n",
        "| exp015 R2 | convnext_pico | 5s | Tucker | OV | 0.15 |\n",
        "| exp083 R1 | convnext_pico | 5s | Hi-freq | PyTorch | 0.15 |\n",
        "| exp084 R1 | convnext_pico | 5s | Hi-time | PyTorch | 0.15 |\n",
        "\n",
        "## Pipeline\n",
        "Each model: forward → clip_logits + framewise → blend in logit space (0.5/0.5)\n",
        "Cross-model: weighted sum in logit space → smooth → sigmoid\n",
        "\n",
        "## Speed monitoring\n",
        "- Per-50-files progress + ETA\n",
        "- Per-model wall-clock breakdown\n",
        "- Final total time\n",
        "\n",
        "## Inputs\n",
        "- birdclef-2026 (competition)\n",
        "- maekeso/birdclef2026-exp015-weights (OV r2_ov/)\n",
        "- maekeso/birdclef2026-exp017-weights (OV r2_ov/)\n",
        "- maekeso/birdclef2026-exp081-weights (OV r2_ov/)\n",
        "- maekeso/birdclef2026-exp082-weights (OV r2_ov/)\n",
        "- maekeso/birdclef2026-exp083-weights (PyTorch ckpt)\n",
        "- maekeso/birdclef2026-exp084-weights (PyTorch ckpt)\n",
        "- openvino runtime wheel dataset (TBD attach)\n",
    ]
})

# ---- Cell 1: Setup ----
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 1: Setup — internet=off, openvino + torch + scipy\n",
        "# ============================================================\n",
        "import os, sys, time, json, gc, glob\n",
        "from pathlib import Path\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import torchaudio\n",
        "import timm\n",
        "from scipy.ndimage import convolve1d\n",
        "import soundfile as sf\n",
        "import librosa\n",
        "import openvino as ov\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "device = torch.device('cpu')   # Kaggle CPU sub\n",
        "torch.set_num_threads(4)\n",
        "print(f'torch={torch.__version__}, timm={timm.__version__}, ov={ov.__version__}')\n",
        "print(f'Device: {device}, num_threads={torch.get_num_threads()}')\n",
        "\n",
        "GLOBAL_START = time.time()\n",
    ]
})

# ---- Cell 2: Config ----
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 2: Config — mel specs + model paths + ensemble weights\n",
        "# ============================================================\n",
        "SR = 32000\n",
        "N_CLASSES = 234\n",
        "PERCH_EMBED_DIM = 1536\n",
        "USE_PERCH_DISTILL = True\n",
        "\n",
        "# Paths\n",
        "BASE = None\n",
        "for p in [Path('/kaggle/input/competitions/birdclef-2026'),\n",
        "          Path('/kaggle/input/birdclef-2026')]:\n",
        "    if p.exists():\n",
        "        BASE = p; break\n",
        "assert BASE is not None\n",
        "TEST_DIR = BASE / 'test_soundscapes'\n",
        "SAMPLE_SUB = pd.read_csv(BASE / 'sample_submission.csv')\n",
        "PRIMARY_LABELS = SAMPLE_SUB.columns[1:].tolist()\n",
        "assert len(PRIMARY_LABELS) == N_CLASSES\n",
        "\n",
        "# Mel transform configs\n",
        "MEL_CONFIGS = {\n",
        "    'Tucker':  dict(n_mels=256, n_fft=2048, hop=512,  win=2048, fmin=20, fmax=16000),\n",
        "    'Babych':  dict(n_mels=224, n_fft=4096, hop=1252, win=4096, fmin=0,  fmax=16000),\n",
        "    'Hi-freq': dict(n_mels=256, n_fft=4096, hop=512,  win=4096, fmin=20, fmax=16000),\n",
        "    'Hi-time': dict(n_mels=256, n_fft=2048, hop=256,  win=2048, fmin=20, fmax=16000),\n",
        "}\n",
        "\n",
        "# Model configs\n",
        "MODELS_CFG = {\n",
        "    'exp017_r2': {\n",
        "        'dataset_slug': 'birdclef2026-exp017-weights',\n",
        "        'mel': 'Tucker', 'duration': 5, 'backbone': 'eca_nfnet_l0',\n",
        "        'window_type': 'simple',   # simple 12 chunks of 5s\n",
        "        'format': 'ov', 'ov_path': 'r2_ov/model.xml',\n",
        "        'pytorch_ckpt': 'r2_ckpt_best_ns22.pth',\n",
        "        'weight': 0.25,\n",
        "    },\n",
        "    'exp081_r2': {\n",
        "        'dataset_slug': 'birdclef2026-exp081-weights',\n",
        "        'mel': 'Tucker', 'duration': 10, 'backbone': 'eca_nfnet_l0',\n",
        "        'window_type': 'sliding',  # padded 65s, 12 windows of 10s step 5s\n",
        "        'format': 'ov', 'ov_path': 'r2_ov/model.xml',\n",
        "        'pytorch_ckpt': 'r2/ckpt_best_ns22.pth',\n",
        "        'weight': 0.20,\n",
        "    },\n",
        "    'exp082_r2': {\n",
        "        'dataset_slug': 'birdclef2026-exp082-weights',\n",
        "        'mel': 'Babych', 'duration': 20, 'backbone': 'eca_nfnet_l0',\n",
        "        'window_type': 'sliding',  # padded 75s, 12 windows of 20s step 5s\n",
        "        'format': 'ov', 'ov_path': 'r2_ov/model.xml',\n",
        "        'pytorch_ckpt': 'r2/ckpt_best_ns22.pth',\n",
        "        'weight': 0.10,\n",
        "    },\n",
        "    'exp015_r2': {\n",
        "        'dataset_slug': 'birdclef2026-exp015-weights',\n",
        "        'mel': 'Tucker', 'duration': 5, 'backbone': 'convnext_pico.d1_in1k',\n",
        "        'window_type': 'simple',\n",
        "        'format': 'ov', 'ov_path': 'r2_ov/model.xml',\n",
        "        'pytorch_ckpt': 'r2_ckpt_best_ns22.pth',\n",
        "        'weight': 0.15,\n",
        "    },\n",
        "    'exp083_r1': {\n",
        "        'dataset_slug': 'birdclef2026-exp083-weights',\n",
        "        'mel': 'Hi-freq', 'duration': 5, 'backbone': 'convnext_pico.d1_in1k',\n",
        "        'window_type': 'simple',\n",
        "        'format': 'pytorch',\n",
        "        'pytorch_ckpt': 'ckpt_best_ns22.pth',\n",
        "        'weight': 0.15,\n",
        "    },\n",
        "    'exp084_r1': {\n",
        "        'dataset_slug': 'birdclef2026-exp084-weights',\n",
        "        'mel': 'Hi-time', 'duration': 5, 'backbone': 'convnext_pico.d1_in1k',\n",
        "        'window_type': 'simple',\n",
        "        'format': 'pytorch',\n",
        "        'pytorch_ckpt': 'ckpt_best_ns22.pth',\n",
        "        'weight': 0.15,\n",
        "    },\n",
        "}\n",
        "\n",
        "# Verify weights sum to 1.0\n",
        "wsum = sum(c['weight'] for c in MODELS_CFG.values())\n",
        "assert abs(wsum - 1.0) < 1e-6, f'weights sum to {wsum}, not 1.0'\n",
        "print(f'Models: {list(MODELS_CFG.keys())}')\n",
        "print(f'Weights sum: {wsum:.3f}')\n",
        "\n",
        "OUT_DIR = Path('/kaggle/working')\n",
    ]
})

# ---- Cell 3: Model class (PyTorch fallback) ----
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 3: BirdSEDModel class for PyTorch CPU fallback (exp083 R1, exp084 R1)\n",
        "# ============================================================\n",
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
        "    def __init__(self, backbone_name, n_mels, time_frames,\n",
        "                 num_classes=N_CLASSES, drop_path_rate=0.0, hidden_dim=512):\n",
        "        super().__init__()\n",
        "        self.backbone = timm.create_model(\n",
        "            backbone_name, pretrained=False, in_chans=1,\n",
        "            num_classes=0, global_pool='', drop_path_rate=drop_path_rate,\n",
        "        )\n",
        "        with torch.no_grad():\n",
        "            dummy = torch.randn(1, 1, n_mels, time_frames)\n",
        "            feat = self.backbone(dummy)\n",
        "            self.backbone_dim = feat.shape[1]\n",
        "        self.gem_freq = GeMFreqPool(p_init=3.0)\n",
        "        self.dense = nn.Sequential(\n",
        "            nn.Dropout(0.25), nn.Linear(self.backbone_dim, hidden_dim),\n",
        "            nn.ReLU(inplace=True), nn.Dropout(0.5),\n",
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
        "print('OK BirdSEDModel ready')\n",
    ]
})

# ---- Cell 4: Mel transforms ----
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 4: Mel transforms (4 configs)\n",
        "# ============================================================\n",
        "class MelSpecTransform(nn.Module):\n",
        "    def __init__(self, n_fft, hop, win, n_mels, fmin, fmax):\n",
        "        super().__init__()\n",
        "        self.mel = torchaudio.transforms.MelSpectrogram(\n",
        "            sample_rate=SR, n_fft=n_fft, hop_length=hop, win_length=win,\n",
        "            n_mels=n_mels, f_min=fmin, f_max=fmax, power=2.0,\n",
        "        )\n",
        "        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)\n",
        "    def forward(self, wav):\n",
        "        return self.db(self.mel(wav))\n",
        "\n",
        "\n",
        "MEL_TRANSFORMS = {}\n",
        "for name, cfg in MEL_CONFIGS.items():\n",
        "    MEL_TRANSFORMS[name] = MelSpecTransform(\n",
        "        n_fft=cfg['n_fft'], hop=cfg['hop'], win=cfg['win'],\n",
        "        n_mels=cfg['n_mels'], fmin=cfg['fmin'], fmax=cfg['fmax'],\n",
        "    ).to(device)\n",
        "    print(f'  {name}: n_fft={cfg[\"n_fft\"]}, hop={cfg[\"hop\"]}, n_mels={cfg[\"n_mels\"]}')\n",
        "\n",
        "def normalize_mel(mel):\n",
        "    '''Per-sample normalize.'''\n",
        "    for i in range(mel.size(0)):\n",
        "        mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)\n",
        "    return mel\n",
        "\n",
        "print('OK mel transforms ready')\n",
    ]
})

# ---- Cell 5: Load all 6 models ----
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 5: Load all 6 models (OV + PyTorch)\n",
        "# ============================================================\n",
        "core = ov.Core()\n",
        "MODELS = {}   # name → loaded model object\n",
        "MODEL_FORMAT = {}\n",
        "TIME_FRAMES = {}\n",
        "\n",
        "for mname, cfg in MODELS_CFG.items():\n",
        "    print(f'\\n--- Loading {mname} ---')\n",
        "    # Find dataset root (Kaggle mount path)\n",
        "    DATASET_ROOTS = [\n",
        "        Path('/kaggle/input/datasets/maekeso') / cfg['dataset_slug'],\n",
        "        Path('/kaggle/input') / cfg['dataset_slug'],\n",
        "    ]\n",
        "    ds_root = None\n",
        "    for p in DATASET_ROOTS:\n",
        "        if p.exists():\n",
        "            ds_root = p; break\n",
        "    assert ds_root is not None, f'{cfg[\"dataset_slug\"]} not mounted'\n",
        "\n",
        "    # Compute mel input shape\n",
        "    mel_cfg = MEL_CONFIGS[cfg['mel']]\n",
        "    duration = cfg['duration']\n",
        "    n_samples = SR * duration\n",
        "    time_frames = n_samples // mel_cfg['hop'] + 1\n",
        "    TIME_FRAMES[mname] = time_frames\n",
        "    print(f'  mel: {cfg[\"mel\"]} ({mel_cfg[\"n_mels\"]}, {time_frames})')\n",
        "\n",
        "    if cfg['format'] == 'ov':\n",
        "        # Load OpenVINO IR\n",
        "        ov_path = ds_root / cfg['ov_path']\n",
        "        assert ov_path.exists(), f'OV not found: {ov_path}'\n",
        "        ov_model = core.read_model(str(ov_path))\n",
        "        compiled = core.compile_model(ov_model, 'CPU')\n",
        "        MODELS[mname] = compiled\n",
        "        MODEL_FORMAT[mname] = 'ov'\n",
        "        # Print I/O names\n",
        "        inputs = list(compiled.inputs)\n",
        "        outputs = list(compiled.outputs)\n",
        "        print(f'  format: OV')\n",
        "        print(f'  input: {inputs[0].any_name} shape={inputs[0].shape}')\n",
        "        for o in outputs:\n",
        "            print(f'  output: {o.any_name} shape={o.shape}')\n",
        "    else:\n",
        "        # Load PyTorch ckpt\n",
        "        ckpt_path = ds_root / cfg['pytorch_ckpt']\n",
        "        if not ckpt_path.exists():\n",
        "            for f in ds_root.rglob(Path(cfg['pytorch_ckpt']).name):\n",
        "                ckpt_path = f; break\n",
        "        assert ckpt_path.exists(), f'ckpt not found in {ds_root}'\n",
        "        state = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)\n",
        "        model = BirdSEDModel(\n",
        "            backbone_name=cfg['backbone'],\n",
        "            n_mels=mel_cfg['n_mels'],\n",
        "            time_frames=time_frames,\n",
        "        ).eval()\n",
        "        msg = model.load_state_dict(state['model_state'], strict=False)\n",
        "        MODELS[mname] = model\n",
        "        MODEL_FORMAT[mname] = 'pytorch'\n",
        "        print(f'  format: PyTorch CPU')\n",
        "        print(f'  ckpt: {ckpt_path.name}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M')\n",
        "        print(f'  load: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}')\n",
        "\n",
        "print(f'\\n=== Loaded {len(MODELS)} models ===')\n",
    ]
})

# ---- Cell 6: Inference helpers ----
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 6: Inference helpers (per model type)\n",
        "# ============================================================\n",
        "GAUSSIAN_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)\n",
        "N_OUT_ROWS = 12   # 12 rows per 60s file\n",
        "\n",
        "def load_60s(path):\n",
        "    '''Load 60s mono 32kHz audio.'''\n",
        "    try:\n",
        "        wav, sr = sf.read(str(path), dtype='float32', always_2d=False)\n",
        "    except Exception as e:\n",
        "        return np.zeros(SR * 60, dtype=np.float32)\n",
        "    if wav.ndim > 1: wav = wav.mean(axis=1)\n",
        "    if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)\n",
        "    target = SR * 60\n",
        "    if len(wav) < target: wav = np.pad(wav, (0, target - len(wav)))\n",
        "    elif len(wav) > target: wav = wav[:target]\n",
        "    return wav.astype(np.float32)\n",
        "\n",
        "\n",
        "def prepare_input_simple(audio_60s, duration):\n",
        "    '''Simple chunking for 5s models: (12, duration*SR).'''\n",
        "    n_samples = SR * duration   # 5*32000 = 160000\n",
        "    return audio_60s.reshape(N_OUT_ROWS, n_samples)\n",
        "\n",
        "\n",
        "def prepare_input_sliding(audio_60s, duration):\n",
        "    '''Sliding window for 10s/20s: padded + 12 windows step 5s.'''\n",
        "    padded_sec = (N_OUT_ROWS - 1) * 5 + duration   # e.g., 65 (10s), 75 (20s)\n",
        "    pad_lead = (padded_sec - 60) // 2   # symmetric pad\n",
        "    padded_samples = padded_sec * SR\n",
        "    pad_lead_samples = pad_lead * SR\n",
        "    audio_padded = np.zeros(padded_samples, dtype=np.float32)\n",
        "    audio_padded[pad_lead_samples:pad_lead_samples + len(audio_60s)] = audio_60s\n",
        "    chunks = np.zeros((N_OUT_ROWS, SR * duration), dtype=np.float32)\n",
        "    step_samples = 5 * SR\n",
        "    win_samples = duration * SR\n",
        "    for k in range(N_OUT_ROWS):\n",
        "        start = k * step_samples\n",
        "        chunks[k] = audio_padded[start:start + win_samples]\n",
        "    return chunks\n",
        "\n",
        "\n",
        "def infer_model(mname, chunks_np):\n",
        "    '''Forward one model on chunks, return blend_logits (12, 234).'''\n",
        "    cfg = MODELS_CFG[mname]\n",
        "    mel_cfg = MEL_CONFIGS[cfg['mel']]\n",
        "    mel_tf = MEL_TRANSFORMS[cfg['mel']]\n",
        "    model = MODELS[mname]\n",
        "    fmt = MODEL_FORMAT[mname]\n",
        "\n",
        "    # To tensor + mel\n",
        "    wav_t = torch.from_numpy(chunks_np).unsqueeze(1)   # (12, 1, samples)\n",
        "    mel = mel_tf(wav_t)\n",
        "    mel = normalize_mel(mel)\n",
        "\n",
        "    if fmt == 'ov':\n",
        "        # OV inference\n",
        "        mel_np = mel.numpy().astype(np.float32)\n",
        "        result = model.create_infer_request().infer({'mel': mel_np})\n",
        "        # Outputs are tagged by name\n",
        "        clip_logits = None\n",
        "        framewise = None\n",
        "        for k, v in result.items():\n",
        "            name = k.any_name if hasattr(k, 'any_name') else str(k)\n",
        "            if 'clip' in name:\n",
        "                clip_logits = v\n",
        "            elif 'frame' in name:\n",
        "                framewise = v\n",
        "        if clip_logits is None or framewise is None:\n",
        "            # Fallback: by index (clip_logits first, framewise second)\n",
        "            outs = list(result.values())\n",
        "            clip_logits = outs[0]\n",
        "            framewise = outs[1]\n",
        "        # framewise shape: (B, time, num_class) — max over time = axis 1\n",
        "        frame_max = framewise.max(axis=1)\n",
        "        blend_logits = 0.5 * clip_logits + 0.5 * frame_max\n",
        "    else:\n",
        "        # PyTorch inference\n",
        "        with torch.no_grad():\n",
        "            clip_logits, framewise = model(mel, return_framewise=True)\n",
        "            frame_max = framewise.max(dim=1).values\n",
        "            blend_logits = 0.5 * clip_logits + 0.5 * frame_max\n",
        "            blend_logits = blend_logits.float().cpu().numpy()\n",
        "\n",
        "    return blend_logits   # (12, 234)\n",
        "\n",
        "\n",
        "def sigmoid_np(x):\n",
        "    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))).astype(np.float32)\n",
        "\n",
        "print('OK inference helpers ready')\n",
    ]
})

# ---- Cell 7: Main inference loop ----
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 7: Main inference loop with speed measurement\n",
        "# ============================================================\n",
        "test_files = sorted(glob.glob(str(TEST_DIR / '*.ogg'))) if TEST_DIR.is_dir() else []\n",
        "if len(test_files) == 0:\n",
        "    fallback = BASE / 'train_soundscapes'\n",
        "    if fallback.is_dir():\n",
        "        test_files = sorted(glob.glob(str(fallback / '*.ogg')))[:5]\n",
        "        print(f'No test_soundscapes, using {len(test_files)} train files for debug')\n",
        "print(f'Test files: {len(test_files)}')\n",
        "\n",
        "# Per-model timing tracker\n",
        "model_times = {mname: 0.0 for mname in MODELS_CFG}\n",
        "\n",
        "all_rows = []\n",
        "all_blended_logits = []\n",
        "t_inference_start = time.time()\n",
        "\n",
        "for fi, fp in enumerate(test_files):\n",
        "    fp = Path(fp)\n",
        "    audio = load_60s(fp)\n",
        "\n",
        "    # Per-model inference, blend in logit space\n",
        "    blended_logits = np.zeros((N_OUT_ROWS, N_CLASSES), dtype=np.float32)\n",
        "    for mname, cfg in MODELS_CFG.items():\n",
        "        t0 = time.time()\n",
        "        if cfg['window_type'] == 'simple':\n",
        "            chunks = prepare_input_simple(audio, cfg['duration'])\n",
        "        else:\n",
        "            chunks = prepare_input_sliding(audio, cfg['duration'])\n",
        "        logits = infer_model(mname, chunks)\n",
        "        blended_logits += cfg['weight'] * logits\n",
        "        model_times[mname] += time.time() - t0\n",
        "\n",
        "    # Save per-file row_ids + logits\n",
        "    stem = fp.stem\n",
        "    for k in range(N_OUT_ROWS):\n",
        "        all_rows.append(f'{stem}_{(k+1)*5}')\n",
        "    all_blended_logits.append(blended_logits)\n",
        "\n",
        "    # Progress\n",
        "    if (fi + 1) % 50 == 0 or fi == 0 or fi == len(test_files) - 1:\n",
        "        elapsed = time.time() - t_inference_start\n",
        "        rate = (fi + 1) / max(elapsed, 1e-6)\n",
        "        eta_min = (len(test_files) - fi - 1) / max(rate, 1e-6) / 60\n",
        "        print(f'  [{fi+1:4d}/{len(test_files)}] {elapsed:.0f}s {rate:.2f}f/s eta={eta_min:.1f}min')\n",
        "\n",
        "# Concatenate logits across files\n",
        "logits_arr = np.concatenate(all_blended_logits, axis=0).astype(np.float32)   # (n_files*12, 234)\n",
        "print(f'\\nInference DONE: {len(all_rows)} rows in {(time.time()-t_inference_start)/60:.1f} min')\n",
        "print(f'\\nPer-model timing (total seconds):')\n",
        "for mname, total in sorted(model_times.items(), key=lambda x: -x[1]):\n",
        "    pct = 100 * total / sum(model_times.values())\n",
        "    print(f'  {mname:15s} {total:6.1f}s ({pct:5.1f}%)')\n",
    ]
})

# ---- Cell 8: Smooth + sigmoid + submission ----
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 8: Logit smoothing + sigmoid + submission CSV\n",
        "# ============================================================\n",
        "# Smooth in logit space across 12 windows per file\n",
        "def smooth_per_file(arr, kernel=GAUSSIAN_KERNEL):\n",
        "    smoothed = arr.reshape(-1, N_OUT_ROWS, arr.shape[1]).copy()\n",
        "    for i in range(smoothed.shape[0]):\n",
        "        smoothed[i] = convolve1d(smoothed[i], kernel, axis=0, mode='nearest')\n",
        "    return smoothed.reshape(-1, arr.shape[1])\n",
        "\n",
        "logits_smoothed = smooth_per_file(logits_arr)\n",
        "probs = sigmoid_np(logits_smoothed)\n",
        "\n",
        "# Build submission CSV\n",
        "sub = pd.DataFrame(probs, columns=PRIMARY_LABELS)\n",
        "sub.insert(0, 'row_id', all_rows)\n",
        "assert sub['row_id'].nunique() == len(sub), 'duplicate row_id'\n",
        "print(f'sub_df: {sub.shape}, mean={sub[PRIMARY_LABELS].mean().mean():.4f}, max={sub[PRIMARY_LABELS].max().max():.4f}')\n",
        "\n",
        "# Reindex to match sample_sub order\n",
        "if len(test_files) > 0 and len(SAMPLE_SUB) > 0:\n",
        "    sub = sub.set_index('row_id').reindex(SAMPLE_SUB['row_id']).reset_index()\n",
        "    sub[PRIMARY_LABELS] = sub[PRIMARY_LABELS].fillna(0.0)\n",
        "\n",
        "sub_path = OUT_DIR / 'submission.csv'\n",
        "sub.to_csv(sub_path, index=False)\n",
        "print(f'\\nSaved: {sub_path} ({sub_path.stat().st_size/1e6:.1f}MB)')\n",
        "print(sub.head(3))\n",
        "\n",
        "total_time = time.time() - GLOBAL_START\n",
        "print(f'\\n=== TOTAL TIME: {total_time/60:.1f} min ===')\n",
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

nb_path = OUT_DIR / "nb_infer_ensemble6.ipynb"
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"OK Built {nb_path} ({len(cells)} cells)")

# Build push script
push_src = '''"""Push 6-model ensemble inference NB (CPU sub).

Note: openvino is pre-installed on Kaggle competition env? May need pip install at runtime.
If internet=off + no preinstall, attach openvino wheel dataset.

Currently uses Kaggle\\'s pre-installed openvino (try first; if fails, attach wheel dataset).
"""
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

NB = Path(__file__).with_name("nb_infer_ensemble6.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-ensemble6-infer"
TITLE = "birdclef2026 ensemble6 infer"

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
        "enable_internet": False,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            f"{USER}/birdclef2026-exp015-weights",
            f"{USER}/birdclef2026-exp017-weights",
            f"{USER}/birdclef2026-exp081-weights",
            f"{USER}/birdclef2026-exp082-weights",
            f"{USER}/birdclef2026-exp083-weights",
            f"{USER}/birdclef2026-exp084-weights",
        ],
        "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None))
    print("Version:", getattr(r, "version_number", None))
'''

push_path = OUT_DIR / "_push_nb_infer_ensemble6.py"
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
