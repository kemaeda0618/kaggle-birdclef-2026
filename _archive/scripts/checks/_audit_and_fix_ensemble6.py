"""Thorough fix for ensemble6 NB:
1. Cell 5 (load): defensive print with try/except (no any_name on outputs without names)
2. Cell 6 (infer): use input/output PORTS directly (not names), identify outputs by SHAPE

This is robust to:
- OV outputs without names (from direct PyTorch convert)
- OV inputs with dynamic shapes
- Mixed model formats (OV + PyTorch)
"""
import json
from pathlib import Path

P = Path("experiment/ensemble6/notebook/nb_infer_ensemble6.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

# ============================================================
# Cell 5: Defensive print (skip if no name)
# ============================================================
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    if "core = ov.Core()" in src and "Load all 6 models" in src:
        new_src = '''# ============================================================
# Cell 5: Load all 6 models (OV + PyTorch) — defensive print
# ============================================================
core = ov.Core()
MODELS = {}        # name → loaded model object (compiled OV or PyTorch nn.Module)
MODEL_FORMAT = {}
TIME_FRAMES = {}
OV_INPUTS = {}     # name → input port (for OV models)

def safe_print_io(compiled):
    """Print I/O info without crashing on missing names or dynamic shapes."""
    try:
        inputs = list(compiled.inputs)
        if inputs:
            try: name = inputs[0].any_name
            except: name = '?'
            try: shape = str(inputs[0].partial_shape)
            except: shape = '?'
            print(f'  input: {name} shape={shape}')
    except Exception as e:
        print(f'  (input info unavailable: {type(e).__name__})')
    try:
        outputs = list(compiled.outputs)
        for o in outputs:
            try: name = o.any_name
            except: name = '?'
            try: shape = str(o.partial_shape)
            except: shape = '?'
            print(f'  output: {name} shape={shape}')
    except Exception as e:
        print(f'  (output info unavailable: {type(e).__name__})')


for mname, cfg in MODELS_CFG.items():
    print(f'\\n--- Loading {mname} ---')
    DATASET_ROOTS = [
        Path('/kaggle/input/datasets/maekeso') / cfg['dataset_slug'],
        Path('/kaggle/input') / cfg['dataset_slug'],
    ]
    ds_root = None
    for p in DATASET_ROOTS:
        if p.exists():
            ds_root = p; break
    assert ds_root is not None, f'{cfg["dataset_slug"]} not mounted'

    mel_cfg = MEL_CONFIGS[cfg['mel']]
    duration = cfg['duration']
    n_samples = SR * duration
    time_frames = n_samples // mel_cfg['hop'] + 1
    TIME_FRAMES[mname] = time_frames
    print(f'  mel: {cfg["mel"]} ({mel_cfg["n_mels"]}, {time_frames})')

    if cfg['format'] == 'ov':
        ov_path = ds_root / cfg['ov_path']
        assert ov_path.exists(), f'OV not found: {ov_path}'
        ov_model = core.read_model(str(ov_path))
        compiled = core.compile_model(ov_model, 'CPU')
        MODELS[mname] = compiled
        MODEL_FORMAT[mname] = 'ov'
        # Use input PORT directly (bypass name lookup)
        OV_INPUTS[mname] = compiled.input(0)
        print(f'  format: OV')
        safe_print_io(compiled)
    else:
        # PyTorch ckpt
        ckpt_path = ds_root / cfg['pytorch_ckpt']
        if not ckpt_path.exists():
            for f in ds_root.rglob(Path(cfg['pytorch_ckpt']).name):
                ckpt_path = f; break
        assert ckpt_path.exists(), f'ckpt not found in {ds_root}'
        state = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        model = BirdSEDModel(
            backbone_name=cfg['backbone'],
            n_mels=mel_cfg['n_mels'],
            time_frames=time_frames,
        ).eval()
        msg = model.load_state_dict(state['model_state'], strict=False)
        MODELS[mname] = model
        MODEL_FORMAT[mname] = 'pytorch'
        print(f'  format: PyTorch CPU')
        print(f'  ckpt: {ckpt_path.name}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M')
        print(f'  load: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}')

print(f'\\n=== Loaded {len(MODELS)} models ===')
'''
        c["source"] = new_src.splitlines(keepends=True)
        print("OK Cell 5 (load): defensive print + input PORT")

# ============================================================
# Cell 6: Robust inference (PORT + SHAPE-based output identification)
# ============================================================
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    if "def infer_model" in src and "Inference helpers" in src:
        new_src = '''# ============================================================
# Cell 6: Inference helpers — robust to OV models without output names
# ============================================================
GAUSSIAN_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
N_OUT_ROWS = 12

def load_60s(path):
    """Load 60s mono 32kHz audio."""
    try:
        wav, sr = sf.read(str(path), dtype='float32', always_2d=False)
    except Exception:
        return np.zeros(SR * 60, dtype=np.float32)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    target = SR * 60
    if len(wav) < target: wav = np.pad(wav, (0, target - len(wav)))
    elif len(wav) > target: wav = wav[:target]
    return wav.astype(np.float32)


def prepare_input_simple(audio_60s, duration):
    """5s simple chunking: (12, duration*SR)."""
    n_samples = SR * duration
    return audio_60s.reshape(N_OUT_ROWS, n_samples)


def prepare_input_sliding(audio_60s, duration):
    """Sliding window for 10s/20s: padded + 12 windows step 5s."""
    padded_sec = (N_OUT_ROWS - 1) * 5 + duration   # 65 (10s), 75 (20s)
    pad_lead = (padded_sec - 60) // 2
    padded_samples = padded_sec * SR
    pad_lead_samples = pad_lead * SR
    audio_padded = np.zeros(padded_samples, dtype=np.float32)
    audio_padded[pad_lead_samples:pad_lead_samples + len(audio_60s)] = audio_60s
    chunks = np.zeros((N_OUT_ROWS, SR * duration), dtype=np.float32)
    step_samples = 5 * SR
    win_samples = duration * SR
    for k in range(N_OUT_ROWS):
        start = k * step_samples
        chunks[k] = audio_padded[start:start + win_samples]
    return chunks


def identify_ov_outputs(result_dict):
    """Identify clip_logits (2D) and framewise (3D) from OV result by SHAPE.
    Robust to missing output names."""
    arrays = list(result_dict.values())
    clip_logits = None
    framewise = None
    for arr in arrays:
        if arr.ndim == 2:
            clip_logits = arr
        elif arr.ndim == 3:
            framewise = arr
    if clip_logits is None or framewise is None:
        raise RuntimeError(f'Could not identify outputs by shape. Got: {[a.shape for a in arrays]}')
    return clip_logits, framewise


def infer_model(mname, chunks_np):
    """Forward one model on chunks, return blend_logits (12, 234)."""
    cfg = MODELS_CFG[mname]
    mel_cfg = MEL_CONFIGS[cfg['mel']]
    mel_tf = MEL_TRANSFORMS[cfg['mel']]
    model = MODELS[mname]
    fmt = MODEL_FORMAT[mname]

    # Mel computation
    wav_t = torch.from_numpy(chunks_np).unsqueeze(1)   # (12, 1, samples)
    mel = mel_tf(wav_t)
    mel = normalize_mel(mel)

    if fmt == 'ov':
        # OV: use input PORT (bypass name lookup)
        mel_np = mel.numpy().astype(np.float32)
        input_port = OV_INPUTS[mname]
        result = model.create_infer_request().infer({input_port: mel_np})
        clip_logits, framewise = identify_ov_outputs(result)
        # framewise shape: (B, time, num_class) — max over time = axis 1
        frame_max = framewise.max(axis=1)
        blend_logits = 0.5 * clip_logits + 0.5 * frame_max
    else:
        # PyTorch
        with torch.no_grad():
            clip_logits, framewise = model(mel, return_framewise=True)
            # framewise shape: (B, time, num_class) after permute in model
            frame_max = framewise.max(dim=1).values
            blend_logits = 0.5 * clip_logits + 0.5 * frame_max
            blend_logits = blend_logits.float().cpu().numpy()

    return blend_logits.astype(np.float32)   # (12, 234)


def sigmoid_np(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


print('OK inference helpers ready (PORT-based + shape identify)')

# ============================================================
# Smoke test: 1 model 1 file で動作確認
# ============================================================
print('\\n=== Smoke test (1 file × 1 model) ===')
if MODELS_CFG:
    test_mname = list(MODELS_CFG.keys())[0]
    cfg = MODELS_CFG[test_mname]
    # Dummy 60s audio
    dummy_audio = np.random.randn(SR * 60).astype(np.float32) * 0.01
    chunks = prepare_input_simple(dummy_audio, cfg['duration']) if cfg['window_type'] == 'simple' else prepare_input_sliding(dummy_audio, cfg['duration'])
    print(f'  Test model: {test_mname} ({cfg["window_type"]}, {cfg["duration"]}s)')
    print(f'  chunks shape: {chunks.shape}')
    try:
        out = infer_model(test_mname, chunks)
        print(f'  output shape: {out.shape}, dtype={out.dtype}')
        print(f'  output stats: mean={out.mean():.4f}, std={out.std():.4f}, max={out.max():.4f}')
        print(f'  [OK] Smoke test passed')
    except Exception as e:
        print(f'  [FAIL] {type(e).__name__}: {str(e)[:300]}')
        raise
'''
        c["source"] = new_src.splitlines(keepends=True)
        print("OK Cell 6 (infer): PORT-based + shape identification + smoke test")

P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# Syntax check
import ast
nb = json.loads(P.read_text(encoding="utf-8"))
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
        print(f"  SyntaxError cell {c.get('id','?')}: {e}")
        err_line = e.lineno or 1
        for i, ln in enumerate(clean.splitlines()[max(0, err_line-2):err_line+2]):
            print(f"    L{max(0, err_line-2)+i+1}: {ln[:150]}")
print(f"\nSyntax: {n_ok} OK, {n_err} ERRORS")
