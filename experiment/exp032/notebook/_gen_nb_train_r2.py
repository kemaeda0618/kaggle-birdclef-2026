"""Generate exp032 R2 train NB (Colab Pro A100, multi-iter NS round 2).

R1 pseudo (train_soundscapes 推論) を teacher として student fine-tune。
R1 と同じ Babych init + no Perch、SHARES に pseudo_sc 追加。

Changes from exp017 R2 template:
- BACKBONE: eca_nfnet_l0 -> tf_efficientnet_b3.ns_jft_in1k
- USE_PERCH_DISTILL: True -> False (Perch 完全除去、Antoine 流維持)
- R1 pseudo source: exp017/r1-pseudo -> exp032/r1-pseudo
- R2 output: exp017/r2 -> exp032/r2
- Kaggle weights dataset: birdclef2026-exp017-weights -> birdclef2026-exp032-weights
- Babych weight init (R1 と同じ load logic) を model 構築時に追加
- Title / SLUG 名前変更

期待結果:
- R1 val 0.9156 → R2 val 0.93-0.95 (+0.015-0.02 lift、Babych iter1→iter2 +0.020 と類似)
- R2 LB 0.90-0.93 (gap -0.020 想定)
"""
import re as _re
from pathlib import Path

HERE = Path(__file__).parent
EXP017_GEN_PATH = HERE.parent.parent / "exp017" / "notebook" / "_gen_nb_train_r2.py"
SRC = EXP017_GEN_PATH.read_text(encoding="utf-8")

# ============================================================================
# Patch 1: BACKBONE
# ============================================================================
SRC = SRC.replace(
    'BACKBONE = "eca_nfnet_l0"',
    'BACKBONE = "tf_efficientnet_b3.ns_jft_in1k"',
)

# ============================================================================
# Patch 2: Disable Perch distill
# ============================================================================
SRC = SRC.replace(
    'USE_PERCH_DISTILL = True',
    'USE_PERCH_DISTILL = False  # ★ exp032: Perch 完全除去 (Antoine 流、Cluster 外独自軸)',
)

# ============================================================================
# Patch 3: Drive paths exp017 -> exp032
# ============================================================================
SRC = SRC.replace(
    '"output" / "exp017"',
    '"output" / "exp032"',
)
SRC = SRC.replace(
    "birdclef2026-exp017-weights",
    "birdclef2026-exp032-weights",
)
SRC = SRC.replace(
    "exp017 R2",
    "exp032 R2",
)

# ============================================================================
# Patch 4: Babych weight init (insert after timm.create_model in model class)
# ============================================================================
BABYCH_INIT_CODE = '''
        # ★ exp032 R2: Babych BC25 1位 b3 weight を init として上書き
        # R1 と同じ Babych iter3 weight を再使用、R1 pseudo で multi-iter NS
        BABYCH_WEIGHT_PATH = None
        _babych_candidates = [
            LOCAL_DATA / "babych_ensemble",
            Path("/kaggle/input/datasets/nikitababich/birdclef2025-1st-place-ensemble"),
            Path("/kaggle/input/birdclef2025-1st-place-ensemble"),
            Path("/content/drive/MyDrive/kaggle/birdclef2026/external/birdclef2025-1st-place-ensemble"),
        ]
        for _bd in _babych_candidates:
            if _bd.exists():
                _pt_files = sorted(_bd.glob("tf_efficientnet_b3*.pt"))
                if _pt_files:
                    BABYCH_WEIGHT_PATH = _pt_files[0]
                    break
        if BABYCH_WEIGHT_PATH is not None and BABYCH_WEIGHT_PATH.exists():
            print(f"  [exp032 R2] Loading Babych b3 init weight: {BABYCH_WEIGHT_PATH.name}")
            _babych_state = torch.load(str(BABYCH_WEIGHT_PATH), map_location="cpu", weights_only=False)
            if isinstance(_babych_state, dict) and "model" in _babych_state:
                _babych_state = _babych_state["model"]
            elif isinstance(_babych_state, dict) and "state_dict" in _babych_state:
                _babych_state = _babych_state["state_dict"]
            _backbone_state = {}
            _stripped = 0
            for k, v in _babych_state.items():
                _key = k
                for _prefix in ["model.backbone.", "backbone.", "model."]:
                    if _key.startswith(_prefix):
                        _key = _key[len(_prefix):]
                        break
                if any(s in _key for s in ["classifier.", "head.", "fc.", "att_block", "attention",
                                            "logit", "framewise", "global_pool"]):
                    _stripped += 1
                    continue
                _backbone_state[_key] = v
            for _stem_key in ["conv_stem.weight", "conv1.weight", "stem.0.weight", "stem.conv.weight"]:
                if _stem_key in _backbone_state and _backbone_state[_stem_key].shape[1] == 3:
                    _w = _backbone_state[_stem_key]
                    _backbone_state[_stem_key] = _w.mean(dim=1, keepdim=True)
                    print(f"  [exp032 R2] Adapted {_stem_key}: 3-ch -> 1-ch")
            _missing, _unexpected = self.backbone.load_state_dict(_backbone_state, strict=False)
            print(f"  [exp032 R2] Babych b3 loaded: {len(_backbone_state) - _stripped} keys, "
                  f"stripped {_stripped}, missing={len(_missing)}, unexpected={len(_unexpected)}")
        else:
            print("  [exp032 R2] Babych b3 weight NOT found, using timm ImageNet only init")

'''

_pattern = _re.compile(
    r'(self\.backbone = timm\.create_model\(\s*\n[^)]+\)\s*\n)',
    _re.MULTILINE,
)
_m = _pattern.search(SRC)
if _m:
    insert_pos = _m.end()
    SRC = SRC[:insert_pos] + BABYCH_INIT_CODE + SRC[insert_pos:]
    print(f"  Inserted Babych init at pos {insert_pos}")
else:
    print(f"  WARNING: timm.create_model pattern not found, skipping Babych init insertion")

# ============================================================================
# Patch 5: Insert Babych weight DL cell (similar to R1)
# We add it after the R1 pseudo copy cell, before model creation
# ============================================================================
BABYCH_DL_CELL = '''cells.append(code_cell(r"""# ============================================================
# Cell ★: exp032 R2: Download Babych BC25 b3 weight (R1 と同じ init を再使用)
# ============================================================
import os, time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

BABYCH_DIR = LOCAL_DATA / "babych_ensemble"
BABYCH_DIR.mkdir(parents=True, exist_ok=True)

existing = list(BABYCH_DIR.glob("tf_efficientnet_b3*.pt"))
if existing:
    print(f"[exp032 R2] Babych b3 weight already exists: {existing[0].name}")
else:
    print("[exp032 R2] Downloading nikitababich/birdclef2025-1st-place-ensemble (b3 only)...")
    api = KaggleApi(); api.authenticate()
    files = api.dataset_list_files("nikitababich/birdclef2025-1st-place-ensemble").files
    b3_files = [f.name for f in files if f.name.startswith("tf_efficientnet_b3") and f.name.endswith(".pt")]
    assert b3_files, "tf_efficientnet_b3 *.pt not found in babych dataset"
    target = b3_files[0]
    api.dataset_download_file("nikitababich/birdclef2025-1st-place-ensemble",
                               file_name=target, path=str(BABYCH_DIR),
                               force=True, quiet=False)
    zip_file = BABYCH_DIR / (target + ".zip")
    if zip_file.exists():
        import zipfile
        with zipfile.ZipFile(zip_file) as zf:
            zf.extractall(BABYCH_DIR)
        zip_file.unlink()
    found = list(BABYCH_DIR.glob("tf_efficientnet_b3*.pt"))
    assert found, f"Download failed, no .pt in {BABYCH_DIR}"
    print(f"  -> {found[0].name} ({found[0].stat().st_size/1e6:.1f} MB)")
""", "babych-dl-r2"))

'''

# Insert AFTER R1 pseudo copy cell (Cell 3 area), BEFORE Cell 4 (config + import)
# Pattern: find "cells.append(code_cell(r\"\"\"# ============================================================\n# Cell 4: Imports + Config"
# Or find the config cell header
_target = 'cells.append(code_cell(r"""# ============================================================\n# Cell 4: Imports + Config'
if _target in SRC:
    SRC = SRC.replace(_target, BABYCH_DL_CELL + _target, 1)
    print("  Inserted Babych DL cell before Cell 4 (imports)")
else:
    # Try other targets
    for marker in [
        'cells.append(code_cell(r"""# ============================================================\n# Cell 3: Imports',
        'BACKBONE = "tf_efficientnet_b3.ns_jft_in1k"',
    ]:
        if marker in SRC:
            SRC = SRC.replace(marker, BABYCH_DL_CELL + marker, 1)
            print(f"  Inserted Babych DL cell before {marker[:60]!r}")
            break
    else:
        print("  WARNING: Could not find insertion point for Babych DL cell")

# ============================================================================
# Patch 6: Header / Title
# ============================================================================
SRC = SRC.replace(
    "# exp017",
    "# exp032",
)
SRC = SRC.replace(
    "exp017 — eca_nfnet_l0",
    "exp032 — tf_efficientnet_b3 + Babych init (no Perch)",
)

# Cleanup leftover exp017 references in markdown + comments + strings
_cleanups = [
    ("exp017/r1-pseudo", "exp032/r1-pseudo"),
    ("exp017/r2-pseudo", "exp032/r2-pseudo"),
    ("exp017/r2/", "exp032/r2/"),
    ("output/exp017", "output/exp032"),
    ("exp017 R3 等で使う想定", "exp032 R3 等で使う想定"),
    ('TITLE = "birdclef2026 exp017 weights"', 'TITLE = "birdclef2026 exp032 weights"'),
    ("exp017 R1 (val ", "exp032 R1 (val "),
    ("exp017 R2 (", "exp032 R2 ("),
    ("exp017 R2、", "exp032 R2、"),
    ("(eca_nfnet_l0)", "(tf_efficientnet_b3 + Babych init, no Perch)"),
]
for old, new in _cleanups:
    if old in SRC:
        SRC = SRC.replace(old, new)
        print(f"  cleaned: {old[:60]}")

# Patch 7: 削除 — mel params は exp017 R2 base = BC2026 標準 (2048/512/256/20) 維持
# R1 v2 mel-fix (N_FFT=4096, HOP=312, N_MELS=224) は LB 0.786 で失敗確認済
# R2 では R1 v1 (LB 0.876) と同じ mel = BC2026 標準を使う
print("[exp032 R2] mel params: BC2026 標準 (2048/512/256/20)、Babych mel-fix は適用しない")

# NOTE: Patch 8 (r1-v1-force-dl cell 挿入) は撤回 — R1 ckpt warm-start ではなく
# R1 pseudo CSV を生成して R2 訓練に使う

# Patch 9: R2 学習開始前に R1 pseudo を自動再生成 (Kaggle Dataset v1 から R1 v1 ckpt DL)
# - setup cell の R1 pseudo assert を条件付きに変更 (再生成するので未存在 OK)
# - config cell の後、load_data cell の前に新 cell 挿入:
#     v1 ckpt DL → inline model load → train_soundscapes inference → pseudo CSV 上書き
SRC = SRC.replace(
    'assert DRIVE_R1_PSEUDO.exists(), f"R1 pseudo CSV missing: {DRIVE_R1_PSEUDO}"',
    '# R1 pseudo CSV はこの後の regen-r1-pseudo cell で自動生成 (R1 v1 ckpt から)\n'
    'if not DRIVE_R1_PSEUDO.exists():\n'
    '    print(f"[INFO] R1 pseudo missing, will regenerate from Kaggle Dataset v1 in regen-r1-pseudo cell")',
)

REGEN_R1_PSEUDO_CELL = '''cells.append(code_cell(r"""# ============================================================
# Cell ★: R1 pseudo 自動再生成 (Kaggle Dataset v1 から R1 v1 ckpt DL → inference)
# 目的: Drive 上の R1 pseudo は R1 v2 (bad、LB 0.786 ckpt 由来) で上書きされてる
#       可能性あるため、強制的に v1 (LB 0.876 good ckpt) で再生成
# Set REGENERATE_R1_PSEUDO=False to skip (既存 Drive CSV 信頼するなら)
# ============================================================
REGENERATE_R1_PSEUDO = True

if REGENERATE_R1_PSEUDO:
    import zipfile
    import torch
    import torch.nn as nn
    import torchaudio
    import timm
    import glob
    from scipy.ndimage import gaussian_filter1d
    # ★ NOTE: use aliased name to avoid shadowing config cell's `from torch.cuda.amp import autocast` (old API)
    from torch.amp import autocast as _regen_autocast

    # --- Download R1 v1 ckpt (version_number=1 pin) ---
    R1_V1_DIR = LOCAL_DATA / "exp032_r1_v1"
    R1_V1_DIR.mkdir(parents=True, exist_ok=True)
    R1_V1_CKPT = R1_V1_DIR / "ckpt_best_ns22.pth"

    # cleanup stale files (incomplete prior DL etc)
    for _stale in R1_V1_DIR.glob("*"):
        if _stale.name != "ckpt_best_ns22.pth":
            try: _stale.unlink()
            except (OSError, IsADirectoryError): pass

    if not R1_V1_CKPT.exists() or R1_V1_CKPT.stat().st_size < 100_000_000:
        # ★ fast path: single-file URL + v1 pin (141 MB only, not 425 MB ZIP)
        print("[regen] Fast DL: ckpt_best_ns22.pth (141 MB) only, v1 pin")
        import requests
        _key = json.loads((KAGGLE_CFG / 'kaggle.json').read_text())["key"]
        _h = {"Authorization": f"Bearer {_key}"} if _key.startswith("KGAT_") else {}
        _auth = None if _key.startswith("KGAT_") else ("maekeso", _key)
        _url = "https://www.kaggle.com/api/v1/datasets/download/maekeso/birdclef2026-exp032-weights/ckpt_best_ns22.pth?datasetVersionNumber=1"
        with requests.get(_url, headers=_h, auth=_auth, stream=True, timeout=600) as r:
            r.raise_for_status()
            _total = int(r.headers.get("Content-Length", 0))
            print(f"  CL={_total/1e6:.1f} MB")
            with open(R1_V1_CKPT, "wb") as f:
                _done = 0
                for chunk in r.iter_content(chunk_size=4*1024*1024):  # 4 MB chunks
                    f.write(chunk)
                    _done += len(chunk)
                    if _total and _done % (32*1024*1024) < 4*1024*1024:
                        print(f"  ... {_done/1e6:.0f}/{_total/1e6:.0f} MB ({100*_done/_total:.0f}%)")
        print(f"  DONE: {R1_V1_CKPT.stat().st_size/1e6:.1f} MB")
        assert R1_V1_CKPT.exists() and R1_V1_CKPT.stat().st_size > 100_000_000, f"v1 DL failed: {R1_V1_CKPT}"

    # Verify v1 ckpt
    _v1_state = torch.load(str(R1_V1_CKPT), map_location="cpu", weights_only=False)
    _val_ns22 = _v1_state.get("best_ns22", float("nan"))
    print(f"[regen] v1 ckpt: epoch={_v1_state.get('epoch')}, val_ns22={_val_ns22:.4f}")
    if not (0.910 < _val_ns22 < 0.920):
        print(f"  WARN: val_ns22 unexpected. v1 expected ~0.9156 (LB 0.876), v2 was ~0.9095 (LB 0.786)")

    # --- Inline model class (BirdSEDModel for b3) ---
    class _RegenMelTF(nn.Module):
        def __init__(self):
            super().__init__()
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0,
            )
            self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        def forward(self, x):
            return self.db(self.mel_spec(x))

    class _RegenGeM(nn.Module):
        def __init__(self, p_init=3.0, eps=1e-6):
            super().__init__()
            self.p = nn.Parameter(torch.tensor(float(p_init)))
            self.eps = eps
        def forward(self, x):
            p = self.p.clamp(min=1.0)
            x = x.clamp(min=self.eps).pow(p)
            return x.mean(dim=2).pow(1.0 / p)

    class _RegenSED(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                BACKBONE, pretrained=False, in_chans=1,
                num_classes=0, global_pool="", drop_path_rate=0.1,
            )
            with torch.no_grad():
                _dummy = torch.randn(1, 1, N_MELS, TRAIN_SAMPLES // HOP_LENGTH + 1)
                self.backbone_dim = self.backbone(_dummy).shape[1]
            self.gem_freq = _RegenGeM(3.0)
            self.dense = nn.Sequential(
                nn.Dropout(0.25),
                nn.Linear(self.backbone_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            )
            self.att = nn.Conv1d(512, NUM_CLASSES, 1, bias=True)
            self.cla = nn.Conv1d(512, NUM_CLASSES, 1, bias=True)
        def forward(self, x):
            h = self.backbone(x)
            h = self.gem_freq(h).permute(0, 2, 1)
            h = self.dense(h).permute(0, 2, 1)
            n_att = torch.softmax(torch.tanh(self.att(h)), dim=-1)
            frame = self.cla(h)
            clip = (n_att * frame).sum(dim=2)
            return clip, frame.permute(0, 2, 1)

    _regen_model = _RegenSED().to(device)
    _msg = _regen_model.load_state_dict(_v1_state["model_state"], strict=False)
    print(f"[regen] v1 ckpt loaded: missing={len(_msg.missing_keys)}, unexpected={len(_msg.unexpected_keys)}")
    _regen_model.eval()
    _regen_model = _regen_model.to(memory_format=torch.channels_last)
    _regen_mel_tf = _RegenMelTF().to(device)

    # --- Inference on train_soundscapes ---
    import soundfile as sf
    import librosa as _librosa
    from tqdm.auto import tqdm

    _CHUNK_N = TRAIN_SAMPLES
    _N_WIN = 12

    def _load_60s(path):
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SR:
            wav = _librosa.resample(wav, orig_sr=sr, target_sr=SR)
        target = _N_WIN * _CHUNK_N
        if len(wav) < target: wav = np.pad(wav, (0, target - len(wav)))
        elif len(wav) > target: wav = wav[:target]
        return wav.astype(np.float32).reshape(_N_WIN, _CHUNK_N)

    sc_files = sorted(glob.glob(str(TS_DIR / "*.ogg")))
    print(f"[regen] Inference on {len(sc_files)} train_soundscapes files...")
    _all_probs, _all_fnames, _all_starts, _all_ends = [], [], [], []
    _t0 = time.time()

    with torch.no_grad():
        for fp in tqdm(sc_files, desc="regen-inf", mininterval=2.0):
            stem = Path(fp).stem
            try:
                _chunks = _load_60s(fp)
            except Exception as e:
                print(f"WARN {stem}: {e}"); _chunks = np.zeros((_N_WIN, _CHUNK_N), dtype=np.float32)
            _wav_t = torch.from_numpy(_chunks).unsqueeze(1).to(device)
            _mel = _regen_mel_tf(_wav_t)
            for i in range(_mel.size(0)):
                _mel[i] = (_mel[i] - _mel[i].mean()) / (_mel[i].std() + 1e-6)
            _mel = _mel.to(memory_format=torch.channels_last)
            with _regen_autocast("cuda"):
                _clip, _frame = _regen_model(_mel)
                _frame_max = _frame.max(dim=1).values
                _p_clip = torch.sigmoid(_clip).float().cpu().numpy()
                _p_fmax = torch.sigmoid(_frame_max).float().cpu().numpy()
            _p = 0.5 * _p_clip + 0.5 * _p_fmax
            _p = gaussian_filter1d(_p, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
            _all_probs.append(_p)
            for wi in range(_N_WIN):
                _all_fnames.append(stem)
                _all_starts.append(wi * TRAIN_DURATION)
                _all_ends.append((wi + 1) * TRAIN_DURATION)

    _prob_mat = np.concatenate(_all_probs, axis=0).astype(np.float32)
    print(f"[regen] Pre-PT: mean={_prob_mat.mean():.6f}, 99%ile={np.percentile(_prob_mat, 99):.4f}")
    _prob_mat = np.power(_prob_mat, 1.2).astype(np.float32)  # POWER_GAMMA
    print(f"[regen] Post-PT (γ=1.2): mean={_prob_mat.mean():.6f}, 99%ile={np.percentile(_prob_mat, 99):.4f}")

    # --- Save pseudo CSV (Drive overwrite + backup) ---
    import pandas as pd
    _sample_sub_path = LOCAL_DATA / "sample_submission.csv"
    _sub = pd.read_csv(_sample_sub_path)
    _PRIMARY_LABELS = _sub.columns[1:].tolist()
    _df = pd.DataFrame(_prob_mat, columns=_PRIMARY_LABELS)
    _df.insert(0, "filename", np.array(_all_fnames))
    _df.insert(1, "start_sec", np.array(_all_starts, dtype=np.float32))
    _df.insert(2, "end_sec", np.array(_all_ends, dtype=np.float32))

    if DRIVE_R1_PSEUDO.exists():
        _bak = DRIVE_R1_PSEUDO.with_suffix(".backup_before_v1_regen.csv")
        shutil.copy(DRIVE_R1_PSEUDO, _bak)
        print(f"[regen] backed up existing R1 pseudo to: {_bak.name}")
    DRIVE_R1_PSEUDO.parent.mkdir(parents=True, exist_ok=True)
    _df.to_csv(DRIVE_R1_PSEUDO, index=False)
    print(f"[regen] OVERWROTE Drive R1 pseudo: {DRIVE_R1_PSEUDO} ({DRIVE_R1_PSEUDO.stat().st_size/1e6:.1f} MB)")
    print(f"[regen] Total time: {(time.time()-_t0)/60:.1f} min")

    # --- Free memory ---
    del _regen_model, _regen_mel_tf, _prob_mat, _all_probs, _df
    gc.collect()
    torch.cuda.empty_cache()
    print("[regen] Memory freed, ready for R2 training")
else:
    print("[regen] REGENERATE_R1_PSEUDO=False, using existing Drive R1 pseudo")
    assert DRIVE_R1_PSEUDO.exists(), f"R1 pseudo missing and regen disabled: {DRIVE_R1_PSEUDO}"
""", "regen-r1-pseudo"))

'''

# Insert just after config cell, before load_data cell
# exp017 R2 base には "# Cell 4: Load data" として load_data cell が定義されてる
_load_data_marker = 'cells.append(code_cell(r"""# ============================================================\n# Cell 4: Load data'

if _load_data_marker in SRC:
    SRC = SRC.replace(_load_data_marker, REGEN_R1_PSEUDO_CELL + _load_data_marker, 1)
    print(f"  Inserted regen-r1-pseudo cell before load_data")
else:
    print(f"  WARNING: load_data cell marker not found, regen cell NOT inserted")

# Exec the patched gen
exec(SRC, {"__file__": str(HERE / "_gen_nb_train_r2_inner.py")})
