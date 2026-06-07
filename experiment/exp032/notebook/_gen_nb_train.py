"""Generate exp032 NB: tf_efficientnet_b3 + Babych XC pretrain (no Perch) on BC2026 (Colab Pro A100).

Path A1 (discussion 700763 由来):
- Antoine Masq 公開: EfficientNetV2-b0 + XC pretrain → 単 LB 0.936
- 我々の shortcut: **Babych BC25 1位 weight (nikitababich/birdclef2025-1st-place-ensemble)** を init として流用
- Backbone: tf_efficientnet_b3.ns_jft_in1k (Babych iter3 weight 既に XC + BC25 学習済)
- **Perch distill 一切なし** (Antoine 流の純 CNN path)
- BC2026 train_audio で R1 fine-tune (15-20 ep)、blend slot 4th 候補

Architecture changes vs exp017:
- BACKBONE: eca_nfnet_l0 → tf_efficientnet_b3.ns_jft_in1k
- USE_PERCH_DISTILL: True → False (Perch 完全除去)
- pretrained init: timm default → Babych b3 iter3 weight 上書き

Expected: 単 LB 0.91-0.94 (R1)、R2 で +0.01-0.02 lift → 0.92-0.96 (gold 射程)
"""
import re
from pathlib import Path

HERE = Path(__file__).parent
EXP017_GEN_PATH = HERE.parent.parent / "exp017" / "notebook" / "_gen_nb_train.py"
SRC = EXP017_GEN_PATH.read_text(encoding="utf-8")

# ============================================================================
# Patch 1: BACKBONE
# ============================================================================
SRC = SRC.replace(
    'BACKBONE = "eca_nfnet_l0"',
    'BACKBONE = "tf_efficientnet_b3.ns_jft_in1k"',
)

# ============================================================================
# Patch 2: Disable Perch distill (Antoine 流)
# ============================================================================
SRC = SRC.replace(
    'USE_PERCH_DISTILL = True',
    'USE_PERCH_DISTILL = False  # ★ exp032: Perch 完全除去 (Antoine 流、Cluster 外独自軸)',
)

# ============================================================================
# Patch 3: Babych weight load cell (insert after model creation)
# Find the "self.backbone = timm.create_model(...)" pattern and add weight override
# ============================================================================
BABYCH_INIT_CODE = '''
        # ★ exp032: Babych BC25 1位 weight (iter3) を init として上書き
        # nikitababich/birdclef2025-1st-place-ensemble の b3 weight を流用
        # ImageNet pretrain + Babych XC + BC25 multi-iter NS 済の "脳"
        BABYCH_WEIGHT_PATH = None
        _babych_candidates = [
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
            print(f"  [exp032] Loading Babych b3 init weight: {BABYCH_WEIGHT_PATH.name}")
            _babych_state = torch.load(str(BABYCH_WEIGHT_PATH), map_location="cpu", weights_only=False)
            if isinstance(_babych_state, dict) and "model" in _babych_state:
                _babych_state = _babych_state["model"]
            elif isinstance(_babych_state, dict) and "state_dict" in _babych_state:
                _babych_state = _babych_state["state_dict"]
            # filter to backbone weights only (drop classifier head and SED-specific layers)
            _backbone_state = {}
            _stripped = 0
            for k, v in _babych_state.items():
                # Babych model may have "backbone." or "model.backbone." prefix
                _key = k
                for _prefix in ["model.backbone.", "backbone.", "model."]:
                    if _key.startswith(_prefix):
                        _key = _key[len(_prefix):]
                        break
                # skip classifier / head / sed-specific
                if any(s in _key for s in ["classifier.", "head.", "fc.", "att_block", "attention",
                                            "logit", "framewise", "global_pool"]):
                    _stripped += 1
                    continue
                _backbone_state[_key] = v
            # adapt first conv: Babych used 3-channel input, we use 1-channel
            for _stem_key in ["conv_stem.weight", "conv1.weight", "stem.0.weight", "stem.conv.weight"]:
                if _stem_key in _backbone_state and _backbone_state[_stem_key].shape[1] == 3:
                    _w = _backbone_state[_stem_key]
                    _backbone_state[_stem_key] = _w.mean(dim=1, keepdim=True)
                    print(f"  [exp032] Adapted {_stem_key}: 3-ch -> 1-ch ({_w.shape} -> {_backbone_state[_stem_key].shape})")
            # load with strict=False (will skip unmatched layers)
            _missing, _unexpected = self.backbone.load_state_dict(_backbone_state, strict=False)
            print(f"  [exp032] Babych b3 loaded: {len(_backbone_state) - _stripped} keys, "
                  f"stripped {_stripped}, missing={len(_missing)}, unexpected={len(_unexpected)}")
        else:
            print("  [exp032] Babych b3 weight NOT found, using timm ImageNet only init")

'''

# Insert AFTER the backbone creation, before any layer-aware code
# Pattern: locate "self.backbone = timm.create_model(...)" followed by close-paren on next line
_pattern = re.compile(
    r'(self\.backbone = timm\.create_model\(\s*\n[^)]+\)\s*\n)',
    re.MULTILINE,
)
_m = _pattern.search(SRC)
if _m:
    insert_pos = _m.end()
    SRC = SRC[:insert_pos] + BABYCH_INIT_CODE + SRC[insert_pos:]
    print(f"  Inserted Babych init at pos {insert_pos}")
else:
    print(f"  WARNING: timm.create_model pattern not found, skipping Babych init insertion")

# ============================================================================
# Patch 4: Update header text — full rewrite of exp017 leftover content
# ============================================================================

EXP017_HEADER_PATTERN = r'cells\.append\(md_cell\(r"""# exp017 .*?""", "hdr"\)\)'

NEW_HEADER = r'''cells.append(md_cell(r"""# exp032 — tf_efficientnet_b3 + Babych BC25 weight init (Perch なし) SED training (Colab Pro A100)

**Goal:** discussion 700763 Antoine path に近い独自軸を確保。Babych BC25 1位 weight (`tf_efficientnet_b3.ns_jft_in1k` iter3) を backbone init として流用、**Perch 完全除去**、BC2026 train_audio で fine-tune。
想定: Colab Pro Blackwell で 1 session 完走 (~45-60 min)、そのまま pseudo CSV + Kaggle Dataset upload まで完結。

## なぜ tf_efficientnet_b3?
- Babych BC25 1位 main backbone、`.ns_jft_in1k` (Noisy Student + JFT-300M + ImageNet) pretrain
- 公開 weight (`nikitababich/birdclef2025-1st-place-ensemble`) を init 流用 → XC pretrain 自前 1 week skip
- discussion 700763 で Antoine が EfficientNet 系で単 LB 0.936 (Perch なし) を実証
- 我々の現 stack (Perch ProtoSSM / Tucker HGNetV2 / e17 eca_nfnet_l0) に存在しない backbone family → cluster 0 外への独自軸

## Recipe
- focal 0.85 / labeled_sc 0.15、pseudo 無し (R1 baseline、teacher pseudo は R2 で導入)
- **Perch distill 一切なし** (USE_PERCH_DISTILL=False、Antoine 流の純 CNN path)
- **25 ep** / **batch=192** / AdamW **lr=3e-4** / cosine + warmup **3 ep** / **NUM_WORKERS=8**
- soft mixup (alpha=0.4)、SpecAugment、AUG 0.5 prob

> ⚠️ batch=192 は Blackwell 96GB 前提。A100 40GB なら BATCH=64-80 + LR sqrt 補正。

## NB 構成 (train + pseudo + upload 統合)
1. Setup (Drive mount + kaggle.json)
2. BC2026 competition data DL
2b. ★ Babych b3 weight DL (~80MB) from nikitababich/birdclef2025-1st-place-ensemble
3. Imports + config
4-13. Train phase (focal + sc、25 ep)
14-17. Pseudo phase (best_ns22 ckpt で train_soundscapes 推論 → CSV → Drive)
18-19. Kaggle Dataset upload (weights → maekeso/birdclef2026-exp032-weights)

## Drive 構成
- input: `/content/drive/MyDrive/kaggle/birdclef2026/`
- output: `/content/drive/MyDrive/kaggle/birdclef2026/output/exp032/r1/`
- pseudo: `/content/drive/MyDrive/kaggle/birdclef2026/output/exp032/r1-pseudo/`

## 出力
- `ckpt_latest.pth`、`ckpt_ep{NN}.pth`、`ckpt_best_ns22.pth`、`ckpt_best_macro.pth`
- `history.json`
- `pseudo_labels.csv`
- すべて Drive ミラー、weights は Kaggle Dataset へ upload

## 期待結果
- 単 LB 0.91-0.94 (Antoine 0.936 標準 + Babych weight init で上振れ期待)
- R2 multi-iter NS で +0.01-0.02 lift → 0.92-0.95
- blend 4-way (NB4+Tucker+e17+exp032) で 0.948-0.952 (gold 0.954 射程)
""", "hdr"))'''

import re as _re
SRC = _re.sub(EXP017_HEADER_PATTERN, NEW_HEADER.replace("\\", "\\\\"), SRC, count=1, flags=_re.DOTALL)

# Cleanup leftover v1/v2/NFNet comments in code cells
_leftovers = [
    ('FORCE_FRESH = True        # ★ v2 で LR/ep 変更したので Drive 上の v1 ckpt は無視して新規学習',
     'FORCE_FRESH = True        # exp032 R1 新規学習、Drive 上の旧 ckpt は無視'),
    ('# ★ R1 v2: NFNet 用に LR を保守 (3e-4)、step 数を稼ぐため ep=25\\n# 元 r1 v1 (LR=5.1e-4, ep=20) は val 0.9065 で plateau — NFNet (BN-free) に sqrt scaling は強すぎ',
     '# exp032 R1: 保守 LR (3e-4) + ep=25 で step 数確保 (Babych weight init 利用、過学習回避)'),
    ('N_TOTAL_EPOCHS = 25       # ★ v1 比 +5、累計 step 3,060 → 3,825 で step 補填',
     'N_TOTAL_EPOCHS = 25       # Babych weight init から 25 ep finetune、focal+labeled_sc'),
    ('BATCH = 192               # ★ Blackwell 96GB の VRAM 活用 (v1 と同じ)',
     'BATCH = 192               # Blackwell 96GB 前提、A100 40GB なら 64-80'),
    ('LR = 3e-4                 # ★ v1 の 5.1e-4 → 3e-4、NFNet は AGC 無しなら保守 LR が定石',
     'LR = 3e-4                 # EfficientNet 系 + Babych init で安定 LR、warmup 3 ep + cosine'),
    ('NUM_WORKERS = 8           # ★ exp016 R2 と同じ (4→8 で +20-40%、batch=128 のデータ供給用)',
     'NUM_WORKERS = 8           # batch=192 のデータ供給用、Colab Blackwell 標準'),
    ('print(f"(LR/ep changed for v2 — resume would inherit old optimizer/scheduler state)")',
     'print(f"(exp032 fresh start, optimizer/scheduler reset)")'),
    ('exp017 R2 (next session) や infer NB から参照可能になる。',
     'exp032 R2 (next session) や infer NB から参照可能になる。'),
    ('TITLE = "birdclef2026 exp017 weights"',
     'TITLE = "birdclef2026 exp032 weights"'),
    ('version_notes = f"R1 v2 (LR=3e-4, ep=25) best_ns22={best_ns22:.4f}"',
     'version_notes = f"exp032 R1 (Babych b3 init, no Perch, LR=3e-4, ep=25) best_ns22={best_ns22:.4f}"'),
    ('print(f"exp017 R1 full session summary")',
     'print(f"exp032 R1 full session summary")'),
    # SLUG references in case any remain
    ('"birdclef2026-exp017-weights"', '"birdclef2026-exp032-weights"'),
    ('birdclef2026-exp017-weights', 'birdclef2026-exp032-weights'),
]
for old, new in _leftovers:
    if old in SRC:
        SRC = SRC.replace(old, new)
        print(f"  cleaned: {old[:70]}...")
    else:
        # Try escaped newline variant
        old_e = old.replace("\\n", "\n")
        if old_e != old and old_e in SRC:
            SRC = SRC.replace(old_e, new.replace("\\n", "\n"))
            print(f"  cleaned (escaped): {old[:70]}...")

# ============================================================================
# Patch 7: Mel param を Babych pretrain に整合 (image size (224, 512))
# 元: n_fft=2048, hop=512, n_mels=256, fmin=20, fmax=16000
# 新: n_fft=4096, hop=312, n_mels=224, fmin=0, fmax=16000
# → 5s 入力で image (224, 512) を生成、Babych b3 conv feature と整合
# ============================================================================
_mel_patches = [
    ('N_FFT          = 2048',     'N_FFT          = 4096        # exp032: Babych b3 mel param 整合'),
    ('HOP_LENGTH     = 512',      'HOP_LENGTH     = 312         # exp032: 5s × 32k / 512 ≈ 312 → image (224, 512)'),
    ('N_MELS         = 256',      'N_MELS         = 224         # exp032: Babych b3 mel param 整合'),
    ('FMIN           = 20',       'FMIN           = 0           # exp032: Babych b3 mel param 整合'),
]
for old, new in _mel_patches:
    if old in SRC:
        SRC = SRC.replace(old, new)
        print(f"  mel-patch: {old[:50]}")
    else:
        print(f"  WARNING: mel-patch not found: {old[:50]}")

# ============================================================================
# Patch 5: ipynb output path
# ============================================================================
SRC = SRC.replace(
    'HERE = Path(__file__).parent',
    'HERE = Path(__file__).parent  # experiment/exp032/notebook',
    1,
)
SRC = re.sub(
    r'out_path = HERE / "nb_train\.ipynb"',
    'out_path = HERE / "nb_train.ipynb"',
    SRC,
)

# Slug / dataset references in output cells (later add via push script)
SRC = SRC.replace(
    "birdclef2026-exp017-weights",
    "birdclef2026-exp032-weights",
)
# Drive path uses Path object concat: "output" / "exp017" / "r1"
SRC = SRC.replace(
    '"output" / "exp017"',
    '"output" / "exp032"',
)
SRC = SRC.replace(
    "output/exp017/",
    "output/exp032/",
)

# ============================================================================
# Patch 6: Insert Babych weight DL cell BEFORE model creation
# We add it after the data prep cell (cell 2) so weights are in Colab local
# ============================================================================
BABYCH_DL_CELL = '''cells.append(code_cell(r"""# ============================================================
# Cell 2b: ★ exp032: Download Babych BC25 1位 ensemble weight (b3 iter3)
# ============================================================
# Babych の XC + BC25 multi-iter NS 済 weight を backbone init に使う。
# Antoine Masq 公開 path (discussion 700763) の shortcut。
import os, time, subprocess
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

BABYCH_DIR = LOCAL_DATA / "babych_ensemble"
BABYCH_DIR.mkdir(parents=True, exist_ok=True)

# Check if already downloaded
existing = list(BABYCH_DIR.glob("tf_efficientnet_b3*.pt"))
if existing:
    print(f"[exp032] Babych b3 weight already exists: {existing[0].name}")
else:
    print("[exp032] Downloading nikitababich/birdclef2025-1st-place-ensemble (b3 only)...")
    api = KaggleApi(); api.authenticate()
    # Download b3 .pt only (avoid 458MB full DL)
    files = api.dataset_list_files("nikitababich/birdclef2025-1st-place-ensemble").files
    b3_files = [f.name for f in files if f.name.startswith("tf_efficientnet_b3") and f.name.endswith(".pt")]
    if not b3_files:
        raise FileNotFoundError("tf_efficientnet_b3 *.pt not found in babych dataset")
    target = b3_files[0]
    print(f"  target file: {target}")
    api.dataset_download_file("nikitababich/birdclef2025-1st-place-ensemble",
                               file_name=target, path=str(BABYCH_DIR),
                               force=True, quiet=False)
    # Unzip if needed (Kaggle download_file sometimes wraps in .zip)
    zip_file = BABYCH_DIR / (target + ".zip")
    if zip_file.exists():
        import zipfile
        with zipfile.ZipFile(zip_file) as zf:
            zf.extractall(BABYCH_DIR)
        zip_file.unlink()
    found = list(BABYCH_DIR.glob("tf_efficientnet_b3*.pt"))
    assert found, f"Download failed, no .pt in {BABYCH_DIR}"
    print(f"  -> {found[0].name} ({found[0].stat().st_size/1e6:.1f} MB)")
""", "babych-dl"))

'''

# Insert AFTER Cell 2 (data prep), BEFORE Cell 3 (imports + config)
# Find the Cell 3 cells.append marker
_target = 'cells.append(code_cell(r"""# ============================================================\n# Cell 3: Imports'
if _target in SRC:
    SRC = SRC.replace(_target, BABYCH_DL_CELL + _target, 1)
    print("  Inserted Babych DL cell before Cell 3")
else:
    print("  WARNING: Cell 3 marker not found, Babych DL cell not inserted")

# Also update the Babych init code to look for the local path
SRC = SRC.replace(
    'Path("/content/drive/MyDrive/kaggle/birdclef2026/external/birdclef2025-1st-place-ensemble"),',
    'LOCAL_DATA / "babych_ensemble",',
)

# Exec the patched gen
exec(SRC, {"__file__": str(HERE / "_gen_nb_train_inner.py")})
