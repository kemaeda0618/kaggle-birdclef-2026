"""v8 patch for exp069b: GPU mode with set_default_device + full 10,658 files.

戦略:
  - torch.set_default_device("cuda") を NB 冒頭で設定 → 以降の全 torch.tensor() 等が cuda 上に作成
  - 個別 .to(DEVICE) patch も維持 (idempotent)
  - .numpy() の前に .cpu() を補完 (6 箇所)
  - subsample 解除、full 10,658 files で実行
  - machine_shape: NvidiaTeslaT4

期待時間: GPU で 6-8h、Kaggle 12h 内
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_exp048.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

patches = [
    # 1. DEVICE: revert from v7 CPU to GPU + set_default_device
    {
        "old": 'DEVICE = "cpu"  # ★ exp069b v7: forced CPU (GPU caused multi-place device mismatch、safer to use CPU + subsample)\nprint(f"[exp069b] DEVICE = {DEVICE} (CPU mode for stability)")\n',
        "new": ('import torch as _torch_init\n'
                'if _torch_init.cuda.is_available():\n'
                '    _torch_init.set_default_device("cuda")\n'
                '    DEVICE = "cuda"\n'
                'else:\n'
                '    DEVICE = "cpu"\n'
                'print(f"[exp069b v8] DEVICE = {DEVICE} (set_default_device set)")\n'),
    },
    # 2. ONNX Perch session: CPU only → CUDA + CPU fallback
    {
        "old": "session = ort.InferenceSession(ONNX_MODEL, sess_opts, providers=[\"CPUExecutionProvider\"])\n",
        "new": "session = ort.InferenceSession(ONNX_MODEL, sess_opts, providers=[\"CUDAExecutionProvider\", \"CPUExecutionProvider\"])\n",
    },
    # 3. ONNX Tucker SED session: CPU only → CUDA + CPU fallback
    {
        "old": "                                providers=[\"CPUExecutionProvider\"])\n",
        "new": "                                providers=[\"CUDAExecutionProvider\", \"CPUExecutionProvider\"])\n",
    },
    # 4. Remove subsample, use full
    {
        "old": ('# === exp069b: Stage A NB4 v11 + ResSSM inference on TRAIN_SC (v7 CPU + subsample) ===\n'
                'import random as _random\n'
                '_all_train_sc = sorted(glob.glob(str(TRAIN_SC_DIR / "*.ogg")))\n'
                'if len(_all_train_sc) == 0:\n'
                '    raise RuntimeError(f"No train_soundscapes files at {TRAIN_SC_DIR}")\n'
                '# v7: subsample 4000 of ~10,658 files for 12h Kaggle limit (CPU mode)\n'
                'N_SUBSAMPLE = 4000\n'
                '_rng_sub = _random.Random(42)\n'
                'if len(_all_train_sc) > N_SUBSAMPLE:\n'
                '    test_files = sorted(_rng_sub.sample(_all_train_sc, N_SUBSAMPLE))\n'
                '    print(f"exp069b v7: subsampled {N_SUBSAMPLE} / {len(_all_train_sc)} train_soundscape files (CPU mode)")\n'
                'else:\n'
                '    test_files = _all_train_sc\n'
                '    print(f"exp069b v7: using all {len(test_files)} train_soundscape files")\n'),
        "new": ('# === exp069b v8: Full TRAIN_SC (10,658 files) on GPU ===\n'
                'test_files = sorted(glob.glob(str(TRAIN_SC_DIR / "*.ogg")))\n'
                'if len(test_files) == 0:\n'
                '    raise RuntimeError(f"No train_soundscapes files at {TRAIN_SC_DIR}")\n'
                'print(f"exp069b v8: using all {len(test_files)} train_soundscape files (GPU mode)")\n'),
    },
    # 5-10. .numpy() → .cpu().numpy() patches
    {
        "old": "rs_perm = torch.randperm(n_lab, generator=torch.Generator().manual_seed(42)).numpy()\n",
        "new": "rs_perm = torch.randperm(n_lab, generator=torch.Generator().manual_seed(42)).cpu().numpy()\n",
    },
    {
        "old": "    all_corr_chk = rs_model(rs_emb_t, rs_fp_t, site_ids=rs_site_t, hours=rs_hour_t).numpy()\n",
        "new": "    all_corr_chk = rs_model(rs_emb_t, rs_fp_t, site_ids=rs_site_t, hours=rs_hour_t).cpu().numpy()\n",
    },
    {
        "old": "        y = torchaudio.functional.resample(y, sr, SR).squeeze(0).numpy()\n",
        "new": "        y = torchaudio.functional.resample(y, sr, SR).squeeze(0).cpu().numpy()\n",
    },
    {
        "old": "        probs_e10 = agg.squeeze(0).numpy()\n",
        "new": "        probs_e10 = agg.squeeze(0).cpu().numpy()\n",
    },
    {
        "old": "        _p_clip   = torch.sigmoid(_clip).numpy().astype(np.float32)\n",
        "new": "        _p_clip   = torch.sigmoid(_clip).cpu().numpy().astype(np.float32)\n",
    },
    {
        "old": "        _p_frame  = torch.sigmoid(_frame_max).numpy().astype(np.float32)\n",
        "new": "        _p_frame  = torch.sigmoid(_frame_max).cpu().numpy().astype(np.float32)\n",
    },
]

total = 0
not_found = []
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") != "code": continue
    src = "".join(c["source"])
    new_src = src
    for p_idx, p in enumerate(patches):
        if p["old"] in new_src:
            new_src = new_src.replace(p["old"], p["new"], 1)
            total += 1
            print(f"  Cell {i}: patch #{p_idx} applied: '{p['old'][:60].strip()}...'")
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None

print(f"\nTotal: {total}/{len(patches)} patches applied")
if total < len(patches):
    print("WARNING: not all patches applied — check NB structure")

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"v8 GPU patched: {NB_PATH}")
