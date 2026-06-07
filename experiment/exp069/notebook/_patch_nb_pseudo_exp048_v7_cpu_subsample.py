"""v7 patch for exp069b: REVERT to CPU mode + subsample train_soundscapes.

理由: GPU 化で device mismatch エラーが反復発生 (exp048 NB は CPU 設計、10+ tensor 創作点が CPU 固定)。
全 GPU patch を完全に当てるのが困難。確実に通る CPU mode に戻し、subsample で 12h 上限内に収める。

Changes:
  1. DEVICE = "cpu" (CUDA detect 削除、強制 CPU)
  2. ONNX providers を CPU only に revert
  3. test_files を 4000 files に subsample (random.seed 固定)

推定時間: 80 min × (4000/700) ≈ 7.6h (Kaggle 12h 内)
Pseudo coverage: 全 train_soundscapes の 37.5%、distillation には十分
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_exp048.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

patches = [
    # 1. Force DEVICE = "cpu" (revert v3 auto-detect)
    {
        "old": 'DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"\nprint(f"[exp069b] DEVICE = {DEVICE}")\n',
        "new": 'DEVICE = "cpu"  # ★ exp069b v7: forced CPU (GPU caused multi-place device mismatch、safer to use CPU + subsample)\nprint(f"[exp069b] DEVICE = {DEVICE} (CPU mode for stability)")\n',
    },
    # 2. ONNX providers revert to CPU only (Perch session)
    {
        "old": "session = ort.InferenceSession(ONNX_MODEL, sess_opts, providers=[\"CUDAExecutionProvider\", \"CPUExecutionProvider\"])\n",
        "new": "session = ort.InferenceSession(ONNX_MODEL, sess_opts, providers=[\"CPUExecutionProvider\"])\n",
    },
    # 3. ONNX providers revert to CPU only (Tucker SED session)
    {
        "old": "                                providers=[\"CUDAExecutionProvider\", \"CPUExecutionProvider\"])\n",
        "new": "                                providers=[\"CPUExecutionProvider\"])\n",
    },
    # 4. Subsample train_soundscapes to 4000 files
    {
        "old": "# === exp069b: Stage A NB4 v11 + ResSSM inference on TRAIN_SC ===\ntest_files = sorted(glob.glob(str(TRAIN_SC_DIR / \"*.ogg\")))\nif len(test_files) == 0:\n    raise RuntimeError(f\"No train_soundscapes files at {TRAIN_SC_DIR}\")\nprint(f\"exp069b: {len(test_files)} train_soundscape files for pseudo gen\")\n",
        "new": "# === exp069b: Stage A NB4 v11 + ResSSM inference on TRAIN_SC (v7 CPU + subsample) ===\nimport random as _random\n_all_train_sc = sorted(glob.glob(str(TRAIN_SC_DIR / \"*.ogg\")))\nif len(_all_train_sc) == 0:\n    raise RuntimeError(f\"No train_soundscapes files at {TRAIN_SC_DIR}\")\n# v7: subsample 4000 of ~10,658 files for 12h Kaggle limit (CPU mode)\nN_SUBSAMPLE = 4000\n_rng_sub = _random.Random(42)\nif len(_all_train_sc) > N_SUBSAMPLE:\n    test_files = sorted(_rng_sub.sample(_all_train_sc, N_SUBSAMPLE))\n    print(f\"exp069b v7: subsampled {N_SUBSAMPLE} / {len(_all_train_sc)} train_soundscape files (CPU mode)\")\nelse:\n    test_files = _all_train_sc\n    print(f\"exp069b v7: using all {len(test_files)} train_soundscape files\")\n",
    },
]

total = 0
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") != "code": continue
    src = "".join(c["source"])
    new_src = src
    for p in patches:
        if p["old"] in new_src:
            new_src = new_src.replace(p["old"], p["new"], 1)
            total += 1
            print(f"  Cell {i}: patch applied: '{p['old'][:60].strip()}...'")
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None

print(f"\nTotal: {total}/{len(patches)} patches applied")
assert total == len(patches), f"Expected {len(patches)} patches, applied {total}"

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"v7 patched: {NB_PATH}")
