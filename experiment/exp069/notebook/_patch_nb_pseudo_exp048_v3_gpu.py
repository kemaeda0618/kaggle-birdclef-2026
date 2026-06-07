"""v3 patch: enable GPU in exp069b NB.

10,658 train_soundscapes files × CPU inference = ~20h 推定で Kaggle 12h 限界超過。
GPU 有効化で ~2-4h に短縮。

Changes:
  1. DEVICE = "cuda" if available else "cpu" (line 89)
  2. ONNX Perch session: providers に CUDAExecutionProvider 追加 (line 967)
  3. ONNX Tucker SED session: providers に CUDAExecutionProvider 追加 (line 1232)
  4. exp029 R3 e17_model.to(DEVICE) (line 1422)
  5. exp029 R3 e17_mel_tf.to(DEVICE) (line 1425)
  6. exp029 R3 inference loop で _wav_t.to(DEVICE)
"""
import json, sys, io, os, re
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_exp048.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

patches = [
    # Cell containing DEVICE = "cpu"
    {
        "old": "DEVICE = \"cpu\"\n",
        "new": 'DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"\nprint(f"[exp069b] DEVICE = {DEVICE}")\n',
    },
    # Perch ONNX session (CPU only) → CUDA + CPU fallback
    {
        "old": "session = ort.InferenceSession(ONNX_MODEL, sess_opts, providers=[\"CPUExecutionProvider\"])\n",
        "new": "session = ort.InferenceSession(ONNX_MODEL, sess_opts, providers=[\"CUDAExecutionProvider\", \"CPUExecutionProvider\"])\n",
    },
    # Tucker SED ONNX session (CPU only) → CUDA + CPU fallback
    {
        "old": "                                providers=[\"CPUExecutionProvider\"])\n",
        "new": "                                providers=[\"CUDAExecutionProvider\", \"CPUExecutionProvider\"])\n",
    },
    # e17 PyTorch model: CPU → DEVICE
    {
        "old": "e17_model = _E17SED().to(torch.device(\"cpu\"))\n",
        "new": "e17_model = _E17SED().to(torch.device(DEVICE))\n",
    },
    {
        "old": "e17_mel_tf = _E17MelTF().to(torch.device(\"cpu\"))\n",
        "new": "e17_mel_tf = _E17MelTF().to(torch.device(DEVICE))\n",
    },
    # exp029 R3 inference loop: move wav tensor to DEVICE before mel transform
    {
        "old": "        _wav_t = torch.from_numpy(_chunks).unsqueeze(1)        # (12, 1, 160000)\n",
        "new": "        _wav_t = torch.from_numpy(_chunks).unsqueeze(1).to(DEVICE)        # (12, 1, 160000)\n",
    },
]

# Apply patches to entire NB source
total_patched = 0
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])
    new_src = src
    for p in patches:
        if p["old"] in new_src:
            new_src = new_src.replace(p["old"], p["new"], 1)
            total_patched += 1
            print(f"  Cell {i}: applied patch '{p['old'][:60].strip()}'")
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None

print(f"\nTotal patches applied: {total_patched}/{len(patches)}")
assert total_patched == len(patches), f"Some patches not applied (expected {len(patches)})"

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nv3 GPU patched: {NB_PATH}")
