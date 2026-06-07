"""v7 patch for exp069a: Keep mel_spectr_generator in fp32 (avoid fp16 mel filterbank issue).

Error V6: RuntimeError: expected scalar type Float but found Half
原因: model.to(device).to(dtype=fp16) で mel_spec の filterbank も fp16 になった
      wave input は fp32 → matmul で型 mismatch

Fix: model 全体 fp16 移動後、mel_spectr_generator だけ fp32 に戻す
     - mel 計算: fp32 (input wave fp32 + filterbank fp32 → mel fp32)
     - backbone/head 入力: Cell 16 で mel_spec.to(fp16) で cast 済
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_babych.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# In Cell 14, after model.to(device).to(dtype), add mel_spectr_generator.to(fp32)
OLD = "        model.to(device).to(dtype)\n"
NEW = ("        model.to(device).to(dtype)\n"
       "        # v7 fix: keep mel filterbank in fp32 to avoid wave-input/filterbank dtype mismatch\n"
       "        model.mel_spectr_generator.to(torch.float32)\n")

total = 0
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") != "code": continue
    src = "".join(c["source"])
    if OLD in src:
        new_src = src.replace(OLD, NEW)
        c["source"] = new_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None
        total += 1
        print(f"  Cell {i}: mel_spec back to fp32 patch applied")

assert total == 1, f"Expected 1 patch, applied {total}"
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"v7 patched: {NB_PATH}")
