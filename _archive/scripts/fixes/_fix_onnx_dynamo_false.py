"""Fix: pass dynamo=False to torch.onnx.export.
torch 2.x is dynamo=True default and requires onnxscript module.
Legacy behavior with dynamo=False is robust for our model class.
"""
import json
from pathlib import Path

P = Path("experiment/convert_to_ov/notebook/convert_to_ov.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    if "torch.onnx.export" in src and "dynamo=" not in src:
        # Add dynamo=False arg
        old = """    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        input_names=['mel'],
        output_names=['clip_logits', 'framewise'],
        dynamic_axes={
            'mel': {0: 'batch'},
            'clip_logits': {0: 'batch'},
            'framewise': {0: 'batch'},
        },
        opset_version=17,
        do_constant_folding=True,
    )"""
        new = """    torch.onnx.export(
        wrapper, dummy, str(onnx_path),
        input_names=['mel'],
        output_names=['clip_logits', 'framewise'],
        dynamic_axes={
            'mel': {0: 'batch'},
            'clip_logits': {0: 'batch'},
            'framewise': {0: 'batch'},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,    # ★ torch 2.x dynamo default requires onnxscript; use legacy path
    )"""
        if old in src:
            src = src.replace(old, new)
            c["source"] = src.splitlines(keepends=True)
            print(f"OK Added dynamo=False to torch.onnx.export")

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
        print(f"  SyntaxError: {e}")
print(f"\nSyntax: {n_ok} OK, {n_err} ERRORS")
