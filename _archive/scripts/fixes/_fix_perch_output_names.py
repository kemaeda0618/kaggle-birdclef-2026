"""Fix Perch ONNX OOM: use output_names=[embedding_name] instead of None.

Applies to exp083 R2 train NB (cell 'model' has PerchTeacher class).

Before:
    results = self.session.run(None, {self.input_name: wav_np})
    return torch.from_numpy(results[self._embed_idx]).float()

After:
    results = self.session.run([self._embed_name], {self.input_name: wav_np})
    return torch.from_numpy(results[0]).float()

This skips label head (14795 classes × frames) → no OOM
"""
import json
from pathlib import Path

P = Path("experiment/exp083/notebook/nb_train_r2.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

cell_map = {c.get("id"): (i, c) for i, c in enumerate(nb["cells"])}

if "model" not in cell_map:
    print("ERR: 'model' cell not found")
else:
    i, c = cell_map["model"]
    src = "".join(c["source"])

    # Old PerchTeacher pattern
    old = '''class PerchTeacher:
    def __init__(self, onnx_path, device_str="cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \\
            if device_str == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self._embed_idx = None
        for i, o in enumerate(self.session.get_outputs()):
            if o.shape and o.shape[-1] == PERCH_EMBED_DIM:
                self._embed_idx = i; break
        if self._embed_idx is None:
            self._embed_idx = 1
        print(f"PerchTeacher: embed_idx={self._embed_idx}, providers={self.session.get_providers()}")

    @torch.no_grad()
    def embed(self, waveforms_5s):
        wav_np = waveforms_5s.cpu().numpy().astype(np.float32)
        results = self.session.run(None, {self.input_name: wav_np})
        return torch.from_numpy(results[self._embed_idx]).float()'''

    new = '''class PerchTeacher:
    def __init__(self, onnx_path, device_str="cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \\
            if device_str == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        # ★ FIX (memory feedback_perch_onnx_output_filter): specify output_name to skip label head OOM
        self._embed_name = None
        for o in self.session.get_outputs():
            if o.shape and o.shape[-1] == PERCH_EMBED_DIM:
                self._embed_name = o.name; break
        if self._embed_name is None:
            # Fallback: 2nd output (typical embedding position)
            self._embed_name = self.session.get_outputs()[1].name
        print(f"PerchTeacher: embed_name={self._embed_name}, providers={self.session.get_providers()}")

    @torch.no_grad()
    def embed(self, waveforms_5s):
        wav_np = waveforms_5s.cpu().numpy().astype(np.float32)
        # ★ FIX: output_names=[embed_name] to skip label head (14795 classes → OOM with BATCH=128)
        results = self.session.run([self._embed_name], {self.input_name: wav_np})
        return torch.from_numpy(results[0]).float()'''

    if old in src:
        src = src.replace(old, new)
        c["source"] = src.splitlines(keepends=True)
        print(f"OK Fixed PerchTeacher in cell 'model'")
    else:
        # Try with different whitespace
        print(f"WARN: exact old pattern not found, trying line-by-line fix")
        # Search for key lines
        for line_old, line_new in [
            ("self._embed_idx = None",
             "self._embed_name = None"),
            ("self._embed_idx = i; break",
             "self._embed_name = o.name; break"),
            ("if self._embed_idx is None:",
             "if self._embed_name is None:"),
            ("self._embed_idx = 1",
             "self._embed_name = self.session.get_outputs()[1].name"),
            (f"PerchTeacher: embed_idx={{self._embed_idx}}",
             f"PerchTeacher: embed_name={{self._embed_name}}"),
            ("results = self.session.run(None, {self.input_name: wav_np})",
             "results = self.session.run([self._embed_name], {self.input_name: wav_np})  # FIX: skip label head OOM"),
            ("return torch.from_numpy(results[self._embed_idx]).float()",
             "return torch.from_numpy(results[0]).float()"),
        ]:
            if line_old in src:
                src = src.replace(line_old, line_new)
                print(f"  Replaced: {line_old[:60]}")
        c["source"] = src.splitlines(keepends=True)

P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# Verify
nb = json.loads(P.read_text(encoding="utf-8"))
src = "".join(nb["cells"][cell_map["model"][0]]["source"])
if "session.run([self._embed_name]" in src:
    print("\n[OK] Fix verified — output_names limit applied")
else:
    print("\n[WARN] Fix not visible in saved file — check manually")
    # Show relevant lines
    for ln in src.splitlines():
        if "session.run" in ln or "_embed_" in ln:
            print(f"  {ln.strip()[:150]}")

# Syntax check
import ast
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
print(f"\nSyntax: {n_ok} OK, {n_err} ERRORS")
