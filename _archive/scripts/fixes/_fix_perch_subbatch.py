"""Fix Perch OOM: sub-batch processing within PerchTeacher.embed().

128 batch → split into 2x 64-chunk Perch calls.
exp083 R1 used BATCH=64 with Perch and worked → 64 is safe sub-batch size.
"""
import json
from pathlib import Path

P = Path("experiment/exp083/notebook/nb_train_r2.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

cell_map = {c.get("id"): (i, c) for i, c in enumerate(nb["cells"])}

i, c = cell_map["model"]
src = "".join(c["source"])

# Find the embed method and replace with sub-batched version
old = '''    @torch.no_grad()
    def embed(self, waveforms_5s):
        wav_np = waveforms_5s.cpu().numpy().astype(np.float32)
        # ★ FIX: output_names=[embed_name] to skip label head (14795 classes → OOM with BATCH=128)
        results = self.session.run([self._embed_name], {self.input_name: wav_np})
        return torch.from_numpy(results[0]).float()'''

new = '''    @torch.no_grad()
    def embed(self, waveforms_5s):
        wav_np = waveforms_5s.cpu().numpy().astype(np.float32)
        B = wav_np.shape[0]
        # ★ FIX v2: sub-batch processing (Perch v2 ONNX OOM at BATCH=128 due to label/protopnet head)
        # exp083 R1 used BATCH=64 with Perch and worked → 64 is safe sub-batch size
        PERCH_SUB_BATCH = 64
        if B <= PERCH_SUB_BATCH:
            results = self.session.run([self._embed_name], {self.input_name: wav_np})
            return torch.from_numpy(results[0]).float()
        # Sub-batch: split B into PERCH_SUB_BATCH chunks, concat embeddings
        outs = []
        for s in range(0, B, PERCH_SUB_BATCH):
            chunk = wav_np[s:s + PERCH_SUB_BATCH]
            r = self.session.run([self._embed_name], {self.input_name: chunk})
            outs.append(r[0])
        return torch.from_numpy(np.concatenate(outs, axis=0)).float()'''

if old in src:
    src = src.replace(old, new)
    c["source"] = src.splitlines(keepends=True)
    print(f"OK Applied sub-batch Perch fix")
else:
    print(f"WARN: old embed method pattern not found, searching alternatives...")
    if "self.session.run([self._embed_name]" in src:
        # Just patch in place
        old2 = "results = self.session.run([self._embed_name], {self.input_name: wav_np})"
        new2 = """B_perch = wav_np.shape[0]
        PERCH_SUB_BATCH = 64
        if B_perch <= PERCH_SUB_BATCH:
            results = self.session.run([self._embed_name], {self.input_name: wav_np})
        else:
            outs = []
            for s in range(0, B_perch, PERCH_SUB_BATCH):
                r = self.session.run([self._embed_name], {self.input_name: wav_np[s:s+PERCH_SUB_BATCH]})
                outs.append(r[0])
            results = [np.concatenate(outs, axis=0)]"""
        if old2 in src:
            src = src.replace(old2, new2)
            c["source"] = src.splitlines(keepends=True)
            print(f"OK Applied sub-batch patch (alt method)")
        else:
            print(f"FAIL: cannot find embed method to patch")

P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# Verify
nb = json.loads(P.read_text(encoding="utf-8"))
src_after = "".join(nb["cells"][cell_map["model"][0]]["source"])
if "PERCH_SUB_BATCH" in src_after:
    print("\n[OK] Sub-batch fix verified in saved NB")
    # Show the embed method
    in_embed = False
    for ln in src_after.splitlines():
        if "def embed(" in ln:
            in_embed = True
        if in_embed:
            print(f"  {ln}")
        if in_embed and ("return torch.from_numpy" in ln and "concatenate" in ln) or (in_embed and "return torch.from_numpy(results[0])" in ln and "PERCH_SUB_BATCH" not in ln):
            # End of method (after final return)
            if "return torch.from_numpy(np.concatenate" in ln:
                break
else:
    print("[FAIL] Sub-batch fix not visible")

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
