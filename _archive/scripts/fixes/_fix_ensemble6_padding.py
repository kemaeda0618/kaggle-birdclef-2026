"""Fix prepare_input_sliding padding to match working v6 pattern (samples-based / 2)."""
import json
from pathlib import Path

P = Path("experiment/ensemble6/notebook/nb_infer_ensemble6.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    if "def prepare_input_sliding" in src:
        old = """def prepare_input_sliding(audio_60s, duration):
    \"\"\"Sliding window for 10s/20s: padded + 12 windows step 5s.\"\"\"
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
    return chunks"""

        new = """def prepare_input_sliding(audio_60s, duration):
    \"\"\"Sliding window for 10s/20s: padded + 12 windows step 5s.

    ★ Padding spec (working v6 pattern):
    - padded_sec = (12-1)*5 + duration = 65 (10s) or 75 (20s)
    - pad_lead_samples = (padded_sec - 60) * SR // 2  ← samples-based (0.5s 切捨て無)
    - 10s case: 80000 samples (2.5s) lead, 80000 trail
    - 20s case: 240000 samples (7.5s) lead, 240000 trail
    \"\"\"
    padded_sec = (N_OUT_ROWS - 1) * 5 + duration   # 65 (10s), 75 (20s)
    padded_samples = padded_sec * SR
    # ★ Samples-based division to preserve 0.5s precision (working v6 pattern)
    pad_lead_samples = ((padded_sec - 60) * SR) // 2
    audio_padded = np.zeros(padded_samples, dtype=np.float32)
    audio_padded[pad_lead_samples:pad_lead_samples + len(audio_60s)] = audio_60s
    chunks = np.zeros((N_OUT_ROWS, SR * duration), dtype=np.float32)
    step_samples = 5 * SR
    win_samples = duration * SR
    for k in range(N_OUT_ROWS):
        start = k * step_samples
        chunks[k] = audio_padded[start:start + win_samples]
    return chunks"""

        if old in src:
            src = src.replace(old, new)
            c["source"] = src.splitlines(keepends=True)
            print("OK Fixed padding (samples-based division, 0.5s precision preserved)")

P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# Verify the fix
nb = json.loads(P.read_text(encoding="utf-8"))
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    if "def prepare_input_sliding" in src:
        # Verify samples-based calc
        if "((padded_sec - 60) * SR) // 2" in src:
            print("VERIFIED: samples-based division applied")
            # Print test values
            for duration in [10, 20]:
                padded_sec = (12 - 1) * 5 + duration
                samples_lead = ((padded_sec - 60) * 32000) // 2
                print(f"  duration={duration}s: padded_sec={padded_sec}, pad_lead={samples_lead} samples = {samples_lead/32000:.1f}s")
        break

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
        print(f"  SyntaxError: {e}")
print(f"\nSyntax: {n_ok} OK, {n_err} ERRORS")
