"""Compare MelSpectrogram setup. Fixed iteration."""
import json, re
from pathlib import Path

files = {
    "exp017 R2 train (REF)": Path("experiment/exp017/notebook/nb_train_r2.ipynb"),
    "exp081 R2 train": Path("experiment/exp081/notebook/nb_train_r2.ipynb"),
    "exp081 R2 infer": Path("experiment/exp081/notebook/nb_infer.ipynb"),
    "exp082 R2 train": Path("experiment/exp082/notebook/nb_train_r2.ipynb"),
    "exp082 R2 infer": Path("experiment/exp082/notebook/nb_infer.ipynb"),
    "exp083 R2 train": Path("experiment/exp083/notebook/nb_train_r2.ipynb"),
}

for name, fp in files.items():
    if not fp.exists():
        print(f"=== {name}: NOT FOUND ===")
        continue
    nb = json.loads(fp.read_text(encoding="utf-8"))
    all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
    print(f"\n=== {name} ===")
    # MelSpectrogram args
    matches = re.findall(r"MelSpectrogram\s*\(([^)]+)\)", all_src, re.DOTALL)
    for i, args in enumerate(matches):
        args = re.sub(r"\s+", " ", args).strip()
        print(f"  Mel[{i}]: {args[:250]}")
    # Key config params
    cfg = {}
    for k in ["N_MELS", "N_FFT", "HOP_LENGTH", "FMIN", "FMAX", "F_MIN", "F_MAX",
              "WINDOW_SEC", "TRAIN_DURATION", "VAL_DURATION", "SR ", "BACKBONE"]:
        m = re.search(rf"^\s*{re.escape(k.rstrip())}\s*=\s*([^#\n,]+)", all_src, re.MULTILINE)
        if m:
            cfg[k.rstrip()] = m.group(1).strip()
    for k, v in cfg.items():
        print(f"  {k:18} = {v[:60]}")
    # Per-sample norm
    if "mean()" in all_src and "mel" in all_src.lower():
        # Find lines with mel and mean
        for line in all_src.splitlines():
            if "mel[i]" in line and "mean" in line:
                print(f"  norm: {line.strip()[:120]}")
                break
            if "mel[i] =" in line and ("mean" in line or "std" in line):
                print(f"  norm: {line.strip()[:120]}")
                break
