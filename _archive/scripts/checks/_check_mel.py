"""Compare MelSpectrogram setup between exp081 R2 train vs infer."""
import json, re
from pathlib import Path

files = {
    "exp081 R2 train": Path(r"experiment/exp081/notebook/nb_train_r2.ipynb"),
    "exp081 R2 infer": Path(r"experiment/exp081/notebook/nb_infer.ipynb"),
    "exp017 R2 train (ref)": Path(r"experiment/exp017/notebook/nb_train_r2.ipynb"),
    "exp082 R2 train": Path(r"experiment/exp082/notebook/nb_train_r2.ipynb"),
    "exp082 R2 infer": Path(r"experiment/exp082/notebook/nb_infer.ipynb"),
    "exp083 R2 train": Path(r"experiment/exp083/notebook/nb_train_r2.ipynb"),
}

for name, fp in files.items():
    if not fp.exists():
        print(f"\n=== {name}: NOT FOUND ===")
        continue
    print(f"\n=== {name} ===")
    nb = json.loads(fp.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell.get("source", []))
        # Find MelSpectrogram call
        m = re.search(r"MelSpectrogram\s*\(([^)]+)\)", src, re.DOTALL)
        if m:
            args = m.group(1).strip()
            # Clean whitespace
            args = re.sub(r"\s+", " ", args)
            print(f"  MelSpectrogram({args[:200]})")
        # Mel config
        for k in ["N_MELS", "N_FFT", "HOP_LENGTH", "FMIN", "FMAX", "F_MIN", "F_MAX", "WINDOW_SEC", "TRAIN_DURATION", "VAL_DURATION"]:
            for line in src.splitlines():
                m = re.match(rf"\s*{k}\s*=\s*([^#\n]+)", line)
                if m:
                    val = m.group(1).strip().rstrip(",")
                    print(f"  {k:18} = {val[:60]}")
                    break
        # Per-sample normalize check
        if "mel[i].mean()" in src or "mel.mean()" in src:
            for line in src.splitlines():
                if "mean()" in line and "mel" in line:
                    print(f"  norm: {line.strip()[:120]}")
                    break
        break
