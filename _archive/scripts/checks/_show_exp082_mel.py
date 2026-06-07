import json
from pathlib import Path

for nb_path in ["experiment/exp082/notebook/nb_train_r1.ipynb",
                "experiment/exp017/notebook/nb_train_r2.ipynb"]:
    print(f"\n=== {nb_path} ===")
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c.get("cell_type") != "code": continue
        src = "".join(c.get("source", []))
        if "N_FFT" in src or "n_mels" in src.lower():
            # Print lines mentioning mel/fft params
            for ln in src.splitlines():
                if any(kw in ln for kw in ["N_MELS", "N_FFT", "HOP_LENGTH", "FMIN", "FMAX",
                                            "F_MIN", "F_MAX", "win_length", "n_fft", "n_mels",
                                            "mel_transform", "MelSpec"]):
                    print(f"  {ln[:100]}")
            break
