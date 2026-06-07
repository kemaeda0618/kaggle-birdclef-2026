"""Show config cell of nb_train_r2.ipynb to verify mel params used in R2 training."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train_r2.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if "BACKBONE" in src and "N_FFT" in src:
        cid = c.get("id", "?")
        print(f"=== cell {cid} ===")
        for line in src.split("\n"):
            ls = line.strip()
            if any(k in ls for k in ["BACKBONE", "N_FFT", "HOP_LENGTH", "N_MELS", "FMIN", "FMAX", "USE_PERCH_DISTILL"]):
                print(f"  {line}")
        print()
