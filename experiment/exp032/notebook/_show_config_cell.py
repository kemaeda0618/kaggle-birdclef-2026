"""Show config cell of nb_infer.ipynb to verify BACKBONE and mel params."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_infer.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if "BACKBONE" in src:
        cid = c.get("id", "?")
        print(f"=== cell {cid} ===")
        for line in src.split("\n"):
            ls = line.strip()
            if any(k in ls for k in ["BACKBONE", "N_FFT", "HOP_LENGTH", "N_MELS", "FMIN", "FMAX", "SR ", "TRAIN_SAMPLES", "USE_PERCH_DISTILL", "PERCH_EMBED_DIM"]):
                print(f"  {line}")
        print()
