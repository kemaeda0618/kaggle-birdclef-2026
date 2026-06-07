"""Check train_r2 saves ckpt with 'model_state' key (matches infer expectation)."""
import json
nb = json.load(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp032\notebook\nb_train_r2.ipynb", encoding="utf-8"))
for c in nb["cells"]:
    src = "".join(c.get("source", []))
    if "torch.save" in src or "model_state" in src or '"model":' in src:
        cid = c.get("id", "?")
        for line in src.split("\n"):
            if ("torch.save" in line or "model_state" in line or '"model_state"' in line or
                'model.state_dict' in line or '"model"' in line):
                if line.strip().startswith("#"):
                    continue
                print(f"  [{cid}] {line.strip()[:150]}")
