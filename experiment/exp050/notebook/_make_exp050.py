"""Clone exp049 NB → exp050 with ConvNeXt-pico backbone."""
import json, shutil
from pathlib import Path

SRC = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp049\notebook\nb_train_effv2s_r1.ipynb")
DST = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp050\notebook\nb_train_convnext_r1.ipynb")
shutil.copy(SRC, DST)
nb = json.load(open(DST, encoding="utf-8"))

# 1. config cell: change BACKBONE
for c in nb["cells"]:
    if c.get("id") == "config":
        src = "".join(c["source"])
        src = src.replace(
            'BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"',
            'BACKBONE = "convnextv2_pico.fcmae_ft_in1k"'
        )
        # Increase batch for smaller model
        src = src.replace("BATCH_SIZE = 64", "BATCH_SIZE = 96  # ConvNeXt-pico smaller → bigger batch OK")
        c["source"] = src.splitlines(keepends=True)
        print("[config] BACKBONE -> convnextv2_pico.fcmae_ft_in1k, BATCH 96")

# 2. hdr: update
for c in nb["cells"]:
    if c.get("id") == "hdr":
        src = "".join(c["source"])
        src = src.replace("exp049 EffNetV2-S", "exp050 ConvNeXt-pico v2")
        src = src.replace("tf_efficientnetv2_s.in21k_ft_in1k", "convnextv2_pico.fcmae_ft_in1k")
        src = src.replace("~22M params", "~9M params (lightweight)")
        src = src.replace("~6-8h", "~3-5h")
        c["source"] = src.splitlines(keepends=True)
        print("[hdr] updated")

# 3. save_dataset SLUG
for c in nb["cells"]:
    if c.get("id") == "save_dataset":
        src = "".join(c["source"])
        src = src.replace("birdclef2026-exp049-effv2s-r1", "birdclef2026-exp050-convnext-pico-r1")
        src = src.replace("BirdCLEF2026 exp049 EffNetV2-S R1", "BirdCLEF2026 exp050 ConvNeXt-pico R1")
        src = src.replace("/kaggle/working/exp049_upload", "/kaggle/working/exp050_upload")
        c["source"] = src.splitlines(keepends=True)
        print("[save_dataset] SLUG updated")

json.dump(nb, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nSaved {DST}")
