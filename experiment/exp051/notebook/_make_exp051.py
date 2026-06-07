"""Clone exp049 NB → exp051 with Swin-Tiny backbone."""
import json, shutil
from pathlib import Path

SRC = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp049\notebook\nb_train_effv2s_r1.ipynb")
DST = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp051\notebook\nb_train_swin_r1.ipynb")
shutil.copy(SRC, DST)
nb = json.load(open(DST, encoding="utf-8"))

# 1. config cell: change BACKBONE + adjust batch for transformer
for c in nb["cells"]:
    if c.get("id") == "config":
        src = "".join(c["source"])
        src = src.replace(
            'BACKBONE = "tf_efficientnetv2_s.in21k_ft_in1k"',
            'BACKBONE = "swin_tiny_patch4_window7_224"'
        )
        # Swin needs smaller batch due to attention memory
        src = src.replace("BATCH_SIZE = 64", "BATCH_SIZE = 48  # Swin attention memory heavy")
        c["source"] = src.splitlines(keepends=True)
        print("[config] BACKBONE -> swin_tiny, BATCH 48")

# 2. hdr
for c in nb["cells"]:
    if c.get("id") == "hdr":
        src = "".join(c["source"])
        src = src.replace("exp049 EffNetV2-S", "exp051 Swin-Tiny (Transformer)")
        src = src.replace("tf_efficientnetv2_s.in21k_ft_in1k", "swin_tiny_patch4_window7_224")
        src = src.replace("~22M params", "~28M params (transformer)")
        src = src.replace("~6-8h", "~5-7h")
        c["source"] = src.splitlines(keepends=True)
        print("[hdr] updated")

# 3. Swin requires fixed input size 224x224. Add resize in model forward.
# Strategy: pad/resize mel to 224x224 before backbone
for c in nb["cells"]:
    if c.get("id") == "model":
        src = "".join(c["source"])
        old = """    def forward(self, mel):
        # mel: (B, n_mels, T)
        # Expand to 3 channels (replicate)
        x = mel.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, 3, n_mels, T)
        feat = self.backbone(x)  # (B, C, H, W)
        # Global avg over frequency dim, keep time dim
        feat = feat.mean(dim=2)  # (B, C, T')
        feat = feat.transpose(1, 2)  # (B, T', C)
        clipwise, framewise = self.head(feat)
        return clipwise, framewise"""
        new = """    def forward(self, mel):
        # mel: (B, n_mels=128, T)
        # Resize to 224x224 for Swin (fixed input requirement)
        x = mel.unsqueeze(1)  # (B, 1, n_mels, T)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)  # (B, 3, 224, 224)
        # Swin outputs (B, H, W, C) by default, we want feature map
        feat = self.backbone.forward_features(x)  # (B, H, W, C) for Swin
        # Swin output: (B, 49, C) after pooling or (B, 7, 7, C) with feature maps
        if feat.ndim == 4:
            # (B, H, W, C) -> avg over H -> (B, W, C) as time-like
            feat = feat.mean(dim=1)  # (B, W=7, C)
        elif feat.ndim == 3:
            # (B, N=49, C) -> reshape to (B, 7, 7, C) -> avg H -> (B, 7, C)
            B, N, C_ = feat.shape
            h = int(N ** 0.5)
            feat = feat.view(B, h, h, C_).mean(dim=1)
        clipwise, framewise = self.head(feat)
        return clipwise, framewise"""
        if old in src:
            src = src.replace(old, new)
            c["source"] = src.splitlines(keepends=True)
            print("[model] Swin-compatible forward updated")
        else:
            print("[model] WARN: forward pattern not found, manual fix may be needed")

# 4. save_dataset SLUG
for c in nb["cells"]:
    if c.get("id") == "save_dataset":
        src = "".join(c["source"])
        src = src.replace("birdclef2026-exp049-effv2s-r1", "birdclef2026-exp051-swin-tiny-r1")
        src = src.replace("BirdCLEF2026 exp049 EffNetV2-S R1", "BirdCLEF2026 exp051 Swin-Tiny R1")
        src = src.replace("/kaggle/working/exp049_upload", "/kaggle/working/exp051_upload")
        c["source"] = src.splitlines(keepends=True)
        print("[save_dataset] SLUG updated")

json.dump(nb, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nSaved {DST}")
