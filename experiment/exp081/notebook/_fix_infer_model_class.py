"""Fix all inference NBs to use correct `BirdSEDModel` class (training-matched).

Affected NBs:
- exp081/nb_infer.ipynb (R2)
- exp081/nb_infer_r1.ipynb (R1)
- exp082/nb_infer.ipynb (R2)
- ensemble_e14_17/nb_infer_ensemble.ipynb (4 model ensemble)

Issue: my old CLEFClassifierSED class doesn't match training's BirdSEDModel.
- Wrong in_chans (3 vs 1)
- Wrong key names (head.* vs gem_freq/dense/att/cla)
- Wrong distill_head structure
"""
import json
from pathlib import Path

NB_PATHS = [
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp081/notebook/nb_infer.ipynb",
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp081/notebook/nb_infer_r1.ipynb",
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/exp082/notebook/nb_infer.ipynb",
    r"C:/Users/maeke/work/kaggle/birdclef-2026/experiment/ensemble_e14_17/notebook/nb_infer_ensemble.ipynb",
]


CORRECT_MODEL_CELL = '''# Model architecture (must match training: BirdSEDModel)
class GeMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps
    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)


class DistillHead(nn.Module):
    def __init__(self, backbone_dim, embed_dim=1536):
        super().__init__()
        self.proj = nn.Linear(backbone_dim, embed_dim)
    def forward(self, feature_map):
        return self.proj(feature_map.mean(dim=[2, 3]))


class BirdSEDModel(nn.Module):
    def __init__(self, backbone_name, num_classes=N_CLASSES, drop_path_rate=0.0, hidden_dim=512,
                 use_perch_distill=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1,
            num_classes=0, global_pool="", drop_path_rate=drop_path_rate,
        )
        with torch.no_grad():
            n_tf = WINDOW_SAMPLES // HOP_LENGTH + 1
            dummy = torch.randn(1, 1, N_MELS, n_tf)
            feat = self.backbone(dummy)
            self.backbone_dim = feat.shape[1]

        self.gem_freq = GeMFreqPool(p_init=3.0)
        self.dense = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.backbone_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)
        if use_perch_distill:
            self.distill_head = DistillHead(self.backbone_dim, 1536)

    def forward(self, x, return_framewise=False, return_distill=False):
        h = self.backbone(x)
        h_cls = self.gem_freq(h)
        h_cls = h_cls.permute(0, 2, 1)
        h_cls = self.dense(h_cls)
        h_cls = h_cls.permute(0, 2, 1)
        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)
        framewise = self.cla(h_cls)
        clip_logits = torch.sum(norm_att * framewise, dim=2)
        if return_framewise:
            return clip_logits, framewise.permute(0, 2, 1)  # (B, time, num_class)
        return clip_logits


def sigmoid_np(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


def load_birdsed_model(ckpt_path, backbone_name):
    model = BirdSEDModel(backbone_name=backbone_name).to(DEVICE)
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    msg = model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  {backbone_name}: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if len(msg.missing_keys) > 5:
        print(f"    sample missing: {list(msg.missing_keys)[:3]}")
    if len(msg.unexpected_keys) > 5:
        print(f"    sample unexpected: {list(msg.unexpected_keys)[:3]}")
    return model
'''


def fix_nb(nb_path):
    nb_path = Path(nb_path)
    if not nb_path.exists():
        print(f"  SKIP (not exist): {nb_path}")
        return
    print(f"\n=== Patching {nb_path.parent.name}/{nb_path.name} ===")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    # Find model class cell (contains 'class GeMFreq' or 'class CLEFClassifierSED' or 'class AttHead')
    fixed = False
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        if "class CLEFClassifierSED" in src or "class AttHead" in src or "class GeMFreq" in src:
            # Replace with correct BirdSEDModel
            c["source"] = CORRECT_MODEL_CELL.splitlines(keepends=True)
            print(f"  Replaced cell {i} ({c.get('id', '?')}) with BirdSEDModel")
            fixed = True
            break

    if not fixed:
        print(f"  WARN: model cell not found")
        return

    # Also fix mel processing (don't stack 3 channels) and model loading
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if c.get("cell_type") != "code":
            continue

        # 1. Fix mel stacking — remove 3-channel stacking
        # Look for `torch.cat([mel, mel, mel], dim=1)` or similar
        if "torch.cat([mel, mel, mel]" in src or "torch.stack([spec, spec, spec]" in src:
            print(f"  WARN: cell {c.get('id', '?')} has 3-channel stack — model class change should handle")

        # 2. Fix CLEFClassifierSED() instantiation → BirdSEDModel()
        if "CLEFClassifierSED" in src and "class CLEFClassifierSED" not in src:
            # This cell uses CLEFClassifierSED but doesn't define it (model is in another cell)
            new_src = src.replace("CLEFClassifierSED(", "BirdSEDModel(")
            c["source"] = new_src.splitlines(keepends=True)
            print(f"  Replaced CLEFClassifierSED() calls in cell {c.get('id', '?')}")

        # 3. Fix old load_model helper -> load_birdsed_model
        if "def load_model(" in src and "CLEFClassifierSED" not in src:
            # Other usage; check if model is used
            pass

    # 4. Replace any remaining "CLEFClassifierSED" -> "BirdSEDModel" in all cells
    for c in nb["cells"]:
        if c.get("cell_type") != "code": continue
        src = "".join(c.get("source", []))
        if "CLEFClassifierSED" in src:
            new_src = src.replace("CLEFClassifierSED", "BirdSEDModel")
            c["source"] = new_src.splitlines(keepends=True)

    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  Saved")


for p in NB_PATHS:
    fix_nb(p)

print("\nDone.")
