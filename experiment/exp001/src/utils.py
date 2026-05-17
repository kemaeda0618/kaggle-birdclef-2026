import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """学習中の値（loss, AUC等）の移動平均を管理する。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_roc_auc(targets: np.ndarray, preds: np.ndarray, labels: list[str]) -> dict:
    """
    クラスごとのROC AUCと平均値を計算する。
    targets, preds: shape (n_samples, n_classes)

    Returns:
        dict with "mean_auc" and per-class AUC keyed by label name.
    """
    result = {}
    aucs = []
    for i, label in enumerate(labels):
        # 正例が存在しないクラスはスキップ
        if targets[:, i].sum() == 0:
            continue
        try:
            auc = roc_auc_score(targets[:, i], preds[:, i])
            result[label] = auc
            aucs.append(auc)
        except Exception:
            pass

    result["mean_auc"] = float(np.mean(aucs)) if aucs else 0.0
    return result


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_auc", 0.0)
