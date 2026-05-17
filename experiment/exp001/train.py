"""
BirdCLEF+ 2026 - exp001 学習スクリプト

Usage (Google Colab):
    !pip install timm librosa scikit-learn pyyaml
    !python train.py --config config/config.yaml
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import yaml

sys.path.append(os.path.dirname(__file__))
from src.dataset import BirdCLEFDataset, mixup
from src.model import BirdCLEFModel
from src.utils import AverageMeter, compute_roc_auc, save_checkpoint, set_seed


# ─── データ準備 ────────────────────────────────────────────────────────────────

def load_data(config: dict):
    train_df = pd.read_csv(config["data"]["train_csv"])
    taxonomy_df = pd.read_csv(config["data"]["taxonomy_csv"])

    # ratingによるフィルタ
    min_rating = config["data"].get("min_rating", 0.0)
    if min_rating > 0.0:
        train_df = train_df[train_df["rating"] >= min_rating].reset_index(drop=True)
        print(f"Rating filter >= {min_rating}: {len(train_df)} samples")

    # 234クラスのラベルリスト（sample_submissionのカラム順に合わせる）
    sub_df = pd.read_csv(config["data"]["sample_submission_csv"], nrows=0)
    labels = [c for c in sub_df.columns if c != "row_id"]

    print(f"Total training samples: {len(train_df)}")
    print(f"Number of classes: {len(labels)}")
    return train_df, labels


def split_folds(train_df: pd.DataFrame, config: dict):
    """StratifiedKFoldでfold列を追加する。"""
    n_folds = config["train"]["n_folds"]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config["train"]["seed"])

    train_df = train_df.copy()
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["primary_label"])):
        train_df.loc[val_idx, "fold"] = fold

    return train_df


# ─── 学習・評価ループ ──────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: GradScaler,
    config: dict,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    loss_meter = AverageMeter()

    aug_cfg = config.get("augmentation", {})
    use_mixup = aug_cfg.get("use_mixup", True)
    mixup_alpha = aug_cfg.get("mixup_alpha", 0.5)
    mixup_prob = aug_cfg.get("mixup_prob", 0.5)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Mixup
        if use_mixup and np.random.random() < mixup_prob:
            images, labels, _ = mixup(images, labels, alpha=mixup_alpha)

        optimizer.zero_grad()
        with autocast(enabled=config["train"]["use_amp"]):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), n=images.size(0))

    return loss_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    labels: list,
    config: dict,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    loss_meter = AverageMeter()

    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        with autocast(enabled=config["train"]["use_amp"]):
            logits = model(images)
            loss = criterion(logits, targets)

        loss_meter.update(loss.item(), n=images.size(0))
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = compute_roc_auc(all_targets, all_preds, labels)
    return loss_meter.avg, metrics["mean_auc"]


# ─── メイン ────────────────────────────────────────────────────────────────────

def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    set_seed(config["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(config["output"]["dir"], exist_ok=True)

    # データ読み込み・fold分割
    train_df, labels = load_data(config)
    train_df = split_folds(train_df, config)

    fold = config["train"]["train_fold"]
    tr_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
    va_df = train_df[train_df["fold"] == fold].reset_index(drop=True)
    print(f"Fold {fold}: train={len(tr_df)}, valid={len(va_df)}")

    # Dataset / DataLoader
    tr_dataset = BirdCLEFDataset(tr_df, labels, config, mode="train")
    va_dataset = BirdCLEFDataset(va_df, labels, config, mode="valid")

    tr_loader = DataLoader(
        tr_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    va_loader = DataLoader(
        va_dataset,
        batch_size=config["train"]["batch_size"] * 2,
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
    )

    # モデル・オプティマイザ
    model = BirdCLEFModel(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    epochs = config["train"]["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=config["train"]["use_amp"])

    # Warmup: 最初のwarmup_epochs間はLRを線形にウォームアップ
    warmup_epochs = config["train"].get("warmup_epochs", 1)

    best_auc = 0.0
    best_epoch = 0

    print("\n" + "=" * 60)
    print(f"Training exp001 | Fold {fold} | {epochs} epochs")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # Warmup
        if epoch <= warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config["train"]["learning_rate"] * epoch / warmup_epochs

        tr_loss = train_one_epoch(model, tr_loader, optimizer, scaler, config, device)

        if epoch > warmup_epochs:
            scheduler.step()

        va_loss, va_auc = validate(model, va_loader, labels, config, device)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{epochs}"
            f" | LR={lr:.2e}"
            f" | Train Loss={tr_loss:.4f}"
            f" | Valid Loss={va_loss:.4f}"
            f" | Valid AUC={va_auc:.4f}"
        )

        if va_auc > best_auc:
            best_auc = va_auc
            best_epoch = epoch
            ckpt_path = os.path.join(config["output"]["dir"], "weights", f"best_fold{fold}.pth")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_auc": best_auc,
                    "config": config,
                    "labels": labels,
                },
                ckpt_path,
            )

    print("\n" + "=" * 60)
    print(f"Training finished!")
    print(f"Best AUC: {best_auc:.4f} @ Epoch {best_epoch}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
