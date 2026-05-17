import torch
import torch.nn as nn
import timm


class BirdCLEFModel(nn.Module):
    """
    EfficientNet-B0 ベースの音響分類モデル。
    入力: Mel Spectrogram (batch, 1, n_mels, time_frames)
    出力: 各クラスのロジット (batch, num_classes)
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]

        self.backbone = timm.create_model(
            model_cfg["name"],
            pretrained=model_cfg["pretrained"],
            in_chans=model_cfg["in_channels"],
            num_classes=0,      # headを除いたbackboneのみ使用
            global_pool="avg",
        )
        in_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, model_cfg["num_classes"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits
