import ast
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset


class BirdCLEFDataset(Dataset):
    """
    BirdCLEF+ 2026 学習用Dataset。
    train.csvの各行から音声を読み込み、Mel Spectrogramに変換して返す。
    """

    def __init__(self, df: pd.DataFrame, labels: list[str], config: dict, mode: str = "train"):
        """
        Args:
            df: train.csvのDataFrame（fold分割済み）
            labels: taxonomy.csvから取得した234クラスのprimary_labelリスト（ソート済み）
            config: config.yamlの内容
            mode: "train" or "valid"
        """
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.label2idx = {label: i for i, label in enumerate(labels)}
        self.num_classes = len(labels)
        self.config = config
        self.mode = mode

        data_cfg = config["data"]
        self.train_audio_dir = data_cfg["train_audio_dir"]
        self.sr = data_cfg["sample_rate"]
        self.n_samples = data_cfg["n_samples"]
        self.n_mels = data_cfg["n_mels"]
        self.n_fft = data_cfg["n_fft"]
        self.hop_length = data_cfg["hop_length"]
        self.fmin = data_cfg["fmin"]
        self.fmax = data_cfg["fmax"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        audio = self._load_audio(row["filename"])
        audio = self._crop_or_pad(audio)

        if self.mode == "train":
            audio = self._augment(audio)

        melspec = self._to_melspec(audio)
        image = torch.tensor(melspec).unsqueeze(0)  # (1, n_mels, time_frames)

        label = self._get_label(row)

        return image, label

    def _load_audio(self, filename: str) -> np.ndarray:
        path = f"{self.train_audio_dir}/{filename}"
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        return audio

    def _crop_or_pad(self, audio: np.ndarray) -> np.ndarray:
        """5秒に切り出す。短い場合はゼロパディング、長い場合はランダムクロップ（train）または中央クロップ（valid）。"""
        if len(audio) >= self.n_samples:
            if self.mode == "train":
                start = np.random.randint(0, len(audio) - self.n_samples + 1)
            else:
                start = (len(audio) - self.n_samples) // 2
            audio = audio[start : start + self.n_samples]
        else:
            pad_len = self.n_samples - len(audio)
            audio = np.pad(audio, (0, pad_len), mode="constant", constant_values=0.0)
        return audio.astype(np.float32)

    def _to_melspec(self, audio: np.ndarray) -> np.ndarray:
        """音声をMel Spectrogramに変換し[0,1]に正規化する。"""
        melspec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        melspec = librosa.power_to_db(melspec, ref=np.max)
        melspec = (melspec - melspec.min()) / (melspec.max() - melspec.min() + 1e-8)
        return melspec.astype(np.float32)

    def _get_label(self, row) -> torch.Tensor:
        """primary_label と secondary_labels からmulti-hotベクトルを生成する。"""
        label = np.zeros(self.num_classes, dtype=np.float32)

        if row["primary_label"] in self.label2idx:
            label[self.label2idx[row["primary_label"]]] = 1.0

        secondary = row["secondary_labels"]
        if isinstance(secondary, str):
            try:
                secondary = ast.literal_eval(secondary)
            except Exception:
                secondary = []
        for sl in secondary:
            if sl in self.label2idx:
                label[self.label2idx[sl]] = 1.0

        return torch.tensor(label, dtype=torch.float32)

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        """学習時の音声レベルaugmentation。"""
        aug_cfg = self.config.get("augmentation", {})

        # Gaussian noise
        if aug_cfg.get("gaussian_noise", False) and np.random.random() < aug_cfg.get("noise_prob", 0.5):
            noise = np.random.randn(len(audio)).astype(np.float32) * aug_cfg.get("noise_std", 0.005)
            audio = audio + noise

        # Time shift
        if aug_cfg.get("time_shift", False) and np.random.random() < aug_cfg.get("time_shift_prob", 0.5):
            shift = np.random.randint(-self.sr // 2, self.sr // 2)
            audio = np.roll(audio, shift)

        return audio.astype(np.float32)


def mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.5):
    """
    Mixup augmentation（バッチレベル）。
    Returns:
        mixed_images, mixed_labels, lambda
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[perm]
    mixed_labels = lam * labels + (1 - lam) * labels[perm]
    return mixed_images, mixed_labels, lam
