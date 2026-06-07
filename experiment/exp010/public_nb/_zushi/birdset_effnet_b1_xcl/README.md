---
library_name: transformers
tags: []
---

# EfficientNet (trained on XCL from BirdSet)

Efficient trained on the XCL dataset from BirdSet, covering 9736 bird species from Xeno-Canto. Please refer to the [BirdSet Paper](https://arxiv.org/pdf/2403.10380) and the 
[BirdSet Repository](https://github.com/DBD-research-group/BirdSet/tree/main) for further information. 


## How to use
The BirdSet data needs a custom processor that is available in the BirdSet repository. The model does not have a processor available. 
The model accepts a mono image (spectrogram) as input (e.g., `torch.Size([16, 1, 256, 417])`)

- The model is trained on 5-second clips of bird vocalizations.
- num_channels: 1
- pretrained checkpoint: google/efficientnet-b1
- sampling_rate: 32_000
- normalize spectrogram: mean: -4.268, std: 4.569 (from esc-50)
- spectrogram: n_fft: 2048, hop_length: 2048, power: 2.0
- melscale: n_mels: 256, n_stft: 1025
- dbscale: top_db: 80

See [model implementation](https://github.com/DBD-research-group/BirdSet/blob/main/birdset/modules/models/birdset_models/efficientnet_bs.py).
Run in [Google Colab](https://colab.research.google.com/drive/15Y4k8kvUV8k7Jay76r_X-wnKi6p8v-7N?usp=sharing):

```python
from transformers import EfficientNetForImageClassification
import torch
import torchaudio
from torchvision import transforms
import requests
import torchaudio
import io


# download the audio file of a bird sound: Common Craw
url = "https://xeno-canto.org/704485/download"
response = requests.get(url)
audio, sample_rate = torchaudio.load(io.BytesIO(response.content))
print("Original shape and sample rate: ", audio.shape, sample_rate)
# crop to 5 seconds
audio = audio[:, : 5 * sample_rate]
# resample to 32kHz
resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=32000)
audio = resample(audio)
print("Resampled shape and sample rate: ", audio.shape, 32000)


CACHE_DIR = "../../data_birdset"  # Change this to your own cache directory

# Load the model
model = EfficientNetForImageClassification.from_pretrained(
    "DBD-research-group/EfficientNet-B1-BirdSet-XCL",
    num_channels=1,
    cache_dir=CACHE_DIR,
    ignore_mismatched_sizes=True,
)


class PowerToDB(torch.nn.Module):
    """
    A power spectrogram to decibel conversion layer. See birdset.datamodule.components.augmentations
    """

    def __init__(self, ref=1.0, amin=1e-10, top_db=80.0):
        super(PowerToDB, self).__init__()
        # Initialize parameters
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

    def forward(self, S):
        # Convert S to a PyTorch tensor if it is not already
        S = torch.as_tensor(S, dtype=torch.float32)

        if self.amin <= 0:
            raise ValueError("amin must be strictly positive")

        if torch.is_complex(S):
            magnitude = S.abs()
        else:
            magnitude = S

        # Check if ref is a callable function or a scalar
        if callable(self.ref):
            ref_value = self.ref(magnitude)
        else:
            ref_value = torch.abs(torch.tensor(self.ref, dtype=S.dtype))

        # Compute the log spectrogram
        log_spec = 10.0 * torch.log10(
            torch.maximum(magnitude, torch.tensor(self.amin, device=magnitude.device))
        )
        log_spec -= 10.0 * torch.log10(
            torch.maximum(ref_value, torch.tensor(self.amin, device=magnitude.device))
        )

        # Apply top_db threshold if necessary
        if self.top_db is not None:
            if self.top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = torch.maximum(log_spec, log_spec.max() - self.top_db)

        return log_spec

# Initialize preprocessors
spectrogram_converter = torchaudio.transforms.Spectrogram(
    n_fft=2048, hop_length=256, power=2.0
)
mel_converter = torchaudio.transforms.MelScale(
    n_mels=256, n_stft=1025, sample_rate=32_000
)
powerToDB = PowerToDB(top_db=80)


def preprocess(audio, sample_rate_of_audio):
    """
    Preprocess the audio to the format that the model expects
    - Resample to 32kHz
    - Convert to melscale spectrogram n_fft: 2048, hop_length: 256, power: 2. melscale: n_mels: 256, n_stft: 1025
    - Normalize the melscale spectrogram with mean: -4.268, std: 4.569 (from AudioSet)

    """
    spectrogram = spectrogram_converter(audio)
    spectrogram = spectrogram.to(torch.float32)
    melspec = mel_converter(spectrogram)
    dbscale = powerToDB(melspec)
    normalized_dbscale = transforms.Normalize((-4.268,), (4.569,))(dbscale)
    # add batch dimension if needed
    if normalized_dbscale.dim() == 3:
        normalized_dbscale = normalized_dbscale.unsqueeze(0)
    return normalized_dbscale

preprocessed_audio = preprocess(audio, sample_rate)
print("Preprocessed_audio shape:", preprocessed_audio.shape)

logits = model(preprocessed_audio).logits
print("Logits shape: ", logits.shape)

top5 = torch.topk(logits, 5)
print("Top 5 logits:", top5.values)
print("Top 5 predicted classes:")
print([model.config.id2label[i] for i in top5.indices.squeeze().tolist()])
```

## Model Source
- **Repository:** [BirdSet Repository](https://github.com/DBD-research-group/BirdSet/tree/main)
- **Paper [optional]:** [BirdSet Paper](https://arxiv.org/pdf/2403.10380)

## Citation