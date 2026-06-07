"""v6 patch for exp069a: Mel STFT device fix.

Error: stft input on cpu but window on cuda
原因: model.to(cuda) で mel_spec_generator も cuda に移動した
      InferenceDataset 内で torch.tensor(wave) は CPU、それを cuda の MelSpectrogram に渡して fail

Fix: torch.tensor(wave) を mel_spec_generator と同じ device に移動
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_babych.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

patches = [
    # Line 197: full_wave path
    {
        "old": "        mel_spec_full = self.feature_extractor(torch.tensor(wave.reshape(1, -1)), without_norm=self.separate_norm)\n",
        "new": ('        _fe_dev = next(iter(self.feature_extractor.buffers()), torch.tensor([0.0])).device\n'
                '        mel_spec_full = self.feature_extractor(torch.tensor(wave.reshape(1, -1), dtype=torch.float32).to(_fe_dev), without_norm=self.separate_norm)\n'),
    },
    # Line 221: segments path (fallback, not main path but fix for safety)
    {
        "old": "            mel_specs = self.feature_extractor(torch.tensor(segments))\n",
        "new": ('            _fe_dev = next(iter(self.feature_extractor.buffers()), torch.tensor([0.0])).device\n'
                '            mel_specs = self.feature_extractor(torch.tensor(segments, dtype=torch.float32).to(_fe_dev))\n'),
    },
]

total = 0
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") != "code": continue
    src = "".join(c["source"])
    new_src = src
    for p in patches:
        if p["old"] in new_src:
            new_src = new_src.replace(p["old"], p["new"], 1)
            total += 1
            print(f"  Cell {i}: patched device-aware mel input")
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None

assert total == 2, f"Expected 2 patches, applied {total}"
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"v6 patched: {NB_PATH}")
