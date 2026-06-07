"""OOM fix patch for all 3 NBs:
1. Use sf.SoundFile + seek + read(chunk_len) instead of sf.read whole file
   → only read what we need, not entire (potentially minutes-long) OGG
2. Reduce batch size for safety
3. Free GPU cache between epochs
"""
import json
from pathlib import Path

NB_CONFIGS = [
    (Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp049\notebook\nb_train_effv2s_r1.ipynb"),
     {"BATCH_SIZE_OLD": "64", "BATCH_SIZE_NEW": "32"}),
    (Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp050\notebook\nb_train_convnext_r1.ipynb"),
     {"BATCH_SIZE_OLD": "96", "BATCH_SIZE_NEW": "48"}),
    (Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp051\notebook\nb_train_swin_r1.ipynb"),
     {"BATCH_SIZE_OLD": "48", "BATCH_SIZE_NEW": "24"}),
]

for NB, ctx in NB_CONFIGS:
    if not NB.exists():
        print(f"[SKIP] {NB.name}")
        continue
    nb = json.load(open(NB, encoding="utf-8"))
    n_changes = 0

    # Fix 1: BATCH_SIZE reduce
    for c in nb["cells"]:
        if c.get("id") == "config":
            src = "".join(c["source"])
            old = f"BATCH_SIZE = {ctx['BATCH_SIZE_OLD']}"
            new = f"BATCH_SIZE = {ctx['BATCH_SIZE_NEW']}  # was {ctx['BATCH_SIZE_OLD']}, reduced for OOM safety"
            if old in src:
                src = src.replace(old, new, 1)  # only first occurrence
                n_changes += 1
            c["source"] = src.splitlines(keepends=True)
            break

    # Fix 2: sf.read whole file → SoundFile seek + read chunk_len only
    for c in nb["cells"]:
        if c.get("id") == "dataset":
            src = "".join(c["source"])
            old = """        try:
            wav, sr = sf.read(str(audio_path), dtype="float32")
            if sr != self.sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
        except Exception:
            wav = np.zeros(self.chunk_len, dtype=np.float32)

        # Random crop or pad to chunk_len
        if len(wav) >= self.chunk_len:
            if self.training:
                start = np.random.randint(0, len(wav) - self.chunk_len + 1)
            else:
                start = (len(wav) - self.chunk_len) // 2
            wav = wav[start:start + self.chunk_len]
        else:
            pad = self.chunk_len - len(wav)
            wav = np.pad(wav, (0, pad), mode="constant")"""
            new = """        try:
            # OOM fix: use SoundFile to seek + read only chunk_len samples
            with sf.SoundFile(str(audio_path)) as f:
                sr = f.samplerate
                # Calculate chunk in source sample rate space
                src_chunk = int(self.chunk_len * sr / self.sr)
                total = f.frames
                if total > src_chunk:
                    if self.training:
                        start = np.random.randint(0, total - src_chunk + 1)
                    else:
                        start = (total - src_chunk) // 2
                    f.seek(start)
                    wav = f.read(src_chunk, dtype="float32")
                else:
                    wav = f.read(dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != self.sr:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sr)
        except Exception:
            wav = np.zeros(self.chunk_len, dtype=np.float32)

        # Pad if needed
        if len(wav) >= self.chunk_len:
            wav = wav[:self.chunk_len]
        else:
            pad = self.chunk_len - len(wav)
            wav = np.pad(wav, (0, pad), mode="constant")"""
            if old in src:
                src = src.replace(old, new)
                n_changes += 1
            c["source"] = src.splitlines(keepends=True)
            break

    # Fix 3: Free GPU cache + empty cache between epochs
    for c in nb["cells"]:
        if c.get("id") == "train_loop":
            src = "".join(c["source"])
            old = "    scheduler.step()"
            new = """    scheduler.step()
    torch.cuda.empty_cache()  # OOM safety"""
            if old in src and "torch.cuda.empty_cache()  # OOM safety" not in src:
                src = src.replace(old, new, 1)
                n_changes += 1
            c["source"] = src.splitlines(keepends=True)
            break

    json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{NB.parent.parent.name}/{NB.name}] {n_changes} changes (batch -> {ctx['BATCH_SIZE_NEW']})")

# Verify
print("\n=== Verify ===")
for NB, _ in NB_CONFIGS:
    nb = json.load(open(NB, encoding="utf-8"))
    for c in nb["cells"]:
        if c.get("id") == "config":
            src = "".join(c["source"])
            for line in src.split("\n"):
                if "BATCH_SIZE = " in line:
                    print(f"  {NB.parent.parent.name} [config] {line.strip()}")
        if c.get("id") == "dataset":
            src = "".join(c["source"])
            if "sf.SoundFile" in src:
                print(f"  {NB.parent.parent.name} [dataset] OK SoundFile seek used")
