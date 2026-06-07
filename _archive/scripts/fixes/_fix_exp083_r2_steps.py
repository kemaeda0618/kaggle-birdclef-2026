"""Fix exp083 R2 train NB:
1. Cap N_STEPS_PER_EP to focal-based value (not sum of all sources)
2. Add LRU cache for SS audio reads (speed up PseudoScDS)
"""
import json
from pathlib import Path
import ast

P = Path("experiment/exp083/notebook/nb_train_r2.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

cell_map = {c.get("id"): (i, c) for i, c in enumerate(nb["cells"])}

# Fix 1: resume_or_init cell — cap N_STEPS_PER_EP to focal-based
if "resume_or_init" in cell_map:
    i, c = cell_map["resume_or_init"]
    src = "".join(c["source"])
    old = "N_STEPS_PER_EP = max(100, int(sum(SIZES) / BATCH))"
    new = """# ★ R2 fix: cap N_STEPS_PER_EP to focal-based (pseudo_sc adds 127K samples, would inflate to 2700 steps)
focal_size = SIZES[NAMES.index('focal')]
focal_share = SHARES.get('focal', 0.70)
N_STEPS_PER_EP = max(500, int(focal_size / BATCH / focal_share))"""
    if old in src:
        src = src.replace(old, new)
        c["source"] = src.splitlines(keepends=True)
        print(f"OK Fixed N_STEPS_PER_EP cap (cell {c.get('id')})")
    else:
        print(f"WARN: old pattern not found in resume_or_init cell")

# Fix 2: dataset cell — add LRU cache for SS audio
if "dataset" in cell_map:
    i, c = cell_map["dataset"]
    src = "".join(c["source"])
    # Insert LRU cache helper near top of cell after imports
    old_imports = """import soundfile as sf
import librosa

def _load_ogg(path, n_samples_target, start_sec=None):
    try:
        if start_sec is not None:
            wav, sr = sf.read(str(path), start=int(start_sec * SR),
                              frames=n_samples_target, dtype='float32', always_2d=False)
        else:
            wav, sr = sf.read(str(path), dtype='float32', always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)
    except Exception:
        return None"""
    new_imports = """import soundfile as sf
import librosa
from functools import lru_cache

# ★ R2 fix: LRU cache for SS audio (PseudoScDS reads same SS file 12x → cache hit)
# Per-worker cache (~4GB), 512 SS files = ~7.5GB total across workers
@lru_cache(maxsize=512)
def _load_full_audio_cached(path_str):
    try:
        wav, sr = sf.read(path_str, dtype='float32', always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)
    except Exception:
        return None

def _load_ogg(path, n_samples_target, start_sec=None):
    if start_sec is not None:
        full = _load_full_audio_cached(str(path))
        if full is None:
            return None
        start = int(start_sec * SR)
        end = start + n_samples_target
        if end <= len(full):
            return full[start:end].copy()
        out = np.zeros(n_samples_target, dtype=np.float32)
        avail = full[start:start + n_samples_target]
        out[:len(avail)] = avail
        return out
    try:
        wav, sr = sf.read(str(path), dtype='float32', always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav.astype(np.float32)
    except Exception:
        return None"""
    if old_imports in src:
        src = src.replace(old_imports, new_imports)
        c["source"] = src.splitlines(keepends=True)
        print(f"OK Added LRU cache for SS audio (cell {c.get('id')})")
    else:
        print(f"WARN: dataset cell old pattern not found")

P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# Syntax check
nb = json.loads(P.read_text(encoding="utf-8"))
n_ok, n_err = 0, 0
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                      for ln in src.splitlines())
    try:
        ast.parse(clean); n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  SyntaxError cell {c.get('id','?')}: {e}")
print(f"\nSyntax: {n_ok} OK, {n_err} ERRORS")
