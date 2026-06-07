"""Build exp112 V2 with OV batching (BATCH_FILES=3) for both sed-infer and e29-infer.

Speedup target: 5-10 min saving (~30% on heavy OV inference parts).
LB risk: 0 (numerically identical, just larger batch sizes per OV call).

Changes:
  - sed-infer (Tucker OV 5-fold): batch 3 files per OV call
  - e29-infer (exp106 OV 2-fold): batch 3 files per OV call
  - Vectorize per-instance standardize (remove Python loop)
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

V1_NB = Path("experiment/exp112/notebook/nb_exp112_e106_fold12_w30_40_30.ipynb")
V2_NB = Path("experiment/exp112/notebook/nb_exp112_v2_batched.ipynb")

nb = json.loads(V1_NB.read_text(encoding="utf-8"))

# ============================================================
# NEW sed-infer cell (batched, Tucker OV 5-fold)
# ============================================================
SED_BATCHED = r"""# === Stage B: Tucker 5-fold SED OpenVINO ensemble (★ V2: BATCH_FILES=3) ===
import librosa, glob
from scipy.ndimage import gaussian_filter1d

# Install openvino offline (idempotent — both cells call this)
try:
    import openvino as ov
    print(f"openvino preinstalled: {ov.__version__}")
except ImportError:
    _hits = sorted(glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True))
    assert _hits, "openvino wheel not found — attach ttahara/birdclef-2026-download-wheels"
    _WHEEL_DIR = str(Path(_hits[0]).parent)
    !pip install -q --no-deps {_WHEEL_DIR}/openvino-*.whl {_WHEEL_DIR}/openvino_telemetry-*.whl
    import openvino as ov
    print(f"openvino installed: {ov.__version__}")

N_MELS_SED = 256
N_FFT_SED  = 2048
HOP_SED    = 512
FMIN_SED   = 20
FMAX_SED   = 16000
TOP_DB_SED = 80


def find_sed_ov_dir():
    # Locate Tucker SED OV IR directory (kernel_sources mount path may vary)
    for p in [
        Path("/kaggle/input/birdclef2026-tucker-sed-ov"),
        Path("/kaggle/input/datasets/maekeso/birdclef2026-tucker-sed-ov"),
    ]:
        if p.exists():
            return p
    hits = sorted(Path("/kaggle/input").rglob("sed_fold0.xml"))
    assert hits, "sed_fold0.xml not found — attach maekeso/birdclef2026-tucker-sed-ov"
    return hits[0].parent


def audio_to_mel_batched(chunks_2d):
    # chunks_2d: shape (N*12, 160000) -> mel shape (N*12, 1, 256, 313)
    mels = []
    for x in chunks_2d:
        s = librosa.feature.melspectrogram(
            y=x, sr=SR, n_fft=N_FFT_SED, hop_length=HOP_SED,
            n_mels=N_MELS_SED, fmin=FMIN_SED, fmax=FMAX_SED, power=2.0,
        )
        s = librosa.power_to_db(s, top_db=TOP_DB_SED)
        s = (s - s.mean()) / (s.std() + 1e-6)
        mels.append(s)
    return np.stack(mels)[:, None].astype(np.float32)


def sigmoid_sed(x):
    return (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))).astype(np.float32)


sed_ov_dir = find_sed_ov_dir()
sed_xml_paths = sorted(sed_ov_dir.glob("sed_fold*.xml"),
                        key=lambda p: int(re.search(r"sed_fold(\d+)", p.name).group(1)))
print(f"SED OV dir: {sed_ov_dir}")
print(f"SED OV folds available: {[p.name for p in sed_xml_paths]}")
assert len(sed_xml_paths) >= 3, f"expected >= 3 folds, got {len(sed_xml_paths)}"
# ★ exp111: restore Tucker SED 5-fold full ensemble (exp102 fold 1+2 で時間節約済、Tucker 5-fold 維持可能)
print(f"SED OV folds USED (exp111 = 5-fold full): {[p.name for p in sed_xml_paths]}")

_ov_core_sed = ov.Core()
sed_compiled = [_ov_core_sed.compile_model(str(p), "CPU") for p in sed_xml_paths]
print(f"compiled {len(sed_compiled)} OV models")

# ★ V2: batched inference
BATCH_FILES_SED = 3
print(f"[batching] BATCH_FILES_SED={BATCH_FILES_SED}")

t0 = time.time()
N_FILES = len(audio_cache)
probs_sed = [None] * N_FILES

for batch_start in range(0, N_FILES, BATCH_FILES_SED):
    batch_end = min(batch_start + BATCH_FILES_SED, N_FILES)
    n_batch = batch_end - batch_start

    # stack chunks across batch (n_batch * 12, 160000)
    batch_chunks_raw = []
    for fi in range(batch_start, batch_end):
        chunks = audio_cache[fi].reshape(N_WINDOWS, WINDOW_SAMPLES)
        batch_chunks_raw.append(chunks)
    batch_chunks_raw = np.concatenate(batch_chunks_raw, axis=0)  # (n_batch*12, 160000)

    mel_batch = audio_to_mel_batched(batch_chunks_raw)  # (n_batch*12, 1, 256, 313)

    # 5-fold ensemble on batched input
    p_sum_batch = np.zeros((n_batch * N_WINDOWS, N_CLASSES), dtype=np.float32)
    for compiled in sed_compiled:
        ov_outs = compiled(mel_batch)
        clip_logits = ov_outs[compiled.outputs[0]]   # (n_batch*12, N_CLASSES)
        frame_max = ov_outs[compiled.outputs[1]].max(axis=1)  # (n_batch*12, N_CLASSES)
        p_sum_batch += 0.5 * sigmoid_sed(clip_logits) + 0.5 * sigmoid_sed(frame_max)
    p_mean_batch = p_sum_batch / len(sed_compiled)  # (n_batch*12, N_CLASSES)

    # split per file + gaussian smooth
    for j in range(n_batch):
        fi = batch_start + j
        p_file = p_mean_batch[j * N_WINDOWS:(j + 1) * N_WINDOWS]  # (12, N_CLASSES)
        p_smooth = gaussian_filter1d(p_file, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
        probs_sed[fi] = p_smooth

    if (batch_end % 30 == 0) or batch_end == N_FILES:
        print(f"  SED-OV-batched [{batch_end}/{N_FILES}] {time.time()-t0:.0f}s")

probs_sed = np.stack(probs_sed)
print(f"SED OV (batched) done: {probs_sed.shape} in {time.time()-t0:.0f}s")
"""


# ============================================================
# NEW e29-infer cell (batched, exp106 OV 2-fold)
# ============================================================
E29_BATCHED = r"""# === exp106 OV inference (2-fold ensemble fold 1+2, ★ V2: BATCH_FILES=3) ===
import torchaudio, glob

try:
    import openvino as ov
    print(f"openvino: {ov.__version__}")
except ImportError:
    _hits = sorted(glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True))
    assert _hits, "openvino wheel not found"
    _WHEEL_DIR = str(Path(_hits[0]).parent)
    !pip install -q --no-deps {_WHEEL_DIR}/openvino-*.whl {_WHEEL_DIR}/openvino_telemetry-*.whl
    import openvino as ov

# Locate exp106 OV IR files
EXP106_DIR = None
for _p in [
    Path("/kaggle/input/birdclef2026-e106-3fold-ov"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-e106-3fold-ov"),
]:
    if _p.exists():
        EXP106_DIR = _p; break
if EXP106_DIR is None:
    _hits = glob.glob("/kaggle/input/**/exp106_fold*.xml", recursive=True)
    if _hits:
        EXP106_DIR = Path(_hits[0]).parent
assert EXP106_DIR is not None, "exp106 OV IR not found"
print(f"[ov] EXP106_DIR: {EXP106_DIR}")

FOLDS_E106 = [1, 2]
EXP106_IR = {}
for _f in FOLDS_E106:
    _xml_hits = list(EXP106_DIR.rglob(f"exp106_fold{_f}.xml"))
    assert _xml_hits, f"exp106_fold{_f}.xml not found in {EXP106_DIR}"
    EXP106_IR[_f] = _xml_hits[0]
    print(f"  fold {_f}: {EXP106_IR[_f]}")

_core_e106 = ov.Core()
_ov_e106 = {}
for _f, _xml in EXP106_IR.items():
    _ov_e106[_f] = _core_e106.compile_model(str(_xml), "CPU")
print(f"[ov] compiled {len(_ov_e106)} fold(s)")

_mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=2048, hop_length=512,
    n_mels=256, f_min=20, f_max=16000, power=2.0,
)
_db_tf = torchaudio.transforms.AmplitudeToDB(top_db=80)

# ★ V2: batched inference
BATCH_FILES_E106 = 3
print(f"[batching] BATCH_FILES_E106={BATCH_FILES_E106}")

t0 = time.time()
N_FILES = len(audio_cache)
probs_e17 = [None] * N_FILES

with torch.no_grad():
    for batch_start in range(0, N_FILES, BATCH_FILES_E106):
        batch_end = min(batch_start + BATCH_FILES_E106, N_FILES)
        n_batch = batch_end - batch_start

        # stack chunks across batch
        batch_chunks = []
        for fi in range(batch_start, batch_end):
            chunks = audio_cache[fi].reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
            batch_chunks.append(chunks)
        batch_chunks = np.concatenate(batch_chunks, axis=0)  # (n_batch*12, 160000)

        wav_t = torch.from_numpy(batch_chunks).unsqueeze(1)  # (n_batch*12, 1, 160000)
        mel = _db_tf(_mel_tf(wav_t))
        # vectorized per-instance standardize (remove Python loop)
        mel_mean = mel.mean(dim=(2, 3), keepdim=True)
        mel_std = mel.std(dim=(2, 3), keepdim=True) + 1e-6
        mel = (mel - mel_mean) / mel_std
        mel_np = mel.numpy()  # (n_batch*12, 1, 256, 313)

        # OV 2-fold ensemble on batched input
        clip_sum = None; frame_sum = None
        for _f, compiled in _ov_e106.items():
            ov_out = compiled(mel_np)
            clip_ov = ov_out[compiled.outputs[0]]    # (n_batch*12, N_CLASSES)
            frame_ov = ov_out[compiled.outputs[1]]   # (n_batch*12, T_frames, N_CLASSES)
            if clip_sum is None:
                clip_sum = clip_ov.copy()
                frame_sum = frame_ov.copy()
            else:
                clip_sum += clip_ov
                frame_sum += frame_ov
        n_folds = len(_ov_e106)
        clip_avg = clip_sum / n_folds      # (n_batch*12, N_CLASSES)
        frame_avg = frame_sum / n_folds    # (n_batch*12, T_frames, N_CLASSES)

        # split per file and process
        for j in range(n_batch):
            fi = batch_start + j
            clip_file = clip_avg[j * N_WINDOWS:(j + 1) * N_WINDOWS]       # (12, N_CLASSES)
            frame_file = frame_avg[j * N_WINDOWS:(j + 1) * N_WINDOWS]     # (12, T_frames, N_CLASSES)
            frame_max = frame_file.max(axis=1)                            # (12, N_CLASSES)

            p_clip = 1.0 / (1.0 + np.exp(-np.clip(clip_file, -50, 50)))
            p_frame = 1.0 / (1.0 + np.exp(-np.clip(frame_max, -50, 50)))
            p_mean = (0.5 * p_clip + 0.5 * p_frame).astype(np.float32)
            p_smooth = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
            probs_e17[fi] = p_smooth

        if (batch_end % 30 == 0) or batch_end == N_FILES:
            print(f"  e106-ov-batched [{batch_end}/{N_FILES}] {time.time()-t0:.0f}s")

probs_e17 = np.stack(probs_e17).astype(np.float32)
print(f"e106-ov (batched) done: {probs_e17.shape} in {time.time()-t0:.0f}s  (folds={list(_ov_e106.keys())}, batch={BATCH_FILES_E106})")
"""


# ============================================================
# Patch cells
# ============================================================
patched = {"sed-infer": False, "e29-infer": False}
for c in nb["cells"]:
    cid = c.get("id")
    if cid == "sed-infer":
        c["source"] = SED_BATCHED.splitlines(keepends=True)
        patched["sed-infer"] = True
    elif cid == "e29-infer":
        c["source"] = E29_BATCHED.splitlines(keepends=True)
        patched["e29-infer"] = True

assert all(patched.values()), f"missing patches: {patched}"

V2_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {V2_NB} ({V2_NB.stat().st_size} bytes)")

# Compile check
import ast
for cid in ["sed-infer", "e29-infer"]:
    for c in nb["cells"]:
        if c.get("id") == cid:
            src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                               for ln in src.splitlines())
            try:
                compile(clean, f"<{cid}>", "exec")
                print(f"  [OK] {cid} compiles")
            except SyntaxError as e:
                print(f"  [FAIL] {cid}: {e}")
            break
