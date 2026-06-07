"""Build exp112 V5: e106 OV async + thread tuning + bigger batch.

Options applied:
  2. BATCH_FILES_E106: 3 → 12 (4x batch)
  5. OV async API: 2-fold parallel inference (true parallelism)
  7. INFERENCE_NUM_THREADS=2 per fold (4 threads total on 4 vCPU)

For Tucker SED: BATCH_FILES_SED 3 → 6 (smaller bump, kept sequential 5-fold).

Expected total savings:
  - e106 async + batch=12 + threads: -20 min (raw 60min → ~30-35min)
  - Tucker batch=6: -3-5 min
  - Total: ~25 min from V4's ~80-92 min → ~55-67 min

Implementation notes:
  - Async API: create_infer_request() + start_async() + wait()
  - Outputs must be .copy()'d before next start_async (buffer reuse)
  - Thread setting via compile_model(config={"INFERENCE_NUM_THREADS": "2"})
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

V4_NB = Path("experiment/exp112/notebook/nb_exp112_v4_pathfix.ipynb")
V5_NB = Path("experiment/exp112/notebook/nb_exp112_v5_async.ipynb")

nb = json.loads(V4_NB.read_text(encoding="utf-8"))


# ============================================================
# Replace sed-infer cell: batch=6, sequential 5-fold (no async)
# ============================================================
SED_V5 = r"""# === Stage B: Tucker 5-fold SED OV ensemble (★ V5: BATCH_FILES=6) ===
import librosa, glob
from scipy.ndimage import gaussian_filter1d

try:
    import openvino as ov
    print(f"openvino preinstalled: {ov.__version__}")
except ImportError:
    _direct = Path("/kaggle/input/notebooks/ttahara/birdclef-2026-download-wheels/wheels")
    if _direct.exists():
        _WHEEL_DIR = str(_direct)
    else:
        _hits = sorted(glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True))
        assert _hits, "openvino wheel not found"
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
    # ★ V4: notebooks/ direct path first
    for p in [
        Path("/kaggle/input/notebooks/maekeso/birdclef2026-tucker-sed-ov"),
        Path("/kaggle/input/birdclef2026-tucker-sed-ov"),
        Path("/kaggle/input/datasets/maekeso/birdclef2026-tucker-sed-ov"),
    ]:
        if p.exists():
            return p
    hits = sorted(Path("/kaggle/input").rglob("sed_fold0.xml"))
    assert hits, "sed_fold0.xml not found"
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
print(f"SED OV folds: {[p.name for p in sed_xml_paths]}")
assert len(sed_xml_paths) >= 3, f"expected >= 3 folds, got {len(sed_xml_paths)}"

# ★ V5: compile Tucker with default threads (sequential 5-fold loop)
_ov_core_sed = ov.Core()
sed_compiled = [_ov_core_sed.compile_model(str(p), "CPU") for p in sed_xml_paths]
print(f"compiled {len(sed_compiled)} OV models (sequential)")

# ★ V5: BATCH_FILES_SED = 6 (was 3)
BATCH_FILES_SED = 6
print(f"[batching] BATCH_FILES_SED={BATCH_FILES_SED}")

t0 = time.time()
N_FILES = len(audio_cache)
probs_sed = [None] * N_FILES

for batch_start in range(0, N_FILES, BATCH_FILES_SED):
    batch_end = min(batch_start + BATCH_FILES_SED, N_FILES)
    n_batch = batch_end - batch_start

    batch_chunks_raw = []
    for fi in range(batch_start, batch_end):
        chunks = audio_cache[fi].reshape(N_WINDOWS, WINDOW_SAMPLES)
        batch_chunks_raw.append(chunks)
    batch_chunks_raw = np.concatenate(batch_chunks_raw, axis=0)

    mel_batch = audio_to_mel_batched(batch_chunks_raw)

    p_sum_batch = np.zeros((n_batch * N_WINDOWS, N_CLASSES), dtype=np.float32)
    for compiled in sed_compiled:
        ov_outs = compiled(mel_batch)
        clip_logits = ov_outs[compiled.outputs[0]]
        frame_max = ov_outs[compiled.outputs[1]].max(axis=1)
        p_sum_batch += 0.5 * sigmoid_sed(clip_logits) + 0.5 * sigmoid_sed(frame_max)
    p_mean_batch = p_sum_batch / len(sed_compiled)

    for j in range(n_batch):
        fi = batch_start + j
        p_file = p_mean_batch[j * N_WINDOWS:(j + 1) * N_WINDOWS]
        p_smooth = gaussian_filter1d(p_file, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
        probs_sed[fi] = p_smooth

    if (batch_end % 30 == 0) or batch_end == N_FILES:
        print(f"  SED-OV-V5 [{batch_end}/{N_FILES}] {time.time()-t0:.0f}s")

probs_sed = np.stack(probs_sed)
print(f"SED OV V5 done: {probs_sed.shape} in {time.time()-t0:.0f}s (batch={BATCH_FILES_SED})")
"""


# ============================================================
# Replace e29-infer cell: batch=12 + 2-fold async + 2 threads/fold
# ============================================================
E29_V5 = r"""# === exp106 OV inference V5 (★ batch=12 + 2-fold ASYNC + 2 threads/fold) ===
import torchaudio, glob

try:
    import openvino as ov
    print(f"openvino: {ov.__version__}")
except ImportError:
    _direct = Path("/kaggle/input/notebooks/ttahara/birdclef-2026-download-wheels/wheels")
    if _direct.exists():
        _WHEEL_DIR = str(_direct)
    else:
        _hits = sorted(glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True))
        assert _hits, "openvino wheel not found"
        _WHEEL_DIR = str(Path(_hits[0]).parent)
    !pip install -q --no-deps {_WHEEL_DIR}/openvino-*.whl {_WHEEL_DIR}/openvino_telemetry-*.whl
    import openvino as ov

# Locate exp106 OV IR files (★ V4: notebooks/ direct first)
EXP106_DIR = None
for _p in [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-e106-3fold-ov"),
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
    assert _xml_hits, f"exp106_fold{_f}.xml not found"
    EXP106_IR[_f] = _xml_hits[0]
    print(f"  fold {_f}: {EXP106_IR[_f]}")

# ★ V5: compile with INFERENCE_NUM_THREADS=2 (2 folds × 2 threads = 4 threads = full Kaggle CPU)
_core_e106 = ov.Core()
_ov_e106 = {}
_compile_cfg = {"INFERENCE_NUM_THREADS": "2"}
for _f, _xml in EXP106_IR.items():
    _ov_e106[_f] = _core_e106.compile_model(str(_xml), "CPU", config=_compile_cfg)
print(f"[ov] compiled {len(_ov_e106)} fold(s) with INFERENCE_NUM_THREADS=2")

# ★ V5: create persistent InferRequest per fold (for async reuse)
_infer_reqs = {_f: _ov_e106[_f].create_infer_request() for _f in _ov_e106}
print(f"[ov] created {len(_infer_reqs)} async InferRequests")

# Mel transform
_mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=2048, hop_length=512,
    n_mels=256, f_min=20, f_max=16000, power=2.0,
)
_db_tf = torchaudio.transforms.AmplitudeToDB(top_db=80)

# ★ V5: batch=12 (was 3)
BATCH_FILES_E106 = 12
print(f"[batching] BATCH_FILES_E106={BATCH_FILES_E106}")

t0 = time.time()
N_FILES = len(audio_cache)
probs_e17 = [None] * N_FILES

# Determine input name from first compiled model
_INPUT_NAME = list(_ov_e106.values())[0].inputs[0].any_name

with torch.no_grad():
    for batch_start in range(0, N_FILES, BATCH_FILES_E106):
        batch_end = min(batch_start + BATCH_FILES_E106, N_FILES)
        n_batch = batch_end - batch_start

        batch_chunks = []
        for fi in range(batch_start, batch_end):
            chunks = audio_cache[fi].reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
            batch_chunks.append(chunks)
        batch_chunks = np.concatenate(batch_chunks, axis=0)

        wav_t = torch.from_numpy(batch_chunks).unsqueeze(1)
        mel = _db_tf(_mel_tf(wav_t))
        mel_mean = mel.mean(dim=(2, 3), keepdim=True)
        mel_std = mel.std(dim=(2, 3), keepdim=True) + 1e-6
        mel = (mel - mel_mean) / mel_std
        mel_np = mel.numpy()

        # ★ V5: ASYNC 2-fold parallel inference
        # Start all folds asynchronously (they run in parallel on different threads)
        for _f, req in _infer_reqs.items():
            req.start_async({_INPUT_NAME: mel_np})

        # Wait for all folds and accumulate
        clip_sum = None; frame_sum = None
        for _f, req in _infer_reqs.items():
            req.wait()
            # ★ V5: .copy() needed because next start_async will overwrite buffer
            clip_ov = req.get_output_tensor(0).data.copy()
            frame_ov = req.get_output_tensor(1).data.copy()
            if clip_sum is None:
                clip_sum = clip_ov
                frame_sum = frame_ov
            else:
                clip_sum = clip_sum + clip_ov
                frame_sum = frame_sum + frame_ov
        n_folds = len(_ov_e106)
        clip_avg = clip_sum / n_folds
        frame_avg = frame_sum / n_folds

        for j in range(n_batch):
            fi = batch_start + j
            clip_file = clip_avg[j * N_WINDOWS:(j + 1) * N_WINDOWS]
            frame_file = frame_avg[j * N_WINDOWS:(j + 1) * N_WINDOWS]
            frame_max = frame_file.max(axis=1)

            p_clip = 1.0 / (1.0 + np.exp(-np.clip(clip_file, -50, 50)))
            p_frame = 1.0 / (1.0 + np.exp(-np.clip(frame_max, -50, 50)))
            p_mean = (0.5 * p_clip + 0.5 * p_frame).astype(np.float32)
            p_smooth = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
            probs_e17[fi] = p_smooth

        if (batch_end % 30 == 0) or batch_end == N_FILES:
            print(f"  e106-async-V5 [{batch_end}/{N_FILES}] {time.time()-t0:.0f}s")

probs_e17 = np.stack(probs_e17).astype(np.float32)
print(f"e106 OV V5 done: {probs_e17.shape} in {time.time()-t0:.0f}s (folds={list(_ov_e106.keys())}, batch={BATCH_FILES_E106}, async)")
"""


# ============================================================
# Apply patches
# ============================================================
patched = {"sed-infer": False, "e29-infer": False}
for c in nb["cells"]:
    cid = c.get("id")
    if cid == "sed-infer":
        c["source"] = SED_V5.splitlines(keepends=True)
        patched["sed-infer"] = True
    elif cid == "e29-infer":
        c["source"] = E29_V5.splitlines(keepends=True)
        patched["e29-infer"] = True

assert all(patched.values()), f"missing patches: {patched}"

V5_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {V5_NB} ({V5_NB.stat().st_size} bytes)")

# verify
nb_check = json.loads(V5_NB.read_text(encoding="utf-8"))
for c in nb_check["cells"]:
    cid = c.get("id")
    if cid == "sed-infer":
        s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        assert "BATCH_FILES_SED = 6" in s
        assert "sequential" in s
        print(f"  [OK] sed-infer V5: BATCH=6, sequential")
    elif cid == "e29-infer":
        s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        assert "BATCH_FILES_E106 = 12" in s
        assert "INFERENCE_NUM_THREADS" in s
        assert "create_infer_request" in s
        assert "start_async" in s
        print(f"  [OK] e29-infer V5: BATCH=12, ASYNC 2-fold, 2 threads/fold")
