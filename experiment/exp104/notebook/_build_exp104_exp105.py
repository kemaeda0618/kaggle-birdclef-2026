"""Build exp104/exp105 inference NBs by cloning exp090 + swapping 2 cells:
  - sed-infer  : Tucker SED 5-fold ONNX  ->  OpenVINO (1.38x speedup)
  - e29-infer  : exp029 R3 PyTorch       ->  exp102 OV (3-fold ensemble or fold 1 single)

Output: probs_sed and probs_e17 in same shapes as exp090 -> blend cell unchanged.
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

EXP090_NB = Path("experiment/exp090/notebook/nb_exp090_tuned_tax.ipynb")
EXP104_NB = Path("experiment/exp104/notebook/nb_exp104_e29_swap_3fold.ipynb")
EXP105_NB = Path("experiment/exp105/notebook/nb_exp105_e29_swap_fold1.ipynb")
EXP104_NB.parent.mkdir(parents=True, exist_ok=True)
EXP105_NB.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# New sed-infer cell: Tucker SED 5-fold OpenVINO (replaces ONNX)
# Output: probs_sed (N_files, 12, N_CLASSES) — same as before
# ============================================================
SED_OV_CELL = r"""# === Stage B: Tucker 5-fold SED OpenVINO ensemble (1.38x faster than ONNX) ===
import librosa, glob
from scipy.ndimage import gaussian_filter1d

# Install openvino offline (idempotent — both sed-infer and e29-infer call this)
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


def audio_to_mel(chunks):
    mels = []
    for x in chunks:
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
assert len(sed_xml_paths) == 5, f"expected 5 folds, got {len(sed_xml_paths)}"

_ov_core_sed = ov.Core()
sed_compiled = [_ov_core_sed.compile_model(str(p), "CPU") for p in sed_xml_paths]
print(f"compiled {len(sed_compiled)} OV models")

t0 = time.time()
probs_sed = []   # (N_files, 12, 234)
for fi, raw_60s in enumerate(audio_cache):
    chunks = raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES)
    mel = audio_to_mel(chunks)
    p_sum = np.zeros((N_WINDOWS, N_CLASSES), dtype=np.float32)
    for compiled in sed_compiled:
        ov_outs = compiled(mel)
        clip_logits = ov_outs[compiled.outputs[0]]
        frame_max = ov_outs[compiled.outputs[1]].max(axis=1)
        p_sum += 0.5 * sigmoid_sed(clip_logits) + 0.5 * sigmoid_sed(frame_max)
    p_mean = p_sum / len(sed_compiled)
    p_mean = gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
    probs_sed.append(p_mean)
    if (fi + 1) % 10 == 0 or fi == len(audio_cache) - 1:
        elapsed = time.time() - t0
        print(f"  SED-OV [{fi+1}/{len(audio_cache)}] {elapsed:.0f}s")

probs_sed = np.stack(probs_sed)
print(f"SED OV done: {probs_sed.shape} in {time.time()-t0:.0f}s")
"""


# ============================================================
# New e29-infer cell: exp102 OV inference (3-fold or fold 1)
# Output: probs_e17 (N_files, 12, N_CLASSES) — replaces exp029 R3 output
# ============================================================
def make_exp102_ov_cell(folds_to_use):
    folds_repr = folds_to_use
    folds_label = ("3-fold ensemble" if len(folds_to_use) == 3
                   else f"fold {folds_to_use[0]} single")
    return f"""# === exp102 OV inference ({folds_label}) — replaces exp029 R3 PyTorch ===
import torchaudio, glob

# openvino install (idempotent — sed-infer already imported, just verify)
try:
    import openvino as ov
    print(f"openvino: {{ov.__version__}}")
except ImportError:
    _hits = sorted(glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True))
    assert _hits, "openvino wheel not found"
    _WHEEL_DIR = str(Path(_hits[0]).parent)
    !pip install -q --no-deps {{_WHEEL_DIR}}/openvino-*.whl {{_WHEEL_DIR}}/openvino_telemetry-*.whl
    import openvino as ov

# ---- Locate exp102 OV IR files ----
EXP102_DIR = None
for _p in [
    Path("/kaggle/input/birdclef2026-exp102-3fold-ov"),
    Path("/kaggle/input/datasets/maekeso/birdclef2026-exp102-3fold-ov"),
]:
    if _p.exists():
        EXP102_DIR = _p; break
if EXP102_DIR is None:
    _hits = glob.glob("/kaggle/input/**/exp102_fold*.xml", recursive=True)
    if _hits:
        EXP102_DIR = Path(_hits[0]).parent
assert EXP102_DIR is not None, "exp102 OV IR not found — attach kernel_sources: maekeso/birdclef2026-exp102-3fold-ov"
print(f"[ov] EXP102_DIR: {{EXP102_DIR}}")

FOLDS_E102 = {folds_repr}
EXP102_IR = {{}}
for _f in FOLDS_E102:
    _xml_hits = list(EXP102_DIR.rglob(f"exp102_fold{{_f}}.xml"))
    assert _xml_hits, f"exp102_fold{{_f}}.xml not found in {{EXP102_DIR}}"
    EXP102_IR[_f] = _xml_hits[0]
    print(f"  fold {{_f}}: {{EXP102_IR[_f]}}")

_core_e102 = ov.Core()
_ov_e102 = {{}}
for _f, _xml in EXP102_IR.items():
    _ov_e102[_f] = _core_e102.compile_model(str(_xml), "CPU")
print(f"[ov] compiled {{len(_ov_e102)}} fold(s)")

# ---- Mel transform (same config as exp029 R3 training) ----
_mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=2048, hop_length=512,
    n_mels=256, f_min=20, f_max=16000, power=2.0,
)
_db_tf = torchaudio.transforms.AmplitudeToDB(top_db=80)

# ---- Inference loop ----
t0 = time.time()
probs_e17 = []  # (N_files, 12, N_CLASSES)
with torch.no_grad():
    for _fi, _raw_60s in enumerate(audio_cache):
        _chunks = _raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
        _wav_t = torch.from_numpy(_chunks).unsqueeze(1)  # (12, 1, 160000)
        _mel = _db_tf(_mel_tf(_wav_t))
        # per-instance standardize (CRITICAL: matches exp029 R3 train/infer)
        for _i in range(_mel.size(0)):
            _mel[_i] = (_mel[_i] - _mel[_i].mean()) / (_mel[_i].std() + 1e-6)
        _mel_np = _mel.numpy()

        # Run each fold, accumulate
        _clip_sum = None; _frame_sum = None
        for _f, _compiled in _ov_e102.items():
            _ov_out = _compiled(_mel_np)
            _clip_ov = _ov_out[_compiled.outputs[0]]
            _frame_ov = _ov_out[_compiled.outputs[1]]
            if _clip_sum is None:
                _clip_sum = _clip_ov.copy()
                _frame_sum = _frame_ov.copy()
            else:
                _clip_sum += _clip_ov
                _frame_sum += _frame_ov
        _n_folds = len(_ov_e102)
        _clip = _clip_sum / _n_folds
        _frame = _frame_sum / _n_folds
        _frame_max = _frame.max(axis=1)

        _p_clip = 1.0 / (1.0 + np.exp(-_clip))
        _p_frame = 1.0 / (1.0 + np.exp(-_frame_max))
        _p_mean = (0.5 * _p_clip + 0.5 * _p_frame).astype(np.float32)
        _p_smooth = gaussian_filter1d(_p_mean, sigma=0.65, axis=0, mode="nearest").astype(np.float32)
        probs_e17.append(_p_smooth)
        if (_fi + 1) % 10 == 0 or _fi == len(audio_cache) - 1:
            print(f"  e102-ov [{{_fi+1}}/{{len(audio_cache)}}] {{time.time()-t0:.0f}}s")

probs_e17 = np.stack(probs_e17).astype(np.float32)
print(f"e102-ov done: {{probs_e17.shape}} in {{time.time()-t0:.0f}}s  (folds={{list(_ov_e102.keys())}})")
"""


def build_nb(out_path: Path, folds_to_use):
    nb = json.loads(EXP090_NB.read_text(encoding="utf-8"))

    swaps = {
        "sed-infer": SED_OV_CELL,
        "e29-infer": make_exp102_ov_cell(folds_to_use),
    }
    found = {k: False for k in swaps}
    for c in nb["cells"]:
        cid = c.get("id")
        if cid in swaps:
            c["source"] = swaps[cid].splitlines(keepends=True)
            found[cid] = True
    for k, ok in found.items():
        assert ok, f"cell id={k} not found in exp090"

    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")

    # compile check both swapped cells
    for cid, new_src in swaps.items():
        clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                           for ln in new_src.splitlines())
        try:
            compile(clean, f"<{cid}>", "exec")
            print(f"  [OK] {cid} compiles")
        except SyntaxError as e:
            print(f"  [FAIL] {cid}: {e}")


if __name__ == "__main__":
    build_nb(EXP104_NB, [0, 1, 2])
    build_nb(EXP105_NB, [1])
