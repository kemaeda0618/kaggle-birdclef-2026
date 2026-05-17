"""Generate Tucker SED ONNX submission notebook."""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(cell_id, lines):
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "outputs": [], "execution_count": None, "source": src}


def md_cell(cell_id, lines):
    src = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src}


cells = []

cells.append(md_cell("hdr", [
    "# Tucker SED ONNX — Submission",
    "",
    "Tucker Arrants の蒸留済み SED (5-fold ONNX) 単独で提出。",
    "",
    "- ONNX: `bc2026-distilled-sed-public/sed_fold{0..4}.onnx`",
    "- Mel: 256 mels, n_fft=2048, hop=512, fmin=20, fmax=16000, SR=32000, top_db=80, per-sample z-norm",
    "- 5秒×12窓/ファイル、logit 空間で fold 平均 → Gaussian smooth → sigmoid",
]))

cells.append(code_cell("install", [
    "import subprocess, sys",
    "wheel_candidates = [",
    '    "/kaggle/input/datasets/tuckerarrants/perch-v2-no-dft-onnx/onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",',
    '    "/kaggle/input/perch-v2-no-dft-onnx/onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",',
    "]",
    "installed = False",
    "for whl in wheel_candidates:",
    "    try:",
    "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', whl])",
    '        print(f"Installed from: {whl}")',
    "        installed = True",
    "        break",
    "    except Exception:",
    "        pass",
    "if not installed:",
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'onnxruntime'])",
    '    print("Installed onnxruntime from PyPI")',
    "import onnxruntime as ort",
    'print(f"onnxruntime: {ort.__version__}")',
]))

cells.append(code_cell("imports", [
    "import os, time, glob, re",
    "import numpy as np",
    "import pandas as pd",
    "from pathlib import Path",
    "from scipy.ndimage import convolve1d",
    "import librosa",
    "WALL_START = time.time()",
    'print(f"librosa: {librosa.__version__}")',
]))

cells.append(code_cell("config", [
    "INF_SR       = 32_000",
    "INF_N_MELS   = 256",
    "INF_N_FFT    = 2048",
    "INF_HOP      = 512",
    "INF_FMIN     = 20",
    "INF_FMAX     = 16_000",
    "INF_TOP_DB   = 80",
    "INF_CHUNK_S  = 5",
    "INF_CHUNK_N  = INF_SR * INF_CHUNK_S",
    "INF_N_FRAMES = INF_CHUNK_N // INF_HOP + 1",
    "N_WINDOWS    = 12",
    "NUM_CLASSES  = 234",
    "GAUSSIAN_KERNEL = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)",
    'print(f"chunk={INF_CHUNK_S}s ({INF_CHUNK_N} samples), frames={INF_N_FRAMES}, windows={N_WINDOWS}")',
]))

cells.append(code_cell("paths", [
    "COMP_DIR = None",
    'for cand in [Path("/kaggle/input/competitions/birdclef-2026"), Path("/kaggle/input/birdclef-2026")]:',
    "    if cand.exists():",
    "        COMP_DIR = cand; break",
    'assert COMP_DIR, "birdclef-2026 not mounted"',
    "",
    "TEST_DIR       = COMP_DIR / 'test_soundscapes'",
    "SAMPLE_SUB_CSV = COMP_DIR / 'sample_submission.csv'",
    "",
    "SED_DIR = None",
    "for cand in [",
    '    "/kaggle/input/datasets/tuckerarrants/bc2026-distilled-sed-public",',
    '    "/kaggle/input/bc2026-distilled-sed-public",',
    "]:",
    "    if os.path.isdir(cand) and any(",
    '        f.endswith(".onnx") and "sed" in f for f in os.listdir(cand)',
    "    ):",
    "        SED_DIR = cand; break",
    'assert SED_DIR, "Tucker SED ONNX not found"',
    "",
    "sample_df = pd.read_csv(SAMPLE_SUB_CSV)",
    "PRIMARY_LABELS = sample_df.columns[1:].tolist()",
    "assert len(PRIMARY_LABELS) == NUM_CLASSES",
    "",
    "test_files = sorted(glob.glob(str(TEST_DIR / '*.ogg'))) if TEST_DIR.exists() else []",
    "if not test_files:",
    "    fallback = COMP_DIR / 'train_soundscapes'",
    "    if fallback.exists():",
    "        test_files = sorted(glob.glob(str(fallback / '*.ogg')))[:5]",
    '        print(f"No test_soundscapes — using {len(test_files)} train files for debug")',
    "",
    'print(f"COMP_DIR: {COMP_DIR}")',
    'print(f"SED_DIR:  {SED_DIR}")',
    'print(f"Test files: {len(test_files)}")',
    'print(f"Species: {len(PRIMARY_LABELS)}")',
]))

cells.append(code_cell("load-onnx", [
    "def make_session(onnx_path):",
    "    so = ort.SessionOptions()",
    "    so.intra_op_num_threads = os.cpu_count() or 4",
    "    so.inter_op_num_threads = 1",
    "    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL",
    "    return ort.InferenceSession(str(onnx_path), sess_options=so,",
    '                                providers=["CPUExecutionProvider"])',
    "",
    'pat = re.compile(r"sed_fold(\\d+)\\.onnx$")',
    "fold_files = sorted(",
    "    (f for f in os.listdir(SED_DIR) if pat.match(f)),",
    "    key=lambda f: int(pat.match(f).group(1))",
    ")",
    'assert fold_files, f"No sed_fold*.onnx in {SED_DIR}"',
    "",
    "fold_sessions = []",
    "for fname in fold_files:",
    "    p = os.path.join(SED_DIR, fname)",
    "    sess = make_session(p)",
    "    fold_sessions.append(sess)",
    "    size_mb = os.path.getsize(p) / 1e6",
    '    inp_names = [x.name for x in sess.get_inputs()]',
    '    print(f"  {fname}: {size_mb:.1f}MB  inputs={inp_names}")',
    "",
    'print(f"\\nLoaded {len(fold_sessions)} fold(s)")',
]))

cells.append(code_cell("mel-fn", [
    "def audio_to_mel(chunks):",
    "    # chunks: (N, 160000) float32 -> mel dB (N, 1, 256, 313), z-norm per chunk",
    "    mels = []",
    "    for i in range(chunks.shape[0]):",
    "        S = librosa.feature.melspectrogram(",
    "            y=chunks[i], sr=INF_SR, n_fft=INF_N_FFT, hop_length=INF_HOP,",
    "            n_mels=INF_N_MELS, fmin=INF_FMIN, fmax=INF_FMAX, power=2.0,",
    "        )",
    "        S_dB = librosa.power_to_db(S, top_db=INF_TOP_DB)",
    "        S_dB = (S_dB - S_dB.mean()) / (S_dB.std() + 1e-6)",
    "        mels.append(S_dB)",
    "    return np.stack(mels)[:, np.newaxis, :, :].astype(np.float32)",
    "",
    "def load_audio(path):",
    "    try:",
    "        import soundfile as sf",
    "        wav, sr = sf.read(str(path), dtype='float32', always_2d=False)",
    "        if wav.ndim > 1: wav = wav.mean(axis=1)",
    "        if sr != INF_SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=INF_SR)",
    "    except Exception:",
    "        wav, _ = librosa.load(str(path), sr=INF_SR, mono=True)",
    "    return wav.astype(np.float32)",
    "",
    "def file_to_chunks(path):",
    "    wav = load_audio(path)",
    "    target_len = 60 * INF_SR",
    "    if len(wav) < target_len:",
    "        wav = np.pad(wav, (0, target_len - len(wav)))",
    "    else:",
    "        wav = wav[:target_len]",
    "    chunks = wav[:N_WINDOWS * INF_CHUNK_N].reshape(N_WINDOWS, INF_CHUNK_N)",
    "    end_times = np.arange(1, N_WINDOWS + 1) * INF_CHUNK_S",
    "    return chunks.astype(np.float32), end_times",
    "",
    "def sigmoid_safe(x):",
    "    return np.where(",
    "        x >= 0,",
    "        1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))),",
    "        np.exp(np.clip(x, -50, 50)) / (1.0 + np.exp(np.clip(x, -50, 50))),",
    "    ).astype(np.float32)",
    "",
    "def gauss_smooth(logits):",
    "    # logits: (N_WINDOWS, 234) -> smoothed",
    "    return convolve1d(logits, GAUSSIAN_KERNEL, axis=0, mode='nearest')",
    "",
    "# smoke test",
    "if test_files:",
    "    _t0 = time.time()",
    "    _c, _et = file_to_chunks(test_files[0])",
    "    _mel = audio_to_mel(_c)",
    '    print(f"Smoke: chunks={_c.shape}, mel={_mel.shape}, t={time.time()-_t0:.2f}s")',
    "    _outs = fold_sessions[0].run(None, {'mel': _mel})",
    '    print(f"ONNX out[0] clip={_outs[0].shape}, out[1] frame={_outs[1].shape}")',
]))

cells.append(code_cell("inference", [
    "t0 = time.time()",
    "all_rows, all_preds = [], []",
    "",
    "for file_idx, file_path in enumerate(test_files):",
    "    basename = Path(file_path).stem",
    "    chunks, end_times = file_to_chunks(file_path)",
    "    mel = audio_to_mel(chunks)",
    "",
    "    logits_sum = np.zeros((N_WINDOWS, NUM_CLASSES), dtype=np.float32)",
    "    for sess in fold_sessions:",
    "        outs = sess.run(None, {'mel': mel})",
    "        clip_logits = outs[0]",
    "        frame_max   = outs[1].max(axis=1)",
    "        logits_sum += 0.5 * clip_logits + 0.5 * frame_max",
    "    logits_mean = logits_sum / len(fold_sessions)",
    "",
    "    logits_sm = gauss_smooth(logits_mean)",
    "    probs = sigmoid_safe(logits_sm)",
    "",
    "    all_rows.extend([f'{basename}_{int(t)}' for t in end_times])",
    "    all_preds.append(probs)",
    "",
    "    if (file_idx + 1) % 50 == 0 or file_idx == 0 or file_idx == len(test_files) - 1:",
    "        elapsed = time.time() - t0",
    "        rate = (file_idx + 1) / elapsed",
    "        eta = (len(test_files) - file_idx - 1) / rate if rate > 0 else 0",
    '        print(f"  [{file_idx+1:4d}/{len(test_files)}] {elapsed:.1f}s  {rate:.2f} files/s  ETA {eta:.0f}s")',
    "",
    "all_preds_arr = np.concatenate(all_preds) if all_preds else np.zeros((0, NUM_CLASSES), np.float32)",
    'print(f"\\nInference done: {len(all_rows)} rows, {time.time()-t0:.1f}s total")',
]))

cells.append(code_cell("submission", [
    "SUB_PATH = Path('/kaggle/working/submission.csv')",
    "if not test_files:",
    "    pd.read_csv(SAMPLE_SUB_CSV).to_csv(SUB_PATH, index=False)",
    '    print(f"No test files — wrote sample_submission to {SUB_PATH}")',
    "else:",
    "    sub_df = pd.DataFrame(all_preds_arr.clip(0.0, 1.0), columns=PRIMARY_LABELS)",
    "    sub_df.insert(0, 'row_id', all_rows)",
    "    assert sub_df.shape[1] == NUM_CLASSES + 1",
    "    assert sub_df['row_id'].is_unique",
    "    assert not sub_df.iloc[:, 1:].isna().any().any()",
    "    sub_df.to_csv(SUB_PATH, index=False)",
    '    print(f"Wrote {len(sub_df)} rows to {SUB_PATH}")',
    "    print(sub_df.head(3).iloc[:, :6])",
    "",
    'print(f"\\nWall time: {(time.time()-WALL_START)/60:.1f} min")',
]))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out_path = HERE / "nb_tucker_submit.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
