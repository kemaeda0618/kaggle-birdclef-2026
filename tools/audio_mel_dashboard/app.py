"""BirdCLEF+ 2026 — Audio + Mel Spectrogram Viewer (Streamlit).

Browse train_audio clips and train_soundscapes, play audio, view mel spectrogram.
Filter by taxon / species / source. Shows labeled-SC ground-truth segments.

Run:
  cd tools/audio_mel_dashboard
  streamlit run app.py
or from repo root:
  streamlit run tools/audio_mel_dashboard/app.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

# ============================================================
# Config
# ============================================================
BC = Path(os.environ.get("BC2026_ROOT", "C:/Users/maeke/work/kaggle/birdclef-2026"))
TRAIN_AUDIO = BC / "train_audio"
TRAIN_SC = BC / "train_soundscapes"

SR = 32_000
N_FFT = 2048
HOP = 512
N_MELS = 256
FMIN = 20
FMAX = 16000
TOP_DB = 80

st.set_page_config(page_title="BC2026 Audio + Mel Viewer", page_icon="🔊", layout="wide")
st.title("🔊 BirdCLEF+ 2026 — Audio + Mel Spectrogram Viewer")


# ============================================================
# Cached metadata
# ============================================================
@st.cache_data
def load_meta():
    tax = pd.read_csv(BC / "taxonomy.csv")
    train = pd.read_csv(BC / "train.csv")
    train["primary_label"] = train["primary_label"].astype(str)
    label2taxon = dict(zip(tax["primary_label"].astype(str), tax["class_name"].astype(str)))
    label2name = dict(zip(tax["primary_label"].astype(str), tax["scientific_name"].astype(str)))
    # SC labels
    sc_labels = None
    p = BC / "train_soundscapes_labels.csv"
    if p.exists():
        sc_labels = pd.read_csv(p)
    return tax, train, label2taxon, label2name, sc_labels


@st.cache_data
def list_train_audio_files():
    # train_audio/<taxon_id>/<file>.ogg ; filename in train.csv is "<id>/<file>.ogg"
    return None  # we use train.csv filename column instead


tax, train_df, label2taxon, label2name, sc_labels = load_meta()


@st.cache_data
def load_audio(path_str, max_sec=60):
    wav, sr = sf.read(path_str, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    if len(wav) > SR * max_sec:
        wav = wav[: SR * max_sec]
    return wav


def compute_mel(wav):
    m = librosa.feature.melspectrogram(
        y=wav, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, power=2.0,
    )
    return librosa.power_to_db(m, top_db=TOP_DB)


def plot_mel(mel_db, title="", seg_marks=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    extent = [0, mel_db.shape[1] * HOP / SR, FMIN, FMAX]
    im = ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma", extent=extent)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel freq (Hz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, format="%+2.0f dB")
    if seg_marks:
        for t in seg_marks:
            ax.axvline(t, color="cyan", lw=0.8, alpha=0.6)
    return fig


# ============================================================
# Sidebar: source + filters
# ============================================================
st.sidebar.header("Source")
source = st.sidebar.radio("Dataset", ["train_audio (clips)", "train_soundscapes"])

if source == "train_audio (clips)":
    # filter by taxon / species
    train_df["taxon"] = train_df["primary_label"].map(label2taxon)
    taxa = ["(all)"] + sorted(train_df["taxon"].dropna().unique().tolist())
    sel_taxon = st.sidebar.selectbox("Taxon", taxa)
    df = train_df if sel_taxon == "(all)" else train_df[train_df["taxon"] == sel_taxon]

    # species with counts (sorted by count ascending so rare species are easy to find)
    sp_counts = df["primary_label"].value_counts()
    sp_index = sorted(sp_counts.index, key=lambda s: sp_counts[s])  # rare first
    sp_options = [f"{sp}  ({label2taxon.get(sp,'?')}, n={sp_counts[sp]})" for sp in sp_index]
    sp_pick = st.sidebar.selectbox("Species (rare first)", sp_options)
    sel_sp = sp_pick.split("  ")[0]
    sp_df = df[df["primary_label"] == sel_sp].reset_index(drop=True)

    st.sidebar.write(f"**{len(sp_df)} clips** for `{sel_sp}` ({label2name.get(sel_sp,'?')})")

    # ---- file picker: show all clips of this species with metadata ----
    def _clip_label(i, r):
        fn = Path(r["filename"]).name
        rt = r.get("rating", "?")
        tp = str(r.get("type", ""))[:24]
        col = r.get("collection", "?")
        return f"[{i}] {fn}  | rating={rt} | {col} | {tp}"

    clip_options = [_clip_label(i, sp_df.iloc[i]) for i in range(len(sp_df))]
    clip_pick = st.sidebar.selectbox(f"Clip ({len(sp_df)} files)", clip_options)
    fidx = int(clip_pick.split("]")[0].lstrip("["))
    row = sp_df.iloc[fidx]
    audio_path = TRAIN_AUDIO / row["filename"]
    title_info = f"{sel_sp} | {label2name.get(sel_sp,'?')} | {label2taxon.get(sel_sp,'?')} | clip {fidx+1}/{len(sp_df)}"
    seg_marks = None

    # show full clip table for this species
    st.sidebar.caption("All clips of this species:")
    _cols = [c for c in ["filename", "type", "rating", "author", "collection"] if c in sp_df.columns]
    st.sidebar.dataframe(sp_df[_cols], use_container_width=True, height=min(300, 40 + 30 * len(sp_df)))
else:
    sc_files = sorted(TRAIN_SC.glob("*.ogg"))
    # mark labeled files
    labeled_stems = set()
    if sc_labels is not None:
        labeled_stems = set(sc_labels["filename"].apply(lambda x: Path(str(x)).stem))
    only_labeled = st.sidebar.checkbox("Only labeled SC files", value=True)
    if only_labeled and labeled_stems:
        sc_files = [p for p in sc_files if p.stem in labeled_stems]
    st.sidebar.write(f"**{len(sc_files)} soundscape files**")
    if not sc_files:
        st.warning("no soundscape files found")
        st.stop()
    names = [p.name for p in sc_files]
    pick = st.sidebar.selectbox("File", names)
    audio_path = TRAIN_SC / pick
    title_info = pick
    # ground-truth segments for this file
    seg_marks = None
    if sc_labels is not None:
        sub = sc_labels[sc_labels["filename"].apply(lambda x: Path(str(x)).stem) == Path(pick).stem]
        if len(sub):
            def t2s(v):
                s = str(v).strip()
                if ":" in s:
                    pp = [float(x) for x in s.split(":")]
                    return pp[0]*3600+pp[1]*60+pp[2] if len(pp) == 3 else pp[0]*60+pp[1]
                return float(s)
            ecol = "end" if "end" in sub.columns else ("end_time" if "end_time" in sub.columns else None)
            if ecol:
                seg_marks = sorted(set(t2s(v) for v in sub[ecol]))


# ============================================================
# Main: audio + mel
# ============================================================
st.subheader(title_info)
st.caption(str(audio_path))

if not audio_path.exists():
    st.error(f"file not found: {audio_path}")
    st.stop()

max_sec = st.slider("max seconds to load", 5, 60, 60 if source == "train_soundscapes" else 30)
wav = load_audio(str(audio_path), max_sec=max_sec)
st.write(f"duration: {len(wav)/SR:.1f}s @ {SR} Hz")

# audio player (write to a temp wav in memory)
import io
buf = io.BytesIO()
sf.write(buf, wav, SR, format="WAV")
st.audio(buf.getvalue(), format="audio/wav")

# mel spectrogram
mel_db = compute_mel(wav)
fig = plot_mel(mel_db, title=f"Mel spectrogram — {title_info}", seg_marks=seg_marks)
st.pyplot(fig)
plt.close(fig)

# waveform
fig2, ax2 = plt.subplots(figsize=(12, 2))
t = np.arange(len(wav)) / SR
ax2.plot(t, wav, lw=0.4)
ax2.set_xlabel("time (s)"); ax2.set_ylabel("amp"); ax2.set_title("Waveform")
if seg_marks:
    for tm in seg_marks:
        ax2.axvline(tm, color="red", lw=0.6, alpha=0.5)
st.pyplot(fig2)
plt.close(fig2)

# SC ground-truth table
if source == "train_soundscapes" and sc_labels is not None:
    sub = sc_labels[sc_labels["filename"].apply(lambda x: Path(str(x)).stem) == Path(audio_path).stem]
    if len(sub):
        st.subheader("Labeled segments (ground truth)")
        st.dataframe(sub, use_container_width=True)

# spectrogram params
with st.expander("mel params"):
    st.write(f"n_fft={N_FFT}, hop={HOP}, n_mels={N_MELS}, fmin={FMIN}, fmax={FMAX}, top_db={TOP_DB}")
