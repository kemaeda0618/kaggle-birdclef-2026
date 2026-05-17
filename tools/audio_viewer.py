import io
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from pathlib import Path

# 日本語フォント設定（Windowsの游ゴシック → メイリオ → MSゴシックの順で試す）
_jp_fonts = ["Yu Gothic", "Meiryo", "MS Gothic", "IPAexGothic", "Noto Sans CJK JP"]
_available = {f.name for f in fm.fontManager.ttflist}
_font = next((f for f in _jp_fonts if f in _available), None)
if _font:
    plt.rcParams["font.family"] = _font

st.set_page_config(page_title="BirdCLEF Audio Viewer", layout="wide")

ROOT       = Path(__file__).parent.parent
AUDIO_ROOT = ROOT / "train_audio"
SC_ROOT    = ROOT / "train_soundscapes"
TRAIN_CSV  = ROOT / "train.csv"
TAX_CSV    = ROOT / "taxonomy.csv"

@st.cache_data
def load_train_df():
    return pd.read_csv(TRAIN_CSV)

@st.cache_data
def load_tax_df():
    return pd.read_csv(TAX_CSV)

@st.cache_data
def load_audio(path_str):
    return librosa.load(path_str, sr=32000, mono=True)

train_df = load_train_df()
tax_df   = load_tax_df()

# ── サイドバーでページ切替 ────────────────────────────────────
page = st.sidebar.radio("ページ", ["Audio Viewer", "Statistics", "Species AUC Report", "Audio Similarity"])

# ════════════════════════════════════════════════════════════════
# Audio Viewer
# ════════════════════════════════════════════════════════════════
if page == "Audio Viewer":
    st.title("Audio Viewer")

    ogg_files = sorted(AUDIO_ROOT.rglob("*.ogg"))
    if not ogg_files:
        st.error(f"音声ファイルが見つかりません: {AUDIO_ROOT}")
        st.stop()

    all_file_labels = [str(f.relative_to(AUDIO_ROOT)).replace("\\", "/") for f in ogg_files]

    # ── フィルター ────────────────────────────────────────────────
    with st.expander("フィルター", expanded=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            all_classes = sorted(train_df["class_name"].dropna().unique().tolist())
            sel_classes = st.multiselect("Class", all_classes, default=[])
        with f2:
            search_common = st.text_input("Common name (部分一致)")
        with f3:
            search_sci = st.text_input("Scientific name (部分一致)")

    # フィルター適用
    filtered_meta = train_df.copy()
    if sel_classes:
        filtered_meta = filtered_meta[filtered_meta["class_name"].isin(sel_classes)]
    if search_common:
        filtered_meta = filtered_meta[filtered_meta["common_name"].str.contains(search_common, case=False, na=False)]
    if search_sci:
        filtered_meta = filtered_meta[filtered_meta["scientific_name"].str.contains(search_sci, case=False, na=False)]

    matched_filenames = set(filtered_meta["filename"].tolist())
    if sel_classes or search_common or search_sci:
        file_labels = [f for f in all_file_labels if f in matched_filenames]
        st.caption(f"{len(file_labels):,} / {len(all_file_labels):,} ファイル")
    else:
        file_labels = all_file_labels

    if not file_labels:
        st.warning("条件に一致するファイルがありません。")
        st.stop()

    mode   = st.radio("モード", ["1ファイル", "複数比較"], horizontal=True)
    n_files = st.slider("比較ファイル数", 2, 4, 2) if mode == "複数比較" else 1

    selected_labels = []
    for i in range(n_files):
        label = st.selectbox(f"ファイル {i+1}", file_labels, key=f"file_{i}")
        selected_labels.append(label)

    st.divider()
    cols = st.columns(n_files)

    for col, selected_label in zip(cols, selected_labels):
        selected_path = AUDIO_ROOT / Path(selected_label)
        filename_key  = selected_label.replace("\\", "/")
        row = train_df[train_df["filename"] == filename_key]

        with col:
            if not row.empty:
                r = row.iloc[0]
                st.markdown(f"### {r.get('common_name', selected_label)}")
                st.markdown(f"*{r.get('scientific_name', '-')}*")
                st.markdown(f"**Class**: {r.get('class_name', '-')}　**Rating**: {r.get('rating', '-')}　**Collection**: {r.get('collection', '-')}")
                lat, lon = r.get('latitude'), r.get('longitude')
                if pd.notna(lat) and pd.notna(lon):
                    st.markdown(f"**Location**: {lat:.4f}, {lon:.4f}")
                st.markdown(f"**Secondary**: {r.get('secondary_labels', '[]')}")
            else:
                st.markdown(f"### {selected_label}")

            with open(selected_path, "rb") as f:
                st.audio(f.read(), format="audio/ogg")

            y, sr = load_audio(str(selected_path))
            duration = len(y) / sr

            c1, c2 = st.columns(2)
            c1.metric("長さ", f"{duration:.1f} s")
            c2.metric("RMS", f"{librosa.feature.rms(y=y).mean():.4f}")

            fig_wave, ax_wave = plt.subplots(figsize=(5, 1.5))
            ax_wave.plot(np.linspace(0, duration, len(y)), y, linewidth=0.3, color="steelblue")
            ax_wave.set_xlabel("Time (s)")
            ax_wave.set_xlim(0, duration)
            st.pyplot(fig_wave, use_container_width=False)
            plt.close(fig_wave)

            mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=320, n_mels=128, fmin=20, fmax=16000)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            fig_mel, ax_mel = plt.subplots(figsize=(5, 2.5))
            img = librosa.display.specshow(mel_db, sr=sr, hop_length=320, x_axis="time", y_axis="mel", fmin=20, fmax=16000, ax=ax_mel)
            fig_mel.colorbar(img, ax=ax_mel, format="%+2.0f dB")
            st.pyplot(fig_mel, use_container_width=False)
            plt.close(fig_mel)

# ════════════════════════════════════════════════════════════════
# Statistics
# ════════════════════════════════════════════════════════════════
elif page == "Statistics":
    st.title("Dataset Statistics")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("総ファイル数",     f"{len(train_df):,}")
    c2.metric("種数（train）",    f"{train_df['primary_label'].nunique()}")
    c3.metric("種数（taxonomy）", f"{len(tax_df)}")
    c4.metric("クラス数",         f"{train_df['class_name'].nunique()}")
    c5.metric("コレクション数",   f"{train_df['collection'].nunique()}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("クラス別ファイル数")
        class_counts = train_df["class_name"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(class_counts.index, class_counts.values, color="steelblue")
        ax.set_xlabel("ファイル数")
        for i, v in enumerate(class_counts.values):
            ax.text(v + 50, i, str(v), va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with col2:
        st.subheader("コレクション別ファイル数")
        col_counts = train_df["collection"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.pie(col_counts.values, labels=col_counts.index, autopct="%1.1f%%", startangle=90)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    st.divider()

    st.subheader("種ごとのファイル数分布")
    species_counts = train_df["primary_label"].value_counts()
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(species_counts.values, bins=30, color="steelblue", edgecolor="white")
        ax.set_xlabel("ファイル数")
        ax.set_ylabel("種数")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with col4:
        st.markdown("**統計**")
        st.dataframe(pd.DataFrame({
            "指標": ["最小", "中央値", "平均", "最大"],
            "値":   [int(species_counts.min()), int(species_counts.median()),
                     f"{species_counts.mean():.1f}", int(species_counts.max())],
        }), hide_index=True)
        st.markdown("**ファイル数 Top10**")
        top10 = species_counts.head(10).reset_index()
        top10.columns = ["primary_label", "count"]
        top10 = top10.merge(tax_df[["primary_label", "common_name", "class_name"]], on="primary_label", how="left")
        st.dataframe(top10[["common_name", "class_name", "count"]], hide_index=True)

    st.divider()

    st.subheader("Rating 分布")
    col5, col6 = st.columns(2)
    with col5:
        fig, ax = plt.subplots(figsize=(5, 3))
        rating_counts = train_df["rating"].value_counts().sort_index()
        ax.bar(rating_counts.index.astype(str), rating_counts.values, color="steelblue")
        ax.set_xlabel("Rating")
        ax.set_ylabel("ファイル数")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with col6:
        st.markdown("**クラス別の平均Rating**")
        rating_by_class = train_df.groupby("class_name")["rating"].mean().round(2).reset_index()
        rating_by_class.columns = ["class_name", "平均Rating"]
        st.dataframe(rating_by_class, hide_index=True)

    st.divider()

    st.subheader("録音場所")
    import plotly.express as px
    loc_df = train_df[["latitude", "longitude", "class_name", "common_name", "primary_label"]].dropna()
    fig_map = px.scatter_mapbox(
        loc_df,
        lat="latitude",
        lon="longitude",
        color="class_name",
        hover_name="common_name",
        hover_data={"primary_label": True, "latitude": ":.4f", "longitude": ":.4f"},
        zoom=3,
        height=1000,
        mapbox_style="open-street-map",
        opacity=0.6,
        title="録音場所の分布（クラス別）",
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()

    st.subheader("種一覧")
    species_table = (
        species_counts.reset_index()
        .rename(columns={"primary_label": "primary_label", "count": "file_count"})
        .merge(tax_df[["primary_label", "common_name", "scientific_name", "class_name"]], on="primary_label", how="left")
        [["primary_label", "common_name", "scientific_name", "class_name", "file_count"]]
    )
    st.dataframe(species_table, hide_index=True, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# Species AUC Report
# ════════════════════════════════════════════════════════════════
elif page == "Species AUC Report":
    st.title("Species AUC Report (OOF)")

    AUC_CSV = ROOT / "experiment" / "exp003" / "notebook" / "species_auc.csv"

    if not AUC_CSV.exists():
        st.warning("species_auc.csv が見つかりません。")
        st.info("""
**手順:**
1. `experiment/exp003/notebook/species_auc.ipynb` を Kaggle にアップロードして実行
2. Output の `species_auc.csv` をダウンロード
3. `experiment/exp003/notebook/species_auc.csv` に配置
        """)
        st.stop()

    df = pd.read_csv(AUC_CSV)
    df_active = df[df["n_positive"] > 0].copy()

    # ── Overview ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("評価対象種数", len(df_active))
    c2.metric("全体 AUC (base)",  f"{df_active['auc_base'].mean():.4f}")
    c3.metric("全体 AUC (final)", f"{df_active['auc_final'].mean():.4f}")
    c4.metric("Probe改善種数", int((df_active['auc_final'] > df_active['auc_base']).sum()))

    st.divider()

    # ── フィルター ─────────────────────────────────────────────
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        class_filter = st.multiselect("クラスで絞り込み", df_active["class_name"].unique().tolist(), default=df_active["class_name"].unique().tolist())
    with col_f2:
        mapped_filter = st.radio("Perchマッピング", ["全て", "マッピング済みのみ", "未マッピングのみ"], horizontal=True)

    filtered = df_active[df_active["class_name"].isin(class_filter)]
    if mapped_filter == "マッピング済みのみ":
        filtered = filtered[filtered["perch_mapped"]]
    elif mapped_filter == "未マッピングのみ":
        filtered = filtered[~filtered["perch_mapped"]]

    sort_col = st.radio("並び順", ["auc_final (高い順)", "auc_final (低い順)", "n_positive (多い順)"], horizontal=True)
    if sort_col == "auc_final (高い順)":
        filtered = filtered.sort_values("auc_final", ascending=False)
    elif sort_col == "auc_final (低い順)":
        filtered = filtered.sort_values("auc_final", ascending=True)
    else:
        filtered = filtered.sort_values("n_positive", ascending=False)

    st.divider()

    # ── AUC 分布ヒストグラム ───────────────────────────────────
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.subheader("AUC分布 (final)")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df_active["auc_final"].dropna(), bins=20, color="steelblue", edgecolor="white")
        ax.axvline(df_active["auc_final"].mean(), color="red", linestyle="--", label=f"平均: {df_active['auc_final'].mean():.3f}")
        ax.set_xlabel("AUC")
        ax.set_ylabel("種数")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with col_h2:
        st.subheader("クラス別 平均AUC")
        class_auc = df_active.groupby("class_name")[["auc_base","auc_final"]].mean().round(4)
        st.dataframe(class_auc)

    st.divider()

    # ── AUCバーチャート（上位・下位）─────────────────────────
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.subheader("AUC Top 20")
        top20 = filtered.nlargest(20, "auc_final")
        fig, ax = plt.subplots(figsize=(5, 6))
        colors = ["steelblue" if m else "orange" for m in top20["perch_mapped"]]
        ax.barh(top20["common_name"], top20["auc_final"], color=colors)
        ax.set_xlabel("AUC (final)")
        ax.invert_yaxis()
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    with col_b2:
        st.subheader("AUC Bottom 20")
        bot20 = filtered.nsmallest(20, "auc_final")
        fig, ax = plt.subplots(figsize=(5, 6))
        colors = ["steelblue" if m else "orange" for m in bot20["perch_mapped"]]
        ax.barh(bot20["common_name"], bot20["auc_final"], color=colors)
        ax.set_xlabel("AUC (final)")
        ax.invert_yaxis()
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    st.caption("青: Perchマッピング済み　オレンジ: 未マッピング")

    st.divider()

    # ── 全データテーブル ──────────────────────────────────────
    st.subheader("全種テーブル")
    show_cols = ["common_name", "scientific_name", "class_name", "n_positive", "perch_mapped", "auc_base", "auc_final"]
    st.dataframe(
        filtered[show_cols].style.background_gradient(subset=["auc_final"], cmap="RdYlGn"),
        hide_index=True, use_container_width=True
    )

# ════════════════════════════════════════════════════════════════
# Audio Similarity
# ════════════════════════════════════════════════════════════════
elif page == "Audio Similarity":
    st.title("Audio Similarity: Train Soundscapes ↔ Train Audio")

    SIM_CSV = ROOT / "experiment" / "exp003" / "notebook" / "audio_similarity.csv"

    if not SIM_CSV.exists():
        st.warning("audio_similarity.csv が見つかりません。")
        st.info("""
**手順:**
1. `experiment/exp003/notebook/audio_similarity.ipynb` を Kaggle にアップロードして実行
2. Output の `audio_similarity.csv` をダウンロード
3. `experiment/exp003/notebook/audio_similarity.csv` に配置
        """)
        st.stop()

    @st.cache_data
    def load_sim_df():
        return pd.read_csv(SIM_CSV)

    sim_df = load_sim_df()

    # ── Soundscape ファイル選択 ───────────────────────────────────
    sc_files = sorted(sim_df["soundscape_file"].unique())
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        sel_sc = st.selectbox("Soundscape ファイル", sc_files)
    sc_df = sim_df[sim_df["soundscape_file"] == sel_sc].copy()

    # ── セグメント選択 ────────────────────────────────────────────
    segments = sorted(sc_df["segment_start_sec"].unique())
    with col_sel2:
        sel_seg = st.selectbox(
            "セグメント (開始秒)",
            segments,
            format_func=lambda s: f"{int(s)}s – {int(s)+5}s",
        )

    seg_df = sc_df[sc_df["segment_start_sec"] == sel_seg].sort_values("rank")

    st.divider()

    # ── Soundscape セグメント再生 ─────────────────────────────────
    sc_path = SC_ROOT / sel_sc
    st.subheader(f"Soundscape: {sel_sc}  [{int(sel_seg)}s – {int(sel_seg)+5}s]")

    if sc_path.exists():
        col_sc_play, col_sc_mel = st.columns(2)
        with col_sc_play:
            y_seg, sr_seg = librosa.load(str(sc_path), sr=32000, offset=float(sel_seg), duration=5.0)
            buf = io.BytesIO()
            wavfile.write(buf, sr_seg, (y_seg * 32767).astype(np.int16))
            buf.seek(0)
            st.audio(buf, format="audio/wav")
        with col_sc_mel:
            mel = librosa.feature.melspectrogram(y=y_seg, sr=sr_seg, n_fft=1024, hop_length=320, n_mels=128, fmin=20, fmax=16000)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            fig, ax = plt.subplots(figsize=(4, 2))
            librosa.display.specshow(mel_db, sr=sr_seg, hop_length=320, x_axis="time", y_axis="mel", ax=ax)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
    else:
        st.info(f"Soundscape ファイルがローカルに存在しません: {sc_path}")

    st.divider()

    # ── 類似 Train Audio Top-K ────────────────────────────────────
    st.subheader(f"類似 Train Audio Top-{len(seg_df)}")

    for _, row in seg_df.iterrows():
        rank   = int(row["rank"])
        label  = f"#{rank}  {row['common_name']}  ({row['class_name']})  — cosine sim: {row['cosine_sim']:.4f}"
        with st.expander(label, expanded=(rank == 1)):
            st.markdown(f"*{row['scientific_name']}*　`{row['train_filename']}`")

            train_path = AUDIO_ROOT / Path(row["train_filename"])
            col_t_play, col_t_mel = st.columns(2)

            if train_path.exists():
                with col_t_play:
                    with open(train_path, "rb") as f:
                        st.audio(f.read(), format="audio/ogg")
                with col_t_mel:
                    y_t, sr_t = load_audio(str(train_path))
                    mel = librosa.feature.melspectrogram(y=y_t[:sr_t*5], sr=sr_t, n_fft=1024, hop_length=320, n_mels=128, fmin=20, fmax=16000)
                    mel_db = librosa.power_to_db(mel, ref=np.max)
                    fig, ax = plt.subplots(figsize=(4, 2))
                    librosa.display.specshow(mel_db, sr=sr_t, hop_length=320, x_axis="time", y_axis="mel", ax=ax)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=False)
                    plt.close(fig)
            else:
                st.warning(f"ファイルが見つかりません: {train_path}")
