import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Statistics", layout="wide")
st.title("Dataset Statistics")

ROOT      = Path(__file__).parent.parent.parent
TRAIN_CSV = ROOT / "train.csv"
TAX_CSV   = ROOT / "taxonomy.csv"

@st.cache_data
def load_data():
    train_df = pd.read_csv(TRAIN_CSV)
    tax_df   = pd.read_csv(TAX_CSV)
    return train_df, tax_df

train_df, tax_df = load_data()

# ── Overview ──────────────────────────────────────────────────
st.subheader("Overview")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("総ファイル数",   f"{len(train_df):,}")
c2.metric("種数（train）",  f"{train_df['primary_label'].nunique()}")
c3.metric("種数（taxonomy）", f"{len(tax_df)}")
c4.metric("クラス数",       f"{train_df['class_name'].nunique()}")
c5.metric("コレクション数", f"{train_df['collection'].nunique()}")

st.divider()

# ── クラス別ファイル数 ─────────────────────────────────────────
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

# ── 種ごとのファイル数分布 ────────────────────────────────────
st.subheader("種ごとのファイル数分布")
species_counts = train_df["primary_label"].value_counts()

col3, col4 = st.columns(2)
with col3:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(species_counts.values, bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel("ファイル数")
    ax.set_ylabel("種数")
    ax.set_title("1種あたりのファイル数ヒストグラム")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

with col4:
    st.markdown("**統計**")
    st.dataframe(pd.DataFrame({
        "指標": ["最小", "中央値", "平均", "最大"],
        "値":   [
            int(species_counts.min()),
            int(species_counts.median()),
            f"{species_counts.mean():.1f}",
            int(species_counts.max()),
        ]
    }), hide_index=True)

    st.markdown("**ファイル数 Top10 種**")
    top10 = species_counts.head(10).reset_index()
    top10.columns = ["primary_label", "count"]
    top10 = top10.merge(
        tax_df[["primary_label", "common_name", "class_name"]], on="primary_label", how="left"
    )
    st.dataframe(top10[["common_name", "class_name", "count"]], hide_index=True)

st.divider()

# ── Rating 分布 ───────────────────────────────────────────────
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

# ── 録音場所 ─────────────────────────────────────────────────
st.subheader("録音場所（緯度・経度）")
loc_df = train_df[["latitude", "longitude", "class_name"]].dropna()

fig, ax = plt.subplots(figsize=(7, 4))
colors = {"Aves": "steelblue", "Insecta": "orange", "Amphibia": "green", "Reptilia": "red"}
for cls, grp in loc_df.groupby("class_name"):
    ax.scatter(grp["longitude"], grp["latitude"],
               label=cls, alpha=0.3, s=5,
               color=colors.get(cls, "gray"))
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(markerscale=3)
ax.set_title("録音場所の分布")
plt.tight_layout()
st.pyplot(fig, use_container_width=False)
plt.close(fig)

st.divider()

# ── 種一覧テーブル ────────────────────────────────────────────
st.subheader("種一覧")
species_table = (
    species_counts.reset_index()
    .rename(columns={"primary_label": "primary_label", "count": "file_count"})
    .merge(tax_df[["primary_label", "common_name", "scientific_name", "class_name"]], on="primary_label", how="left")
    [["primary_label", "common_name", "scientific_name", "class_name", "file_count"]]
)
st.dataframe(species_table, hide_index=True, use_container_width=True)
