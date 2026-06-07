"""Phase 0: pseudo_e17.csv の max(prob) 分布を分析、threshold の妥当性を検証。

1. Kaggle NB output から pseudo_e17.csv を DL
2. max(prob) per chunk の分布計算
3. 各 threshold での残存率
4. 削減後の class カバレッジ
5. 推奨 threshold 算出
"""
import json, os, sys, io
import tempfile
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if not os.environ.get("KAGGLE_API_TOKEN"):
    _kgat = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    os.environ["KAGGLE_API_TOKEN"] = _kgat

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

# Download pseudo_e17.csv
PSEUDO_LOCAL = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp045\notebook\_pseudo_e17_cache.csv")
if not PSEUDO_LOCAL.exists():
    print("DL pseudo_e17.csv from Kaggle NB output (full download)...")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        api.kernels_output("maekeso/birdclef2026-exp028-pseudo-e17", path=str(td), force=True)
        src = next(td.rglob("pseudo_e17.csv"), None)
        assert src is not None and src.exists(), "pseudo_e17.csv not found in NB output"
        print(f"  found at {src}")
        import shutil
        shutil.copy2(str(src), str(PSEUDO_LOCAL))
    print(f"  DL OK: {PSEUDO_LOCAL.stat().st_size/1e6:.1f} MB")
else:
    print(f"Using cached: {PSEUDO_LOCAL.stat().st_size/1e6:.1f} MB")

import numpy as np
import pandas as pd

df = pd.read_csv(PSEUDO_LOCAL)
print(f"\nCSV shape: {df.shape}")
print(f"Columns (first 5): {df.columns[:5].tolist()}")
print(f"Columns (last 3): {df.columns[-3:].tolist()}")

# Identify prob columns
META_COLS = ["filename", "start_sec", "end_sec"]
PROB_COLS = [c for c in df.columns if c not in META_COLS]
print(f"Prob columns: {len(PROB_COLS)}")
assert len(PROB_COLS) == 234, f"Expected 234 species, got {len(PROB_COLS)}"

# Get prob matrix
probs = df[PROB_COLS].values  # (N, 234)
print(f"Prob matrix: {probs.shape}, dtype={probs.dtype}")
print(f"  prob range: [{probs.min():.6f}, {probs.max():.6f}]")
print(f"  prob mean: {probs.mean():.6f}, std: {probs.std():.6f}")

# Per-row max
max_prob = probs.max(axis=1)  # (N,)
print(f"\nMax prob per row (chunk):")
print(f"  min:    {max_prob.min():.4f}")
print(f"  max:    {max_prob.max():.4f}")
print(f"  mean:   {max_prob.mean():.4f}")
print(f"  median: {np.median(max_prob):.4f}")

# Percentiles
print(f"\nPercentile distribution of max_prob:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(max_prob, p)
    print(f"  {p:3d}%ile: {val:.4f}")

# Retention rate at various thresholds
print(f"\n{'='*60}")
print(f"Retention rate at various thresholds")
print(f"{'='*60}")
print(f"{'threshold':12s} | {'kept':10s} | {'rate':8s} | {'drop':10s}")
for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]:
    n_keep = int((max_prob > t).sum())
    rate = 100 * n_keep / len(max_prob)
    drop = len(max_prob) - n_keep
    print(f"  > {t:.2f}     | {n_keep:8d}   | {rate:5.1f}%  | {drop:8d}")

# Text histogram
print(f"\n{'='*60}")
print(f"Histogram of max_prob (text)")
print(f"{'='*60}")
bins = np.linspace(0, 1.0, 21)
hist, edges = np.histogram(max_prob, bins=bins)
max_count = hist.max()
for i in range(len(hist)):
    bar_len = int(50 * hist[i] / max_count)
    bar = "#" * bar_len
    print(f"  {edges[i]:.2f}-{edges[i+1]:.2f}: {hist[i]:6d} {bar}")

# Class coverage at filtered thresholds
print(f"\n{'='*60}")
print(f"Class coverage (how many classes have >=1 chunk with prob > threshold)")
print(f"{'='*60}")
print(f"{'threshold':12s} | {'classes_w_chunks':18s} | {'coverage':10s}")
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    kept_mask = max_prob > t
    n_chunks_kept = int(kept_mask.sum())
    if n_chunks_kept == 0:
        cls_with_chunks = 0
    else:
        probs_kept = probs[kept_mask]
        # For each class, count chunks where this class is argmax AND > t
        argmax_classes = probs_kept.argmax(axis=1)
        cls_with_chunks = int(len(np.unique(argmax_classes)))
    print(f"  > {t:.2f}     | {cls_with_chunks:14d}     | {100*cls_with_chunks/234:5.1f}%")

# Per-file diagnostic
print(f"\n{'='*60}")
print(f"Per-file diagnostic")
print(f"{'='*60}")
df["max_prob"] = max_prob
file_stats = df.groupby("filename")["max_prob"].agg(["max", "mean", "count"]).reset_index()
file_stats.columns = ["filename", "file_max", "file_mean", "n_chunks"]
print(f"Total files: {len(file_stats)}")
print(f"Files with file_max > 0.5: {int((file_stats['file_max'] > 0.5).sum())} ({100*int((file_stats['file_max'] > 0.5).sum())/len(file_stats):.1f}%)")
print(f"Files with file_max > 0.3: {int((file_stats['file_max'] > 0.3).sum())} ({100*int((file_stats['file_max'] > 0.3).sum())/len(file_stats):.1f}%)")
print(f"Avg chunks per file: {file_stats['n_chunks'].mean():.1f}")

# Recommendation
print(f"\n{'='*60}")
print(f"Recommendation")
print(f"{'='*60}")
rate_05 = 100 * (max_prob > 0.5).sum() / len(max_prob)
rate_03 = 100 * (max_prob > 0.3).sum() / len(max_prob)
print(f"Retention at threshold=0.5: {rate_05:.1f}%")
print(f"Retention at threshold=0.3: {rate_03:.1f}%")
if rate_05 > 30:
    print(f"  → threshold=0.5 (paper 256 spec) は妥当: 残存率 30-100% で学習データ十分")
elif rate_05 > 10:
    print(f"  → threshold=0.5 は厳しめ: 残存率 {rate_05:.1f}%、学習データ大幅減 (10-30%)")
    print(f"  → threshold=0.3 推奨 (残存率 {rate_03:.1f}%)")
else:
    print(f"  → threshold=0.5 は災害的: 残存率 {rate_05:.1f}% (<10%) で学習データ不足")
    if rate_03 > 30:
        print(f"  → threshold=0.3 推奨 (残存率 {rate_03:.1f}%)")
    elif rate_03 > 10:
        print(f"  → threshold=0.2 検討 ({100*(max_prob>0.2).sum()/len(max_prob):.1f}% 残存)")
    else:
        print(f"  → chunk filter 概念自体 BC2026 に合わない可能性、percentile-based を検討")
