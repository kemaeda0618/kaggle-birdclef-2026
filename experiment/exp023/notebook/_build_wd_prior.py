"""Offline build of species-level wet/dry empirical prior from train_soundscapes_labels.

Output: wd_prior.npz with arrays:
  - boost_wet  (N_CLASSES,) — multiplicative factor for wet season
  - boost_dry  (N_CLASSES,) — multiplicative factor for dry season
  - primary_labels (N_CLASSES,) — column order

Method:
  Empirical Bayes shrinkage toward global prior:
    p_c_global = (n_c_total + ALPHA * uniform) / (n_total + ALPHA)
    p_c_s = (n_c_s + ALPHA * p_c_global) / (n_total_s + ALPHA)
    boost[c, s] = p_c_s / p_c_global  (clamped to [0.7, 1.5])

  Season definition: 11-4 月 = wet, 5-10 月 = dry (Pantanal Brazil)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import re

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent  # birdclef-2026/

# Hyperparams
ALPHA = 50.0           # shrinkage strength (larger = more global-prior-like)
CLAMP_MIN = 0.7        # multiplier lower bound
CLAMP_MAX = 1.5        # multiplier upper bound
MIN_SAMPLES = 5        # if n_c < MIN_SAMPLES, set boost = 1.0 (no signal)

# --- Load data ---
tax = pd.read_csv(ROOT / "taxonomy.csv")
PRIMARY_LABELS = tax["primary_label"].astype(str).tolist()
N_CLASSES = len(PRIMARY_LABELS)
print(f"Loaded {N_CLASSES} species from taxonomy.csv")

labels_df = pd.read_csv(ROOT / "train_soundscapes_labels.csv")
print(f"Loaded {len(labels_df)} segments from train_soundscapes_labels.csv")

# --- Parse season from filename ---
FN_RE = re.compile(r"BC2026_(?:Train|Test)_\d+_S(\d+)_(\d{8})_(\d{6})")

def parse_month(fn):
    m = FN_RE.match(fn)
    if not m:
        return None
    return int(m.group(2)[4:6])

def month_to_season(mm):
    if mm is None: return None
    # Pantanal: wet (Nov-Apr), dry (May-Oct)
    return "wet" if mm in (11, 12, 1, 2, 3, 4) else "dry"

labels_df["month"] = labels_df["filename"].apply(parse_month)
labels_df["season"] = labels_df["month"].apply(month_to_season)
print(f"Season distribution: {labels_df['season'].value_counts().to_dict()}")

# --- Build per-(species, season) co-occurrence ---
# A segment can have multiple primary_label entries separated by ';'
co_wet = np.zeros(N_CLASSES, dtype=np.float64)
co_dry = np.zeros(N_CLASSES, dtype=np.float64)
label_to_idx = {lbl: i for i, lbl in enumerate(PRIMARY_LABELS)}

n_wet_segs = (labels_df["season"] == "wet").sum()
n_dry_segs = (labels_df["season"] == "dry").sum()
print(f"Wet segs: {n_wet_segs}, Dry segs: {n_dry_segs}")

n_unmapped = 0
n_mapped = 0
for _, row in labels_df.iterrows():
    if row["season"] not in ("wet", "dry"):
        continue
    if pd.isna(row["primary_label"]):
        continue
    labs = str(row["primary_label"]).split(";")
    for lab in labs:
        if not lab.strip():
            continue
        idx = label_to_idx.get(lab.strip())
        if idx is None:
            n_unmapped += 1
            continue
        n_mapped += 1
        if row["season"] == "wet":
            co_wet[idx] += 1
        else:
            co_dry[idx] += 1

print(f"Label mapping: {n_mapped} mapped, {n_unmapped} unmapped (not in taxonomy)")

# --- Empirical Bayes shrinkage ---
# Global rate per species (across both seasons)
n_total = n_wet_segs + n_dry_segs
uniform = 1.0 / N_CLASSES
p_c_global = (co_wet + co_dry + ALPHA * uniform) / (n_total + ALPHA)

# Per-season rate with shrinkage toward global
p_c_wet = (co_wet + ALPHA * p_c_global) / (n_wet_segs + ALPHA)
p_c_dry = (co_dry + ALPHA * p_c_global) / (n_dry_segs + ALPHA)

# Boost factor = season rate / global rate
boost_wet = p_c_wet / p_c_global
boost_dry = p_c_dry / p_c_global

# Clamp to [CLAMP_MIN, CLAMP_MAX]
boost_wet_clamped = np.clip(boost_wet, CLAMP_MIN, CLAMP_MAX)
boost_dry_clamped = np.clip(boost_dry, CLAMP_MIN, CLAMP_MAX)

# Species with low sample size → neutralize (boost = 1.0)
n_obs = co_wet + co_dry
mask_low = n_obs < MIN_SAMPLES
boost_wet_clamped[mask_low] = 1.0
boost_dry_clamped[mask_low] = 1.0

print(f"\n--- Boost statistics ---")
print(f"Active species (n>={MIN_SAMPLES}): {(~mask_low).sum()} / {N_CLASSES}")
print(f"boost_wet: min={boost_wet_clamped.min():.3f}, max={boost_wet_clamped.max():.3f}, "
      f"mean={boost_wet_clamped.mean():.3f}, n_clamped_max={int((boost_wet == boost_wet_clamped.max()).sum() if False else (boost_wet > CLAMP_MAX).sum())}, "
      f"n_clamped_min={int((boost_wet < CLAMP_MIN).sum())}")
print(f"boost_dry: min={boost_dry_clamped.min():.3f}, max={boost_dry_clamped.max():.3f}, "
      f"mean={boost_dry_clamped.mean():.3f}, n_clamped_max={int((boost_dry > CLAMP_MAX).sum())}, "
      f"n_clamped_min={int((boost_dry < CLAMP_MIN).sum())}")

# Top 10 wet-boosted species
print(f"\nTop 10 wet-boosted species:")
order_wet = np.argsort(-boost_wet_clamped)
for i in order_wet[:10]:
    if mask_low[i]: continue
    print(f"  {PRIMARY_LABELS[i]:15s}  boost_wet={boost_wet_clamped[i]:.3f}  "
          f"boost_dry={boost_dry_clamped[i]:.3f}  (n_wet={int(co_wet[i])}, n_dry={int(co_dry[i])})")

print(f"\nTop 10 dry-boosted species:")
order_dry = np.argsort(-boost_dry_clamped)
for i in order_dry[:10]:
    if mask_low[i]: continue
    print(f"  {PRIMARY_LABELS[i]:15s}  boost_wet={boost_wet_clamped[i]:.3f}  "
          f"boost_dry={boost_dry_clamped[i]:.3f}  (n_wet={int(co_wet[i])}, n_dry={int(co_dry[i])})")

# --- Save ---
out_path = HERE / "wd_prior.npz"
np.savez_compressed(
    out_path,
    boost_wet=boost_wet_clamped.astype(np.float32),
    boost_dry=boost_dry_clamped.astype(np.float32),
    primary_labels=np.array(PRIMARY_LABELS),
    n_wet_obs=co_wet.astype(np.int32),
    n_dry_obs=co_dry.astype(np.int32),
    alpha=ALPHA,
    clamp_min=CLAMP_MIN,
    clamp_max=CLAMP_MAX,
    min_samples=MIN_SAMPLES,
)
print(f"\nSaved: {out_path} ({out_path.stat().st_size/1024:.1f}KB)")
