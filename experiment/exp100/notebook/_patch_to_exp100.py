"""Patch exp029 R3 gen script -> exp100 (Babych spec + per-class WRS).

Changes from exp029 R3 (single fold, l1):
  1. MIXUP_ALPHA: 0.4 -> 1.0          (Babych BC25 1st)
  2. drop_path_rate: 0.1 -> 0.15      (Babych R2+ spec, [[feedback_droppath_stage_mismatch]])
  3. SHARES_R2 focal/pseudo_sc: 0.65/0.25 -> 0.60/0.30  (pseudo distill 強化)
  4. MixSamp + train_r2: focal source per-class inverse-freq WRS 追加
  5. Slug / output dir / docstrings 更新

Run: python _patch_to_exp100.py  ->  modifies _gen_nb_train_r3.py in place
Then: python _gen_nb_train_r3.py  ->  produces nb_train_r3.ipynb
"""
import io, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path(__file__).with_name("_gen_nb_train_r3.py")
content = SRC.read_text(encoding="utf-8")
print(f"Original size: {len(content)} chars")

# ============================================================
# Patch 1: MIXUP_ALPHA 0.4 -> 1.0
# ============================================================
OLD1 = "MIXUP_ALPHA = 0.4"
NEW1 = "MIXUP_ALPHA = 1.0  # ★ exp100: 0.4 -> 1.0 (Babych BC25 1st spec)"
assert content.count(OLD1) == 1, f"MIXUP_ALPHA pattern count: {content.count(OLD1)}"
content = content.replace(OLD1, NEW1)
print("[OK] MIXUP_ALPHA: 0.4 -> 1.0")

# ============================================================
# Patch 2: drop_path_rate=0.1 -> 0.15 (model class signature)
# ============================================================
OLD2 = "drop_path_rate=0.1, hidden_dim=512):"
NEW2 = "drop_path_rate=0.15, hidden_dim=512):  # ★ exp100: 0.1 -> 0.15 (Babych R2+ spec)"
assert content.count(OLD2) == 1, f"drop_path_rate pattern count: {content.count(OLD2)}"
content = content.replace(OLD2, NEW2)
print("[OK] drop_path_rate: 0.1 -> 0.15")

# ============================================================
# Patch 3: SHARES_R2 focal 0.65 -> 0.60, pseudo_sc 0.25 -> 0.30
# ============================================================
OLD3 = 'SHARES_R2 = {"focal": 0.65, "labeled_sc": 0.10, "pseudo_sc": 0.25}'
NEW3 = 'SHARES_R2 = {"focal": 0.60, "labeled_sc": 0.10, "pseudo_sc": 0.30}'
assert content.count(OLD3) == 1, f"SHARES_R2 pattern count: {content.count(OLD3)}"
content = content.replace(OLD3, NEW3)
print("[OK] SHARES_R2: focal 0.65 -> 0.60, pseudo_sc 0.25 -> 0.30")

# ============================================================
# Patch 4: MixSamp class - support focal_weights
# ============================================================
OLD4 = '''class MixSamp(torch.utils.data.Sampler):
    def __init__(self, sizes, names, shares, bs, nst, seed=0):
        self.sizes, self.names, self.bs, self.nst = sizes, names, bs, nst
        self.rng = np.random.default_rng(seed)
        per_src = [max(1, int(round(bs * shares.get(n, 0.0)))) for n in names]
        total = sum(per_src)
        if total != bs:
            per_src[int(np.argmax(per_src))] += (bs - total)
        self.per_src = per_src
        self.offsets = [0]
        for s in sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)
    def __len__(self): return self.nst
    def __iter__(self):
        for _ in range(self.nst):
            batch = []
            for off, size, n in zip(self.offsets, self.sizes, self.per_src):
                if n <= 0 or size <= 0: continue
                idxs = self.rng.integers(0, size, size=n)
                batch.extend([off + int(i) for i in idxs])
            self.rng.shuffle(batch)
            yield batch'''

NEW4 = '''class MixSamp(torch.utils.data.Sampler):
    """★ exp100: per-source mixing + per-class inverse-freq weighting for focal source.

    Changes from exp029:
      - focal_weights (per-sample prob vector) optional kwarg
      - If provided AND name == 'focal', use rng.choice with p=focal_weights
        (rare class up-sampled per inverse freq, like Babych WeightedRandomSampler)
      - Other sources keep uniform sampling
    """
    def __init__(self, sizes, names, shares, bs, nst, seed=0, focal_weights=None):
        self.sizes, self.names, self.bs, self.nst = sizes, names, bs, nst
        self.rng = np.random.default_rng(seed)
        per_src = [max(1, int(round(bs * shares.get(n, 0.0)))) for n in names]
        total = sum(per_src)
        if total != bs:
            per_src[int(np.argmax(per_src))] += (bs - total)
        self.per_src = per_src
        self.offsets = [0]
        for s in sizes[:-1]:
            self.offsets.append(self.offsets[-1] + s)
        # ★ exp100: focal_weights — must match focal source size, normalized to sum 1
        self.focal_weights = None
        if focal_weights is not None:
            fw = np.asarray(focal_weights, dtype=np.float64)
            assert "focal" in names, "focal_weights provided but no 'focal' source"
            focal_idx = names.index("focal")
            assert len(fw) == sizes[focal_idx], \\
                f"focal_weights len {len(fw)} != focal size {sizes[focal_idx]}"
            s = fw.sum()
            assert s > 0, "focal_weights must sum > 0"
            self.focal_weights = fw / s
            print(f"  [MixSamp] focal WRS active: weights sum=1, min={self.focal_weights.min():.2e}, "
                  f"max={self.focal_weights.max():.2e}, max/min ratio={self.focal_weights.max()/max(self.focal_weights.min(),1e-12):.1f}")
    def __len__(self): return self.nst
    def __iter__(self):
        for _ in range(self.nst):
            batch = []
            for off, size, name, n in zip(self.offsets, self.sizes, self.names, self.per_src):
                if n <= 0 or size <= 0: continue
                if name == "focal" and self.focal_weights is not None:
                    # ★ exp100: per-class inverse-freq weighted sampling for focal
                    idxs = self.rng.choice(size, size=n, replace=True, p=self.focal_weights)
                else:
                    idxs = self.rng.integers(0, size, size=n)
                batch.extend([off + int(i) for i in idxs])
            self.rng.shuffle(batch)
            yield batch'''

assert OLD4 in content, "MixSamp pattern not found"
content = content.replace(OLD4, NEW4)
print("[OK] MixSamp: focal_weights support added")

# ============================================================
# Patch 5: train_r2 — compute focal_weights + pass to MixSamp
# ============================================================
# Insert focal_weights computation after fds = FocalDS(...) and pass to MixSamp.
OLD5_fds = '''items = []
    fds = FocalDS(train_df[train_df["fold"] != fold_k],
                  LABEL2IDX, secondary_lookup=focal_secondary_labels, aug=True)
    items.append(("focal", fds, len(fds)))'''

NEW5_fds = '''items = []
    fds_df = train_df[train_df["fold"] != fold_k].reset_index(drop=True)
    fds = FocalDS(fds_df,
                  LABEL2IDX, secondary_lookup=focal_secondary_labels, aug=True)
    items.append(("focal", fds, len(fds)))

    # ★ exp100: per-class inverse-freq weights for focal WRS
    _cls_counts = fds_df["primary_label"].astype(str).value_counts().to_dict()
    _focal_weights = np.array([
        1.0 / max(_cls_counts.get(str(lbl), 1), 1)
        for lbl in fds_df["primary_label"].astype(str).values
    ], dtype=np.float64)
    # Stats for log
    print(f"  [WRS] focal n_classes={len(_cls_counts)}, "
          f"min_count={min(_cls_counts.values())}, max_count={max(_cls_counts.values())}")'''

assert OLD5_fds in content, "FocalDS construction pattern not found"
content = content.replace(OLD5_fds, NEW5_fds)
print("[OK] train_r2: focal_weights computation inserted")

OLD5_smp = "smp = MixSamp(SIZES, NAMES, SHARES_R2, BATCH, n_steps_ep, seed=42 + epoch)"
NEW5_smp = "smp = MixSamp(SIZES, NAMES, SHARES_R2, BATCH, n_steps_ep, seed=42 + epoch, focal_weights=_focal_weights)"
assert OLD5_smp in content, "MixSamp construction pattern not found"
content = content.replace(OLD5_smp, NEW5_smp)
print("[OK] train_r2: MixSamp construction takes focal_weights")

# ============================================================
# Patch 6: Slug / output dir / docstring
# ============================================================
# Output dir reference
content = content.replace(
    'exp029',
    'exp100',
)
print("[OK] All exp029 references -> exp100")

# Restore docstring 'exp029 R3' reference (keep historical correct names where useful)
# Actually, the patch above blanket-replaced all instances. Let's reverse some
# to keep readability of pseudo source description (pseudo_e17.csv comes from exp017).
# Check for accidental breakage:
checks = ["pseudo_e17", "exp017", "exp020"]
for kw in checks:
    if kw in content:
        print(f"  preserved: {kw}")
    else:
        print(f"  WARN: {kw} missing — check accidental replacement")

# Save
SRC.write_text(content, encoding="utf-8")
print(f"\n[OK] Saved {SRC} ({len(content)} chars)")
