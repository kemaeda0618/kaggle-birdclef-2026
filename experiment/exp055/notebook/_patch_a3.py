"""Apply A3 (retrieval pool dedup) patch to exp055 NB.

Strategy:
1. TA pool loading cell: keep ta_pool_authors + ta_pool_label_str (drop only emb_raw, _ta_norm)
2. External concat: also concat author/label arrays (pseudo "external_X" author)
3. compute_retrieval_ta_logit: top-3K → dedup by (author, label_str) → top-K
"""
import json
import re
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp055\notebook\nb_blend_a3.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

# ============================================================
# Fix 1: TA pool loading — preserve author + label_str
# ============================================================
TA_LOAD_OLD = """    _ta_meta = pd.read_parquet(_ta_pq)
    _ta_data = np.load(_ta_npz)
    _ta_emb_raw = _ta_data["embeddings"].astype(np.float32)

    label_to_idx = {l: i for i, l in enumerate(PRIMARY_LABELS)}
    ta_pool_labels = np.zeros((len(_ta_meta), N_CLASSES), dtype=np.float32)
    for _i, _lbl in enumerate(_ta_meta["primary_label"].values):
        if _lbl in label_to_idx:
            ta_pool_labels[_i, label_to_idx[_lbl]] = 1.0

    _ta_norm = np.linalg.norm(_ta_emb_raw, axis=1, keepdims=True) + 1e-8
    ta_pool_emb_norm = (_ta_emb_raw / _ta_norm).astype(np.float32)

    del _ta_emb_raw, _ta_norm, _ta_meta"""

TA_LOAD_NEW = """    _ta_meta = pd.read_parquet(_ta_pq)
    _ta_data = np.load(_ta_npz)
    _ta_emb_raw = _ta_data["embeddings"].astype(np.float32)

    label_to_idx = {l: i for i, l in enumerate(PRIMARY_LABELS)}
    ta_pool_labels = np.zeros((len(_ta_meta), N_CLASSES), dtype=np.float32)
    for _i, _lbl in enumerate(_ta_meta["primary_label"].values):
        if _lbl in label_to_idx:
            ta_pool_labels[_i, label_to_idx[_lbl]] = 1.0

    # A3: preserve author + label_str for retrieval dedup
    ta_pool_label_str = _ta_meta["primary_label"].astype(str).values
    if "author" in _ta_meta.columns:
        ta_pool_authors = _ta_meta["author"].astype(str).values
    else:
        # Fall back: load BC2026 train.csv + map by filename
        try:
            _bc_train = pd.read_csv(BC2026_TRAIN_CSV)
            _fn_to_author = dict(zip(_bc_train["filename"], _bc_train["author"]))
            if "filename" in _ta_meta.columns:
                ta_pool_authors = np.array(
                    [_fn_to_author.get(fn, f"unk_{i}") for i, fn in enumerate(_ta_meta["filename"].values)]
                )
            else:
                ta_pool_authors = np.array([f"unk_{i}" for i in range(len(_ta_meta))])
        except Exception:
            ta_pool_authors = np.array([f"unk_{i}" for i in range(len(_ta_meta))])
    print(f"A3 dedup: ta_pool_authors unique={len(set(ta_pool_authors))}, "
          f"ta_pool_label_str unique={len(set(ta_pool_label_str))}")

    _ta_norm = np.linalg.norm(_ta_emb_raw, axis=1, keepdims=True) + 1e-8
    ta_pool_emb_norm = (_ta_emb_raw / _ta_norm).astype(np.float32)

    del _ta_emb_raw, _ta_norm, _ta_meta"""

# ============================================================
# Fix 2: External pool concat — add pseudo author/label_str
# ============================================================
EXT_CONCAT_OLD = """        ta_pool_emb_norm = np.concatenate([ta_pool_emb_norm, _ext_emb], axis=0)
        ta_pool_labels   = np.concatenate([ta_pool_labels, _ext_lbl], axis=0)"""

EXT_CONCAT_NEW = """        ta_pool_emb_norm = np.concatenate([ta_pool_emb_norm, _ext_emb], axis=0)
        ta_pool_labels   = np.concatenate([ta_pool_labels, _ext_lbl], axis=0)
        # A3 dedup: extend with unique pseudo author/label_str per external entry
        # (each ext entry unique → dedup never removes them)
        _n_ext = len(_ext_lbl)
        _ext_authors = np.array([f"ext_{i}" for i in range(_n_ext)])
        _ext_label_str = np.array([f"ext_lbl_{i}" for i in range(_n_ext)])
        ta_pool_authors = np.concatenate([ta_pool_authors, _ext_authors])
        ta_pool_label_str = np.concatenate([ta_pool_label_str, _ext_label_str])"""

# ============================================================
# Fix 3: compute_retrieval_ta_logit — dedup top-3K → top-K
# ============================================================
COMPUTE_TA_OLD = """    sim = e_norm @ ta_pool_emb_norm.T          # (N_windows, 265k)
    K_eff = min(K, sim.shape[1])
    topk_idx = np.argpartition(sim, -K_eff, axis=1)[:, -K_eff:]
    topk_sim = np.take_along_axis(sim, topk_idx, axis=1)
    topk_label = ta_pool_labels[topk_idx]      # (N_windows, K, N_CLASSES)"""

COMPUTE_TA_NEW = """    sim = e_norm @ ta_pool_emb_norm.T          # (N_windows, 265k)
    K_eff = min(K, sim.shape[1])
    # A3: search top-3K candidates, then dedup by (author, label_str) → top-K
    K_SEARCH_FACTOR = 3
    K_search = min(K_eff * K_SEARCH_FACTOR, sim.shape[1])
    topk_idx_raw = np.argpartition(sim, -K_search, axis=1)[:, -K_search:]
    topk_sim_raw = np.take_along_axis(sim, topk_idx_raw, axis=1)
    # Sort descending by similarity
    _sort = np.argsort(-topk_sim_raw, axis=1)
    topk_idx_raw = np.take_along_axis(topk_idx_raw, _sort, axis=1)
    topk_sim_raw = np.take_along_axis(topk_sim_raw, _sort, axis=1)
    # Per-window dedup by (author, label_str)
    n_test = topk_idx_raw.shape[0]
    topk_idx = np.zeros((n_test, K_eff), dtype=np.int64)
    topk_sim = np.zeros((n_test, K_eff), dtype=np.float32)
    for _ti in range(n_test):
        _seen = set()
        _out = 0
        for _j in range(topk_idx_raw.shape[1]):
            _idx = int(topk_idx_raw[_ti, _j])
            _key = (ta_pool_authors[_idx], ta_pool_label_str[_idx])
            if _key in _seen:
                continue
            _seen.add(_key)
            topk_idx[_ti, _out] = _idx
            topk_sim[_ti, _out] = topk_sim_raw[_ti, _j]
            _out += 1
            if _out >= K_eff:
                break
        # Fill rest with last valid (if dedup left < K_eff)
        if _out < K_eff and _out > 0:
            topk_idx[_ti, _out:] = topk_idx[_ti, _out - 1]
            topk_sim[_ti, _out:] = 0.0
    topk_label = ta_pool_labels[topk_idx]      # (N_windows, K, N_CLASSES)"""

# ============================================================
# Header: add BC2026_TRAIN_CSV constant
# ============================================================
BC_TRAIN_CSV_DEF = """
# A3: BC2026 train.csv path for author lookup fallback
BC2026_TRAIN_CSV = "/kaggle/input/birdclef-2026/train.csv"
if not Path(BC2026_TRAIN_CSV).exists():
    BC2026_TRAIN_CSV = "/kaggle/input/competitions/birdclef-2026/train.csv"
"""

# Apply patches
n_changes = 0
for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"])

    # Add BC2026_TRAIN_CSV constant near top (look for ROOT or BASE def)
    if 'EMB_DIR = ' in src and 'BC2026_TRAIN_CSV' not in src:
        # Insert after EMB_DIR definition
        src = src.replace(
            "EMB_DIR = ",
            BC_TRAIN_CSV_DEF + "\nEMB_DIR = "
        )
        n_changes += 1

    if TA_LOAD_OLD in src:
        src = src.replace(TA_LOAD_OLD, TA_LOAD_NEW)
        n_changes += 1
        print("[Fix 1] TA pool loading patched (author + label_str preserved)")

    if EXT_CONCAT_OLD in src:
        src = src.replace(EXT_CONCAT_OLD, EXT_CONCAT_NEW)
        n_changes += 1
        print("[Fix 2] External pool concat patched (pseudo author/label_str)")

    if COMPUTE_TA_OLD in src:
        src = src.replace(COMPUTE_TA_OLD, COMPUTE_TA_NEW)
        n_changes += 1
        print("[Fix 3] compute_retrieval_ta_logit patched (top-3K dedup)")

    c["source"] = src.splitlines(keepends=True)

# Update hdr cell for A3 marker
for c in nb["cells"]:
    if c.get("id") == "hdr":
        src = "".join(c["source"])
        if "A3 Retrieval" not in src:
            prefix = """# exp055: A3 Retrieval Pool Dedup on exp048 (BC2026)

**Base**: exp048 (LB 0.950、TopN N=1 PP)
**新規変更**: TA pool retrieval で **(author × species) dedup**

## A3 仕組み

```
旧 (exp048): top-K=20 retrieval (重複可)
  → John Doe × Amazona が top-15/20 占める可能性

新 (exp055 A3):
  1. top-3K=60 取得 (margin あり)
  2. (author, primary_label) で dedup
  3. 上位 K=20 取り出し
  → diverse retrieval、Pantanal field 域への transfer 改善期待
```

## 期待 LB

- exp048 base: 0.950
- A3 効果: **+0.001-0.004**
- 目標 LB: **0.951-0.954** (gold border 接近)

## TopN N=1 との関係

- TopN N=1 (exp048 既存): **出力後 PP** (per-file × per-class scale)
- A3 (exp055 新規): **NB4 内部 retrieval (入力側 dedup)**
- → **直交 (orthogonal)**、効果累積期待

---

"""
            c["source"] = (prefix + src).splitlines(keepends=True)
            n_changes += 1
            print("[hdr] A3 prefix added")
        break

print(f"\nTotal changes: {n_changes}")
json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB}")
