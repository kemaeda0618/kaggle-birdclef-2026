# Cell 6 — Load or compute full-file Perch cache
def resolve_full_cache_paths():
    candidates = []

    # Working dir cache
    candidates.append((
        CFG["full_cache_work_dir"] / "full_perch_meta.parquet",
        CFG["full_cache_work_dir"] / "full_perch_arrays.npz"
    ))

    # Legacy working paths
    candidates.append((
        Path("/kaggle/working/full_perch_meta.parquet"),
        Path("/kaggle/working/full_perch_arrays.npz")
    ))

    # Attached input dataset
    # Try multiple cache locations
    for cache_dir in [
        CFG["full_cache_input_dir"],
        Path("/kaggle/input/perch-meta"),
        Path("/kaggle/input/datasets/jaejohn/perch-meta"),
    ]:
        if cache_dir.exists():
            candidates.append((
                cache_dir / "full_perch_meta.parquet",
                cache_dir / "full_perch_arrays.npz"
            ))

    _skip_original_input = True
    if False:  # disabled, replaced by multi-path search above
        candidates.append((
            CFG["full_cache_input_dir"] / "full_perch_meta.parquet",
            CFG["full_cache_input_dir"] / "full_perch_arrays.npz"
        ))

    for meta_path, npz_path in candidates:
        if meta_path.exists() and npz_path.exists():
            return meta_path, npz_path

    return None, None

cache_meta, cache_npz = resolve_full_cache_paths()

if cache_meta is not None and cache_npz is not None:
    print("Loading cached full-file Perch outputs from:")
    print("  ", cache_meta)
    print("  ", cache_npz)

    meta_full = pd.read_parquet(cache_meta)
    arr = np.load(cache_npz)
    scores_full_raw = arr["scores_full_raw"].astype(np.float32)
    emb_full = arr["emb_full"].astype(np.float32)

else:
    if CFG["mode"] == "submit" and CFG["require_full_cache_in_submit"]:
        raise FileNotFoundError(
            "Submit mode requires cached full-file Perch outputs. "
            "Attach the cache dataset or place full_perch_meta.parquet/full_perch_arrays.npz in working dir."
        )

    print("No cache found. Running Perch on trusted full files...")
    full_paths = [BASE / "train_soundscapes" / fn for fn in full_files]

    # Use CFG["proxy_reduce"] for consistency with grid search
    meta_full, scores_full_raw, emb_full = infer_perch_with_embeddings(
        full_paths,
        batch_files=CFG["batch_files"],
        verbose=CFG["verbose"],
        proxy_reduce=CFG["proxy_reduce"],
    )

    out_meta = CFG["full_cache_work_dir"] / "full_perch_meta.parquet"
    out_npz = CFG["full_cache_work_dir"] / "full_perch_arrays.npz"

    meta_full.to_parquet(out_meta, index=False)
    np.savez_compressed(
        out_npz,
        scores_full_raw=scores_full_raw,
        emb_full=emb_full,
    )

    print("Saved cache to:")
    print("  ", out_meta)
    print("  ", out_npz)

# Align truth to cached order
full_truth_aligned = full_truth.set_index("row_id").loc[meta_full["row_id"]].reset_index()
Y_FULL = Y_SC[full_truth_aligned["index"].to_numpy()]

assert np.all(full_truth_aligned["filename"].values == meta_full["filename"].values)
assert np.all(full_truth_aligned["row_id"].values == meta_full["row_id"].values)

print("meta_full:", meta_full.shape)
print("scores_full_raw:", scores_full_raw.shape, scores_full_raw.dtype)
print("emb_full:", emb_full.shape, emb_full.dtype)
print("Y_FULL:", Y_FULL.shape, Y_FULL.dtype)

# [MODIFIED - Opsi 3] Grid search proxy_reduce: evaluasi "max" vs "mean" via OOF AUC
# Dilakukan hanya saat train mode; hasilnya di-freeze ke CFG["proxy_reduce"] untuk submit
PROXY_REDUCE_CACHE = CFG["full_cache_work_dir"] / "proxy_reduce_grid.json"

if CFG.get("run_proxy_reduce_grid", False):
    print("\n[Opsi 3] Running proxy_reduce grid search: max vs mean...")
    proxy_reduce_results = {}

    for pr in CFG["proxy_reduce_grid"]:
        full_paths = [BASE / "train_soundscapes" / fn for fn in full_files]
        _meta, _scores, _emb = infer_perch_with_embeddings(
            full_paths,
            batch_files=CFG["batch_files"],
            verbose=False,
            proxy_reduce=pr,
        )

        # OOF baseline AUC untuk proxy_reduce ini (tanpa probe)
        _oof_b, _oof_p, _ = build_oof_base_prior(
            scores_full_raw=_scores,
            meta_full=_meta,
            sc_clean=sc_clean,
            Y_SC=Y_SC,
            n_splits=5,
            verbose=False,
        )
        auc = macro_auc_skip_empty(Y_FULL, _oof_b)
        proxy_reduce_results[pr] = float(auc)
        print(f"  proxy_reduce={pr!r:6s} → OOF baseline AUC = {auc:.6f}")

    best_pr = max(proxy_reduce_results, key=proxy_reduce_results.get)
    CFG["proxy_reduce"] = best_pr
    print(f"\n  Best proxy_reduce = {best_pr!r} (AUC={proxy_reduce_results[best_pr]:.6f})")

    PROXY_REDUCE_CACHE.write_text(json.dumps({
        "results": proxy_reduce_results,
        "best_proxy_reduce": best_pr,
    }, indent=2))
    print("  Saved to:", PROXY_REDUCE_CACHE)

elif PROXY_REDUCE_CACHE.exists():
    _pr_data = json.loads(PROXY_REDUCE_CACHE.read_text())
    CFG["proxy_reduce"] = _pr_data["best_proxy_reduce"]
    print(f"[Opsi 3] Loaded proxy_reduce from cache: {CFG['proxy_reduce']!r}")
    print("  Grid results:", _pr_data["results"])

else:
    print(f"[Opsi 3] Using default proxy_reduce={CFG['proxy_reduce']!r} (submit mode or no cache)")