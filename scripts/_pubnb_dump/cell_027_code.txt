# Cell 8 — Honest OOF base/prior meta-features (required for final stacker fit)
def build_oof_base_prior(scores_full_raw, meta_full, sc_clean, Y_SC, n_splits=5, verbose=True):
    groups_full = meta_full["filename"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    oof_base = np.zeros_like(scores_full_raw, dtype=np.float32)
    oof_prior = np.zeros_like(scores_full_raw, dtype=np.float32)
    fold_id = np.full(len(meta_full), -1, dtype=np.int16)

    splits = list(gkf.split(scores_full_raw, groups=groups_full))
    iterator = tqdm(splits, desc="OOF base/prior folds", disable=not verbose)

    for fold, (tr_idx, va_idx) in enumerate(iterator, 1):
        tr_idx = np.sort(tr_idx)
        va_idx = np.sort(va_idx)

        val_files = set(meta_full.iloc[va_idx]["filename"].tolist())

        # Fold-safe prior tables: exclude all validation files
        prior_mask = ~sc_clean["filename"].isin(val_files).values
        prior_df_fold = sc_clean.loc[prior_mask].reset_index(drop=True)
        Y_prior_fold = Y_SC[prior_mask]

        tables = fit_prior_tables(prior_df_fold, Y_prior_fold)

        va_base, va_prior = fuse_scores_with_tables(
            scores_full_raw[va_idx],
            sites=meta_full.iloc[va_idx]["site"].to_numpy(),
            hours=meta_full.iloc[va_idx]["hour_utc"].to_numpy(),
            tables=tables,
        )

        oof_base[va_idx] = va_base
        oof_prior[va_idx] = va_prior
        fold_id[va_idx] = fold

    assert (fold_id >= 0).all()
    return oof_base, oof_prior, fold_id

OOF_META_CACHE = CFG["full_cache_work_dir"] / "full_oof_meta_features.npz"

if OOF_META_CACHE.exists():
    print("Loading cached OOF meta-features from:", OOF_META_CACHE)
    arr = np.load(OOF_META_CACHE)
    oof_base = arr["oof_base"].astype(np.float32)
    oof_prior = arr["oof_prior"].astype(np.float32)
    oof_fold_id = arr["fold_id"].astype(np.int16)
else:
    print("Building OOF meta-features...")
    oof_base, oof_prior, oof_fold_id = build_oof_base_prior(
        scores_full_raw=scores_full_raw,
        meta_full=meta_full,
        sc_clean=sc_clean,
        Y_SC=Y_SC,
        n_splits=5,
        verbose=CFG["verbose"],
    )

    np.savez_compressed(
        OOF_META_CACHE,
        oof_base=oof_base,
        oof_prior=oof_prior,
        fold_id=oof_fold_id,
    )
    print("Saved OOF meta-features to:", OOF_META_CACHE)

baseline_oof_auc = macro_auc_skip_empty(Y_FULL, oof_base)

if MODE == "train":
    raw_local_auc = macro_auc_skip_empty(Y_FULL, scores_full_raw)
    print(f"Raw local AUC (not OOF-dependent): {raw_local_auc:.6f}")
    print(f"Honest OOF baseline AUC: {baseline_oof_auc:.6f}")