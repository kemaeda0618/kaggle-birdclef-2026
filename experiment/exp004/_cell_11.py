# Cell 9 — Classwise embedding-probe helpers
def build_class_features(emb_proj, raw_col, prior_col, base_col):
    """
    emb_proj: (n, d)
    raw_col, prior_col, base_col: (n,)
    returns: (n, d + 13)

    Fitur: embedding + 7 sequential + 3 interaction + std + 3 diff
    """
    prev_base, next_base, mean_base, max_base, std_base = seq_features_1d(base_col)

    # Diff features: posisi window relatif terhadap konteks file
    diff_mean = base_col - mean_base   # apakah window ini lebih tinggi dari rata2 file?
    diff_prev = base_col - prev_base   # onset: naik dari window sebelumnya?
    diff_next = base_col - next_base   # offset: turun ke window berikutnya?

    feats = np.concatenate([
        emb_proj,
        raw_col[:, None],
        prior_col[:, None],
        base_col[:, None],
        prev_base[:, None],
        next_base[:, None],
        mean_base[:, None],
        max_base[:, None],
        std_base[:, None],             # variance temporal dalam file
        diff_mean[:, None],            # deviasi dari mean file
        diff_prev[:, None],            # deteksi onset
        diff_next[:, None],            # deteksi offset
        # interaction terms
        (raw_col * prior_col)[:, None],
        (raw_col * base_col)[:, None],
        (prior_col * base_col)[:, None],
    ], axis=1)

    return feats.astype(np.float32, copy=False)

def run_oof_embedding_probe(
    scores_raw,
    emb,
    meta_df,
    y_true,
    pca_dim=64,
    min_pos=8,
    C=0.25,
    alpha=0.5,
):
    groups = meta_df["filename"].to_numpy()
    gkf = GroupKFold(n_splits=5)

    oof_base_local = np.zeros_like(scores_raw, dtype=np.float32)
    oof_final = np.zeros_like(scores_raw, dtype=np.float32)

    modeled_counts = np.zeros(scores_raw.shape[1], dtype=np.int32)

    split_list = list(gkf.split(scores_raw, groups=groups))

    for fold, (tr_idx, va_idx) in enumerate(tqdm(split_list, desc="Embedding-probe folds", disable=not CFG["verbose"]), 1):
    # for fold, (tr_idx, va_idx) in enumerate(tqdm(split_list, desc="Embedding-probe folds"), 1):
        tr_idx = np.sort(tr_idx)
        va_idx = np.sort(va_idx)

        val_files = set(meta_df.iloc[va_idx]["filename"].tolist())

        # Fold-safe priors
        prior_mask = ~sc_clean["filename"].isin(val_files).values
        prior_df_fold = sc_clean.loc[prior_mask].reset_index(drop=True)
        Y_prior_fold = Y_SC[prior_mask]
        tables = fit_prior_tables(prior_df_fold, Y_prior_fold)

        base_tr, prior_tr = fuse_scores_with_tables(
            scores_raw[tr_idx],
            sites=meta_df.iloc[tr_idx]["site"].to_numpy(),
            hours=meta_df.iloc[tr_idx]["hour_utc"].to_numpy(),
            tables=tables,
        )
        base_va, prior_va = fuse_scores_with_tables(
            scores_raw[va_idx],
            sites=meta_df.iloc[va_idx]["site"].to_numpy(),
            hours=meta_df.iloc[va_idx]["hour_utc"].to_numpy(),
            tables=tables,
        )

        oof_base_local[va_idx] = base_va
        oof_final[va_idx] = base_va

        # Embedding preprocessing on train fold only
        scaler = StandardScaler()
        emb_tr_s = scaler.fit_transform(emb[tr_idx])
        emb_va_s = scaler.transform(emb[va_idx])

        n_comp = min(pca_dim, emb_tr_s.shape[0] - 1, emb_tr_s.shape[1])
        pca = PCA(n_components=n_comp)
        Z_tr = pca.fit_transform(emb_tr_s).astype(np.float32)
        Z_va = pca.transform(emb_va_s).astype(np.float32)

        class_iterator = np.where(y_true[tr_idx].sum(axis=0) >= min_pos)[0].tolist()

        for cls_idx in tqdm(class_iterator, desc=f"Fold {fold} classes", leave=False, disable=not CFG["verbose"]):
        # for cls_idx in tqdm(class_iterator, desc=f"Fold {fold} classes", leave=False):
            y_tr = y_true[tr_idx, cls_idx]

            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                continue

            X_tr_cls = build_class_features(
                Z_tr,
                raw_col=scores_raw[tr_idx, cls_idx],
                prior_col=prior_tr[:, cls_idx],
                base_col=base_tr[:, cls_idx],
            )
            X_va_cls = build_class_features(
                Z_va,
                raw_col=scores_raw[va_idx, cls_idx],
                prior_col=prior_va[:, cls_idx],
                base_col=base_va[:, cls_idx],
            )

            # Pilih backend probe: mlp | lgbm | logreg
            backend = CFG.get("probe_backend", "mlp")
            n_pos = int(y_tr.sum())
            n_neg = len(y_tr) - n_pos

            if backend == "mlp":
                # MLPClassifier tidak support sample_weight
                # Gunakan oversampling: duplikasi positif agar balance
                if n_pos > 0 and n_neg > n_pos:
                    repeat = max(1, n_neg // n_pos)
                    pos_idx = np.where(y_tr == 1)[0]
                    X_bal = np.vstack([X_tr_cls, np.tile(X_tr_cls[pos_idx], (repeat, 1))])
                    y_bal = np.concatenate([y_tr, np.ones(len(pos_idx) * repeat, dtype=y_tr.dtype)])
                else:
                    X_bal, y_bal = X_tr_cls, y_tr
                clf = MLPClassifier(**CFG["mlp_params"])
                clf.fit(X_bal, y_bal)
                pred_va = clf.predict_proba(X_va_cls)[:, 1].astype(np.float32)
                pred_va = np.log(pred_va + 1e-7) - np.log(1 - pred_va + 1e-7)
            elif backend == "lgbm" and _LGBM_AVAILABLE:
                scale_pos = max(1.0, n_neg / max(n_pos, 1))
                clf = LGBMClassifier(
                    **CFG["lgbm_params"],
                    scale_pos_weight=scale_pos,
                )
                clf.fit(X_tr_cls, y_tr)
                pred_va = clf.predict_proba(X_va_cls)[:, 1].astype(np.float32)
                pred_va = np.log(pred_va + 1e-7) - np.log(1 - pred_va + 1e-7)
            else:
                clf = LogisticRegression(
                    C=C, max_iter=400, solver="liblinear",
                    class_weight="balanced",
                )
                clf.fit(X_tr_cls, y_tr)
                pred_va = clf.decision_function(X_va_cls).astype(np.float32)

            oof_final[va_idx, cls_idx] = (
                (1.0 - alpha) * base_va[:, cls_idx] +
                alpha * pred_va
            )

            modeled_counts[cls_idx] += 1

    score_base = macro_auc_skip_empty(y_true, oof_base_local)
    score_final = macro_auc_skip_empty(y_true, oof_final)

    return {
        "oof_base": oof_base_local,
        "oof_final": oof_final,
        "modeled_counts": modeled_counts,
        "score_base": score_base,
        "score_final": score_final,
    }