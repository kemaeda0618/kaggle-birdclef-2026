# Instantiate and train ProtoSSM

# Reshape training data to file-level tensors
emb_files, file_list = reshape_to_files(emb_full, meta_full)
logits_files, _ = reshape_to_files(scores_full_raw, meta_full)
labels_files, _ = reshape_to_files(Y_FULL, meta_full)

print(f"Reshaped to file-level: emb={emb_files.shape}, logits={logits_files.shape}, labels={labels_files.shape}")
print(f"Files: {len(file_list)}")

# Build family mapping for taxonomic auxiliary loss
n_families, class_to_family, fam_to_idx = build_family_mapping(taxonomy, PRIMARY_LABELS)
print(f"Taxonomic families: {n_families}")

# Build per-file family labels (multi-hot)
file_families = np.zeros((len(file_list), n_families), dtype=np.float32)
for fi in range(len(file_list)):
    active_classes = np.where(labels_files[fi].sum(axis=0) > 0)[0]
    for ci in active_classes:
        file_families[fi, class_to_family[ci]] = 1.0

# Instantiate model
ssm_cfg = CFG["proto_ssm"]
model = ProtoSSM(
    d_input=emb_full.shape[1],  # 1536
    d_model=ssm_cfg["d_model"],
    d_state=ssm_cfg["d_state"],
    n_ssm_layers=ssm_cfg["n_ssm_layers"],
    n_classes=N_CLASSES,
    n_windows=N_WINDOWS,
    dropout=ssm_cfg["dropout"],
).to(DEVICE)

# Initialize prototypes from class-mean embeddings
emb_flat_tensor = torch.tensor(emb_full, dtype=torch.float32)
labels_flat_tensor = torch.tensor(Y_FULL, dtype=torch.float32)
model.init_prototypes_from_data(emb_flat_tensor, labels_flat_tensor)
print("Prototypes initialized from class means.")

# Initialize taxonomic head
model.init_family_head(n_families, class_to_family)
print(f"Family head initialized: {n_families} families")

print(f"ProtoSSM parameters: {model.count_parameters():,}")

# Train
t0 = time.time()
model, train_history = train_proto_ssm(
    model,
    emb_files, logits_files, labels_files.astype(np.float32),
    file_families=file_families,
    cfg=CFG["proto_ssm_train"],
    verbose=True,
)
train_time = time.time() - t0
print(f"ProtoSSM training time: {train_time:.1f}s")

# Also train MLP probes as ensemble component (same as V11)
# Keep the proven MLP probe pipeline for ensemble with ProtoSSM
PROBE_CLASS_IDX = np.where(Y_FULL.sum(axis=0) >= int(CFG["frozen_best_probe"]["min_pos"]))[0].astype(np.int32)

probe_models = {}
oof_base_for_probes = oof_base
oof_prior_for_probes = oof_prior

for cls_idx in tqdm(PROBE_CLASS_IDX, desc="Training MLP probes (ensemble)", disable=not CFG["verbose"]):
    y = Y_FULL[:, cls_idx]
    if y.sum() == 0 or y.sum() == len(y):
        continue
    
    X_cls = build_class_features(
        Z_FULL,
        raw_col=scores_full_raw[:, cls_idx],
        prior_col=oof_prior_for_probes[:, cls_idx],
        base_col=oof_base_for_probes[:, cls_idx],
    )
    
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos > 0 and n_neg > n_pos:
        repeat = max(1, n_neg // n_pos)
        pos_idx = np.where(y == 1)[0]
        X_bal = np.vstack([X_cls, np.tile(X_cls[pos_idx], (repeat, 1))])
        y_bal = np.concatenate([y, np.ones(len(pos_idx) * repeat, dtype=y.dtype)])
    else:
        X_bal, y_bal = X_cls, y
    
    clf = MLPClassifier(**CFG["mlp_params"])
    clf.fit(X_bal, y_bal)
    probe_models[cls_idx] = clf

print(f"MLP probes trained: {len(probe_models)}")