# Cell 13 — Fit embedding scaler + PCA on all trusted full windows
emb_scaler = StandardScaler()
emb_full_scaled = emb_scaler.fit_transform(emb_full)

n_comp = min(
    int(BEST_PROBE["pca_dim"]),
    emb_full_scaled.shape[0] - 1,
    emb_full_scaled.shape[1]
)

emb_pca = PCA(n_components=n_comp)
Z_FULL = emb_pca.fit_transform(emb_full_scaled).astype(np.float32)

print("emb_full:", emb_full.shape)
print("Z_FULL:", Z_FULL.shape)
print("Explained variance ratio sum:", emb_pca.explained_variance_ratio_.sum())