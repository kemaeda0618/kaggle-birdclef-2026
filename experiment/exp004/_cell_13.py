# ProtoSSM Training Loop

def build_family_mapping(taxonomy_df, primary_labels):
    """Build class-to-family index mapping from taxonomy."""
    if "family" not in taxonomy_df.columns:
        # Derive family from class_name or use order as fallback
        if "order" in taxonomy_df.columns:
            family_map = taxonomy_df.set_index("primary_label")["order"].to_dict()
        elif "class_name" in taxonomy_df.columns:
            family_map = taxonomy_df.set_index("primary_label")["class_name"].to_dict()
        else:
            family_map = {label: "Unknown" for label in primary_labels}
    else:
        family_map = taxonomy_df.set_index("primary_label")["family"].to_dict()
    families = sorted(set(family_map.values()))
    fam_to_idx = {f: i for i, f in enumerate(families)}
    class_to_family = []
    for label in primary_labels:
        fam = family_map.get(label, "Unknown")
        class_to_family.append(fam_to_idx.get(fam, 0))
    return len(families), class_to_family, fam_to_idx

def reshape_to_files(flat_array, meta_df, n_windows=N_WINDOWS):
    """Reshape flat (n_windows*n_files, ...) to (n_files, n_windows, ...).
    
    Groups by filename from meta_df, preserving file order.
    Returns reshaped array and list of unique filenames.
    """
    filenames = meta_df["filename"].to_numpy()
    unique_files = []
    seen = set()
    for f in filenames:
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    
    n_files = len(unique_files)
    assert len(flat_array) == n_files * n_windows, \
        f"Expected {n_files * n_windows} rows, got {len(flat_array)}"
    
    new_shape = (n_files, n_windows) + flat_array.shape[1:]
    return flat_array.reshape(new_shape), unique_files

def train_proto_ssm(model, emb_files, logits_files, labels_files, 
                    file_families=None, cfg=None, verbose=True):
    """Train ProtoSSM with multi-task loss and early stopping.
    
    Args:
        model: ProtoSSM instance
        emb_files: (n_files, n_windows, d_input) Perch embeddings
        logits_files: (n_files, n_windows, n_classes) Perch mapped logits
        labels_files: (n_files, n_windows, n_classes) binary labels
        file_families: (n_files, n_families) multi-hot family labels (optional)
        cfg: training config dict
    
    Returns:
        model with best weights loaded
        training history dict
    """
    if cfg is None:
        cfg = CFG["proto_ssm_train"]
    
    n_files = len(emb_files)
    n_val = max(1, int(n_files * cfg["val_ratio"]))
    
    # Deterministic train/val split by file
    perm = torch.randperm(n_files, generator=torch.Generator().manual_seed(42))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    
    # Convert to tensors
    emb_train = torch.tensor(emb_files[train_idx], dtype=torch.float32)
    logits_train = torch.tensor(logits_files[train_idx], dtype=torch.float32)
    labels_train = torch.tensor(labels_files[train_idx], dtype=torch.float32)
    
    emb_val = torch.tensor(emb_files[val_idx], dtype=torch.float32)
    logits_val = torch.tensor(logits_files[val_idx], dtype=torch.float32)
    labels_val = torch.tensor(labels_files[val_idx], dtype=torch.float32)
    
    # Family labels for auxiliary loss
    fam_train = fam_val = None
    if file_families is not None and model.family_head is not None:
        fam_train = torch.tensor(file_families[train_idx], dtype=torch.float32)
        fam_val = torch.tensor(file_families[val_idx], dtype=torch.float32)
    
    # Class weights for imbalanced data
    pos_counts = labels_train.sum(dim=(0, 1))  # (C,)
    total = labels_train.shape[0] * labels_train.shape[1]
    pos_weight = ((total - pos_counts) / (pos_counts + 1)).clamp(max=cfg["pos_weight_cap"])
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"], 
        epochs=cfg["n_epochs"], steps_per_epoch=1,
        pct_start=0.1, anneal_strategy='cos'
    )
    
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    
    for epoch in range(cfg["n_epochs"]):
        # === Train ===
        model.train()
        species_out, family_out, _ = model(emb_train, logits_train)
        
        # Primary loss: weighted BCE
        loss_bce = F.binary_cross_entropy_with_logits(
            species_out, labels_train,
            pos_weight=pos_weight[None, None, :]
        )
        
        # Knowledge distillation loss: MSE between model output and Perch logits
        loss_distill = F.mse_loss(species_out, logits_train)
        
        # Total loss
        loss = loss_bce + cfg["distill_weight"] * loss_distill
        
        # Taxonomic auxiliary loss
        if family_out is not None and fam_train is not None:
            loss_family = F.binary_cross_entropy_with_logits(family_out, fam_train)
            loss = loss + 0.1 * loss_family
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # === Validate ===
        model.eval()
        with torch.no_grad():
            val_out, val_fam, _ = model(emb_val, logits_val)
            val_loss = F.binary_cross_entropy_with_logits(
                val_out, labels_val,
                pos_weight=pos_weight[None, None, :]
            )
            
            # Compute validation AUC
            val_pred = val_out.reshape(-1, val_out.shape[-1]).numpy()
            val_true = labels_val.reshape(-1, labels_val.shape[-1]).numpy()
            try:
                val_auc = macro_auc_skip_empty(val_true, val_pred)
            except Exception:
                val_auc = 0.0
        
        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_auc"].append(val_auc)
        
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        
        if verbose and (epoch + 1) % 20 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}: train={loss.item():.4f} val={val_loss.item():.4f} "
                  f"auc={val_auc:.4f} lr={lr_now:.6f} wait={wait}")
        
        if wait >= cfg["patience"]:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
            break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    if verbose:
        print(f"  Training complete. Best val_loss={best_val_loss:.4f}")
        # Report fusion alpha distribution
        with torch.no_grad():
            alphas = torch.sigmoid(model.fusion_alpha).numpy()
            print(f"  Fusion alpha: mean={alphas.mean():.3f} min={alphas.min():.3f} max={alphas.max():.3f}")
            print(f"  Proto temperature: {F.softplus(model.proto_temp).item():.3f}")
    
    return model, history

print("ProtoSSM training functions defined.")