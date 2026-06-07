"""exp037: NB4 mtoshidesu 路線 (Level A) - ProtoSSM + MLP + ResidualSSM.

Base: exp028 (`_gen_nb_protossm_mirror.py`)
Changes (Level A):
  1. W_PROTO=1.0 → 0.5, W_MLP=0.0 → 0.5 (NB4 v7 baseline 50:50 復帰)
  2. FCS_POWER=0.4 → 0.05 (gentle, mtoshidesu 流)
  3. Per-class temperature: Texture taxa (Amphibia/Insecta) T=0.95, event taxa (Aves) T=1.10
  4. adaptive_delta_smooth: NB3 v21 流 (alpha = base × (1 - max_conf))、+0.010 旧 実証
  5. ResidualSSM (mtoshidesu Cell 21): in-NB train on labeled SS, second-pass error correction

期待 ΔLB: standalone +0.005-0.014 (exp027 base 0.918 → 0.92x-0.93x)
       blend (exp031) +0.001-0.004

Output: experiment/exp037/notebook/nb_protossm_resssm.ipynb
"""
import re as _re
from pathlib import Path

HERE = Path(__file__).parent
EXP028_GEN_PATH = HERE.parent.parent / "exp028" / "notebook" / "_gen_nb_protossm_mirror.py"
SRC = EXP028_GEN_PATH.read_text(encoding="utf-8")

print(f"[exp037] reading exp028 base: {EXP028_GEN_PATH}")
print(f"  base size: {len(SRC)} chars")

# ============================================================================
# Patch 1: W_PROTO=1.0 → 0.5, W_MLP=0.0 → 0.5
# ============================================================================
SRC = SRC.replace(
    "W_PROTO = 1.0\nW_MLP = 0.0",
    "W_PROTO = 0.5    # exp037: NB4 v7 baseline 50:50 復帰\nW_MLP = 0.5",
)

# ============================================================================
# Patch 2: FCS_POWER 0.4 → 0.05 (gentle)
# ============================================================================
SRC = SRC.replace(
    "FCS_POWER = 0.4",
    "FCS_POWER = 0.05    # exp037: gentle, mtoshidesu 流 (was 0.4 aggressive)",
)

# ============================================================================
# Patch 3: Per-class temperature - inject at end of TAXONOMY cell
# ============================================================================
TEMP_INJECT = """
# === exp037: Per-class temperature by taxonomy ===
# Texture taxa (Amphibia, Insecta) 連続発声 → sharper (T=0.95)
# Event taxa (Aves, Reptilia) 単発 → softer (T=1.10)
TEXTURE_TAXA = {"Amphibia", "Insecta"}
_cn_lookup = dict(zip(taxonomy["primary_label"].astype(str), taxonomy["class_name"]))
PER_CLASS_TEMP = np.ones(N_CLASSES, dtype=np.float32)
for ci, lbl in enumerate(PRIMARY_LABELS):
    cls = _cn_lookup.get(lbl, "Aves")
    if cls in TEXTURE_TAXA:
        PER_CLASS_TEMP[ci] = 0.95
    else:
        PER_CLASS_TEMP[ci] = 1.10
PER_CLASS_TEMP_T = torch.tensor(PER_CLASS_TEMP, dtype=torch.float32)
print(f"Per-class temperature: texture={int((PER_CLASS_TEMP < 1.0).sum())} (T=0.95), event={int((PER_CLASS_TEMP > 1.0).sum())} (T=1.10)")
# === end per-class temperature ==="""

SRC = SRC.replace(
    'print(f"Species: {N_CLASSES}, Mapped: {MAPPED_MASK.sum()}, Proxies: {len(proxy_map)}")',
    'print(f"Species: {N_CLASSES}, Mapped: {MAPPED_MASK.sum()}, Proxies: {len(proxy_map)}")' + TEMP_INJECT,
)

# ============================================================================
# Patch 4: ResidualSSM class definition - inject after ProtoSSM/MLPHead defs
# ============================================================================
RESIDUAL_MODEL_DEF = """
# === exp037: ResidualSSM (mtoshidesu Cell 21) - second-pass error correction ===
# Input: emb + first_pass_logit → predict residual = labels - sigmoid(first_pass)
# Lightweight 2nd-pass model trained on labeled SS only (~20s on CPU)
class ResidualSSM(nn.Module):
    def __init__(self, d_input=1536, d_scores=N_CLASSES,
                 d_model=64, d_state=8, n_classes=N_CLASSES,
                 dropout=0.1, n_sites=32, meta_dim=8):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, N_WINDOWS, d_model) * 0.02)
        self.ssm_block = BiSSMBlock(d_model, d_state, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, n_classes)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
    def forward(self, emb, first_pass, site_ids=None, hours=None):
        B, T, _ = emb.shape
        x = torch.cat([emb, first_pass], dim=-1)
        h = self.input_proj(x) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            s_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
            h_e = self.hour_emb(hours.clamp(0, 23))
            meta = self.meta_proj(torch.cat([s_e, h_e], dim=-1))
            h = h + meta.unsqueeze(1)
        h = self.norm(self.ssm_block(h))
        return self.output_head(h)

RESIDUAL_CORRECTION_WEIGHT = 0.30  # mtoshidesu 0.15-0.35 推奨
print("ResidualSSM defined")
# === end ResidualSSM ==="""

SRC = SRC.replace(
    'print("ProtoSSM + MLPHead defined.")',
    'print("ProtoSSM + MLPHead defined.")' + RESIDUAL_MODEL_DEF,
)

# ============================================================================
# Patch 5: ResidualSSM training cell - new cell after TRAIN
# ============================================================================
# Note: this cell uses single-forward (no TTA) for first-pass computation on labeled SS,
#       which is faster and sufficient for Residual SSM training target.
RESIDUAL_TRAIN_BODY = r"""# exp037: Train ResidualSSM on labeled SS (in-NB)
print()
print("[exp037] Training ResidualSSM on labeled SS...")
_t0_rs = time.time()

# Single-forward first-pass on labeled SS (no TTA, faster)
def _first_pass_single(models_list, emb_t, scores_t, site_t, hour_t, prior_t):
    probs = []
    for m in models_list:
        m.eval()
        with torch.no_grad():
            p = m(emb_t, scores_t, site_ids=site_t, hours=hour_t,
                  prior_logit=prior_t, lambda_prior=LAMBDA_PRIOR)
        probs.append(p)
    return torch.stack(probs, dim=0).mean(dim=0)

n_lab = lab_emb_files.shape[0]
fp_lab_chunks = []
for fi in range(n_lab):
    emb_t = torch.tensor(lab_emb_files[fi:fi+1], dtype=torch.float32)
    sc_t = torch.tensor(lab_scores_files[fi:fi+1], dtype=torch.float32)
    site_t = torch.tensor([lab_site_ids[fi]], dtype=torch.long)
    hour_t = torch.tensor([lab_hours[fi]], dtype=torch.long)
    prior_t = torch.tensor(lab_prior_files[fi:fi+1], dtype=torch.float32)
    p_proto = _first_pass_single(proto_models, emb_t, sc_t, site_t, hour_t, prior_t)
    p_mlp = _first_pass_single(mlp_models, emb_t, sc_t, site_t, hour_t, prior_t)
    p_proto_c = p_proto.clamp(min=1e-7, max=1 - 1e-7)
    p_mlp_c = p_mlp.clamp(min=1e-7, max=1 - 1e-7)
    l_proto = torch.log(p_proto_c) - torch.log1p(-p_proto_c)
    l_mlp = torch.log(p_mlp_c) - torch.log1p(-p_mlp_c)
    l_blend = W_PROTO * l_proto + W_MLP * l_mlp
    fp_lab_chunks.append(l_blend.squeeze(0).cpu().numpy())

first_pass_lab = np.stack(fp_lab_chunks).astype(np.float32)
print(f"  first_pass_lab: {first_pass_lab.shape}  in {time.time()-_t0_rs:.1f}s")

# Residual target: labels - sigmoid(first_pass)
fp_prob = 1.0 / (1.0 + np.exp(-np.clip(first_pass_lab, -30, 30)))
residuals_lab = (lab_labels_files - fp_prob).astype(np.float32)
print(f"  residuals: mean={residuals_lab.mean():+.4f}  std={residuals_lab.std():.4f}  abs_mean={np.abs(residuals_lab).mean():.4f}")

# Train Residual SSM
rs_model = ResidualSSM(d_input=lab_emb_files.shape[-1], d_scores=N_CLASSES,
                       d_model=64, d_state=8, n_classes=N_CLASSES)
print(f"  ResidualSSM params: {sum(p.numel() for p in rs_model.parameters()):,}")

rs_emb_t = torch.tensor(lab_emb_files, dtype=torch.float32)
rs_fp_t = torch.tensor(first_pass_lab, dtype=torch.float32)
rs_res_t = torch.tensor(residuals_lab, dtype=torch.float32)
rs_site_t = torch.tensor(lab_site_ids, dtype=torch.long)
rs_hour_t = torch.tensor(lab_hours, dtype=torch.long)

# 85/15 split for early stopping
rs_perm = torch.randperm(n_lab, generator=torch.Generator().manual_seed(42)).numpy()
rs_n_val = max(1, int(n_lab * 0.15))
rs_val_i = rs_perm[:rs_n_val]
rs_tr_i = rs_perm[rs_n_val:]

RS_EPOCHS = 30
RS_PATIENCE = 8
rs_opt = torch.optim.AdamW(rs_model.parameters(), lr=1e-3, weight_decay=1e-3)
rs_sched = torch.optim.lr_scheduler.OneCycleLR(
    rs_opt, max_lr=1e-3, epochs=RS_EPOCHS, steps_per_epoch=1,
    pct_start=0.1, anneal_strategy="cos"
)
best_rs_loss, best_rs_state, rs_wait = float("inf"), None, 0
for ep in range(RS_EPOCHS):
    rs_model.train()
    corr = rs_model(rs_emb_t[rs_tr_i], rs_fp_t[rs_tr_i],
                    site_ids=rs_site_t[rs_tr_i], hours=rs_hour_t[rs_tr_i])
    rs_loss = F.mse_loss(corr, rs_res_t[rs_tr_i])
    rs_opt.zero_grad(); rs_loss.backward()
    torch.nn.utils.clip_grad_norm_(rs_model.parameters(), 1.0)
    rs_opt.step(); rs_sched.step()
    rs_model.eval()
    with torch.no_grad():
        v_corr = rs_model(rs_emb_t[rs_val_i], rs_fp_t[rs_val_i],
                          site_ids=rs_site_t[rs_val_i], hours=rs_hour_t[rs_val_i])
        v_loss = F.mse_loss(v_corr, rs_res_t[rs_val_i]).item()
    if v_loss < best_rs_loss:
        best_rs_loss = v_loss
        best_rs_state = {k: v.clone() for k, v in rs_model.state_dict().items()}
        rs_wait = 0
    else:
        rs_wait += 1
    if rs_wait >= RS_PATIENCE:
        print(f"  ResidualSSM early stop ep {ep+1}")
        break
rs_model.load_state_dict(best_rs_state)
rs_model.eval()
print(f"  ResidualSSM trained: best val MSE={best_rs_loss:.6f} in {time.time()-_t0_rs:.1f}s")

# Sanity check on correction magnitude
with torch.no_grad():
    all_corr = rs_model(rs_emb_t, rs_fp_t, site_ids=rs_site_t, hours=rs_hour_t).numpy()
print(f"  correction magnitude: mean_abs={np.abs(all_corr).mean():.4f}  max_abs={np.abs(all_corr).max():.4f}")"""

# Inject Residual training cell into the cell list, after TRAIN cell.
# We do this by appending a new `cells.append(...)` line in the generator.
RESIDUAL_TRAIN_INJECT = f'''
# ── exp037: Residual SSM Training ──
RESIDUAL_TRAIN = """{RESIDUAL_TRAIN_BODY}"""
cells.append(code_cell("residual-train", RESIDUAL_TRAIN))
'''

SRC = SRC.replace(
    'cells.append(code_cell("train", TRAIN))',
    'cells.append(code_cell("train", TRAIN))' + RESIDUAL_TRAIN_INJECT,
)

# ============================================================================
# Patch 6: Apply ResidualSSM correction in INFERENCE cell
#          Insert BEFORE retrieval logit addition, AFTER l_blend computation.
# ============================================================================
RESIDUAL_INFER_APPLY = '''        # exp037: ResidualSSM correction (second-pass)
        with torch.no_grad():
            rs_corr = rs_model(emb_t, l_blend.float(),
                               site_ids=site_t, hours=hour_t)
            l_blend = l_blend + RESIDUAL_CORRECTION_WEIGHT * rs_corr
'''

SRC = SRC.replace(
    '        # Add retrieval logit (shared, applied once after blend)',
    RESIDUAL_INFER_APPLY + '\n        # Add retrieval logit (shared, applied once after blend)',
)

# ============================================================================
# Patch 7: Per-class temperature in INFERENCE - apply on l_blend before final sigmoid
# ============================================================================
SRC = SRC.replace(
    '        agg = torch.sigmoid(l_blend)',
    '        # exp037: per-class temperature (logit / T)\n'
    '        l_blend = l_blend / PER_CLASS_TEMP_T\n'
    '        agg = torch.sigmoid(l_blend)',
)

# ============================================================================
# Patch 8: Add adaptive_delta_smooth in SUBMISSION, after FCS, before season prior
# ============================================================================
ADAPTIVE_SMOOTH = """
# === exp037: adaptive_delta_smooth (NB3 v21、+0.010 旧 実証) ===
# alpha = base * (1 - max_conf) → confident peak 保護
ADAPTIVE_BASE_ALPHA = 0.20
_view2 = preds_array.reshape(-1, N_WINDOWS, N_CLASSES).copy()
for _t in range(1, N_WINDOWS - 1):
    _conf = _view2[:, _t, :].max(axis=-1, keepdims=True)
    _alpha = ADAPTIVE_BASE_ALPHA * (1.0 - _conf)
    _neigh = (_view2[:, _t - 1, :] + _view2[:, _t + 1, :]) / 2.0
    _view2[:, _t, :] = (1.0 - _alpha) * _view2[:, _t, :] + _alpha * _neigh
preds_array = np.clip(_view2, 0.0, 1.0).reshape(-1, N_CLASSES).astype(np.float32)
print(f"adaptive_delta_smooth applied (base_alpha={ADAPTIVE_BASE_ALPHA})")
# === end adaptive_delta_smooth ===

"""

SRC = SRC.replace(
    "# === Season prior MVP (v12: filename signal A) ===",
    ADAPTIVE_SMOOTH + "# === Season prior MVP (v12: filename signal A) ===",
)

# ============================================================================
# Patch 9: Update output ipynb name and header references
# ============================================================================
SRC = SRC.replace('nb_protossm_mirror.ipynb', 'nb_protossm_resssm.ipynb')
# Update header markdown title (the cell at line 60)
SRC = SRC.replace("exp028", "exp037")

print(f"[exp037] patches applied, exec'ing modified source ({len(SRC)} chars)")

# ============================================================================
# Execute modified source
# ============================================================================
exec(SRC, {"__file__": str(HERE / "_gen_nb_protossm_resssm_inner.py")})
