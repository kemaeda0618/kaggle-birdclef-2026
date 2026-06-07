"""exp040: exp038 base + NB4 stream に ResidualSSM (second-pass error correction) 追加.

Base: exp038 nb_blend.ipynb (w30-40-30, LB 0.949)
Changes:
- model cell: ResidualSSM class 定義追加 (BiSSMBlock 既存 class 流用、~50 行)
- train cell: ResSSM training (labeled SS first-pass residuals で fit、~80 行、in-NB ~20-30 sec)
- nb4-infer cell: ResSSM correction 適用 (l_blend += W * rs_correction、~5 行)
- Tucker SED / exp029 / blend: 完全に exp038 と同じ

期待:
- NB4 v11 standalone 0.926 → 0.927-0.929 (+0.001-0.003)
- blend (slot 1 で NB4 を使用) exp038 0.949 → 0.950-0.951 (+0.001-0.002)

Approach: surgical cell modification on generated NB ipynb (not chained patcher).
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
EXP038_NB_PATH = HERE.parent.parent / "exp038" / "notebook" / "nb_blend.ipynb"

# First make sure exp038 NB is generated
if not EXP038_NB_PATH.exists():
    print(f"[exp040] generating exp038 NB first ...")
    import subprocess
    subprocess.run(["python", str(EXP038_NB_PATH.parent / "_gen_nb_blend.py")], check=True)

src_nb = json.loads(EXP038_NB_PATH.read_text(encoding="utf-8"))
print(f"[exp040] read exp038 NB: {len(src_nb['cells'])} cells")


# ============================================================================
# Patch 1: model cell — append ResidualSSM class definition
# ============================================================================
RESIDUAL_MODEL_DEF = r"""

# === exp040: ResidualSSM (mtoshidesu Cell 7j 流) - second-pass error correction ===
# Input: emb (1536d) + first_pass_logits (234d) concat → correction (234d)
# Lightweight 2nd-pass、BiSSMBlock 既存 class 流用、~439K params、in-NB train ~20-30 sec
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
print("exp040: ResidualSSM defined")
"""


# ============================================================================
# Patch 2: train cell — append ResidualSSM training code at the end
# ============================================================================
RESIDUAL_TRAIN_CODE = r"""

# === exp040: Train ResidualSSM on labeled SS first-pass residuals ===
print("\n[exp040] Training ResidualSSM on labeled SS...")
_t0_rs = time.time()

# Compute first-pass predictions on labeled SS (single-forward across multi-seed, no TTA)
def _first_pass_single_lab(models_list, emb_t, scores_t, site_t, hour_t, prior_t):
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
    p_proto = _first_pass_single_lab(proto_models, emb_t, sc_t, site_t, hour_t, prior_t)
    p_mlp = _first_pass_single_lab(mlp_models, emb_t, sc_t, site_t, hour_t, prior_t)
    p_proto_c = p_proto.clamp(min=1e-7, max=1 - 1e-7)
    p_mlp_c = p_mlp.clamp(min=1e-7, max=1 - 1e-7)
    l_proto = torch.log(p_proto_c) - torch.log1p(-p_proto_c)
    l_mlp = torch.log(p_mlp_c) - torch.log1p(-p_mlp_c)
    l_blend = W_PROTO * l_proto + W_MLP * l_mlp
    fp_lab_chunks.append(l_blend.squeeze(0).cpu().numpy())
first_pass_lab = np.stack(fp_lab_chunks).astype(np.float32)
print(f"  first_pass_lab: {first_pass_lab.shape}")

fp_prob = 1.0 / (1.0 + np.exp(-np.clip(first_pass_lab, -30, 30)))
residuals_lab = (lab_labels_files - fp_prob).astype(np.float32)
print(f"  residuals: mean={residuals_lab.mean():+.4f}  std={residuals_lab.std():.4f}  abs_mean={np.abs(residuals_lab).mean():.4f}")

rs_model = ResidualSSM(d_input=lab_emb_files.shape[-1], d_scores=N_CLASSES,
                       d_model=64, d_state=8, n_classes=N_CLASSES)
print(f"  ResidualSSM params: {sum(p.numel() for p in rs_model.parameters()):,}")

rs_emb_t = torch.tensor(lab_emb_files, dtype=torch.float32)
rs_fp_t = torch.tensor(first_pass_lab, dtype=torch.float32)
rs_res_t = torch.tensor(residuals_lab, dtype=torch.float32)
rs_site_t = torch.tensor(lab_site_ids, dtype=torch.long)
rs_hour_t = torch.tensor(lab_hours, dtype=torch.long)

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

with torch.no_grad():
    all_corr_chk = rs_model(rs_emb_t, rs_fp_t, site_ids=rs_site_t, hours=rs_hour_t).numpy()
print(f"  correction magnitude: mean_abs={np.abs(all_corr_chk).mean():.4f}  max_abs={np.abs(all_corr_chk).max():.4f}")
"""


# ============================================================================
# Patch 3: nb4-infer cell — insert ResSSM correction after l_blend
# ============================================================================
RESIDUAL_INFER_PATCH_FROM = """        p_proto_c = p_proto.clamp(min=1e-7, max=1 - 1e-7)
        p_mlp_c = p_mlp.clamp(min=1e-7, max=1 - 1e-7)
        l_proto = torch.log(p_proto_c) - torch.log1p(-p_proto_c)
        l_mlp = torch.log(p_mlp_c) - torch.log1p(-p_mlp_c)
        l_blend = W_PROTO * l_proto + W_MLP * l_mlp"""

RESIDUAL_INFER_PATCH_TO = RESIDUAL_INFER_PATCH_FROM + """

        # === exp040: ResidualSSM correction (second-pass) ===
        with torch.no_grad():
            rs_corr = rs_model(emb_t, l_blend.float(),
                               site_ids=site_t, hours=hour_t)
            l_blend = l_blend + RESIDUAL_CORRECTION_WEIGHT * rs_corr"""


# Apply patches to cells
patched_count = {"model": 0, "train": 0, "nb4-infer": 0}
for c in src_nb["cells"]:
    if c["cell_type"] != "code":
        continue
    cid = c.get("id", "")
    src = "".join(c.get("source", []))

    if cid == "model":
        new_src = src + RESIDUAL_MODEL_DEF
        c["source"] = new_src.splitlines(keepends=True)
        if not c["source"][-1].endswith("\n"):
            c["source"][-1] = c["source"][-1]
        patched_count["model"] = 1
    elif cid == "train":
        new_src = src + RESIDUAL_TRAIN_CODE
        c["source"] = new_src.splitlines(keepends=True)
        patched_count["train"] = 1
    elif cid == "nb4-infer":
        if RESIDUAL_INFER_PATCH_FROM in src:
            new_src = src.replace(RESIDUAL_INFER_PATCH_FROM, RESIDUAL_INFER_PATCH_TO)
            c["source"] = new_src.splitlines(keepends=True)
            patched_count["nb4-infer"] = 1
        else:
            print(f"  WARNING: nb4-infer patch pattern not found")

# Sanity check
for k, v in patched_count.items():
    print(f"  patched {k}: {v}")

# ============================================================================
# Patch 4: clean stale references
# ============================================================================
# Header: rewrite cleanly from scratch (exp038 base のコメントを置換)
new_hdr = (
    "# exp040 = NB4 v11 + ResSSM (0.30) + Tucker SED (0.40) + exp029 (0.30)\n"
    "\n"
    "## 構成\n"
    "- **Slot 1**: NB4 v11 (ProtoSSM + MLPHead + multi-seed=5 + KD + Retrieval + E19 + FCS + Season prior + Mirror) **+ ResidualSSM (second-pass)**\n"
    "- **Slot 2**: Tucker SED 5-fold ONNX ensemble\n"
    "- **Slot 3**: exp029 R3 eca_nfnet_l1 distill (single fold)\n"
    "\n"
    "## exp040 (新規 vs exp038)\n"
    "- NB4 stream に **ResidualSSM 追加** (BiSSMBlock 既存 class 流用、in-NB train ~20-30 sec)\n"
    "- Inference: `l_blend += 0.30 * rs_model(emb, l_blend)` (logit-space correction)\n"
    "- 既存 PP / multi-seed / KD は全 keep\n"
    "\n"
    "## 期待 LB\n"
    "- exp038 base 0.949 → **0.950-0.951** (NB4 v11 standalone 0.926 + ResSSM +0.001-0.002 を blend で transfer)\n"
)
src_nb["cells"][0]["source"] = new_hdr.splitlines(keepends=True)

# Stale ref fixes in code cells
STALE_REPLACEMENTS = [
    # cell nb4-infer: "NB4 v7 inference" → "NB4 v11 + ResSSM inference"
    ("# === Stage A: NB4 v7 inference + konbu17 LinearHead ===",
     "# === Stage A: NB4 v11 + ResSSM inference ==="),
    # cell e29-infer: assertion message
    ('"exp038: exp029 ckpt not found',
     '"exp040: exp029 ckpt not found'),
    ('print(f"exp038 (exp029 ckpt) state dir:',
     'print(f"exp040 (exp029 ckpt) state dir:'),
    # cell blend: comment
    ("# exp038 ratio config (w30-40-30, exp029 boost from w35-40-25):",
     "# exp040 ratio config (w30-40-30 inherited from exp038):"),
    ("# exp038: NB4 0.35→0.30 (-0.05)",
     "# exp040 (inherited from exp038): NB4 0.35→0.30"),
    ("# exp038: exp029 0.25→0.30 (+0.05) — note var name kept",
     "# exp040 (inherited from exp038): exp029 0.25→0.30 — note var name kept"),
]
fix_count = 0
for c in src_nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    modified = False
    for old, new in STALE_REPLACEMENTS:
        if old in src:
            src = src.replace(old, new)
            modified = True
            fix_count += 1
    if modified:
        c["source"] = src.splitlines(keepends=True)
print(f"  applied {fix_count} stale-ref fixes")

# Write output
out_path = HERE / "nb_blend.ipynb"
out_path.write_text(json.dumps(src_nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Written: {out_path.name}")
print(f"  cells: {len(src_nb['cells'])}")
print(f"  size: {out_path.stat().st_size/1024:.1f} KB")
