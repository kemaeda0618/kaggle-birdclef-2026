"""Build exp094 R1 NB: exp017 R1 + Ali's KL distillation (T=1, low alpha).

Concept (from BirdCLEF 2026 discussion #694479):
- Ali Ozan (3rd place) reported "very good results" with:
  - Temperature T=1 (no scaling, Perch logits stay hot)
  - Slightly lower KL divergence weight (auxiliary signal only)
  - Applied "directly from the first iteration" (= R1 stage)
- Tucker (Reason 2): Perch logits LB ~0.82, embedding LB ~0.90
  - Direct logit prediction = 0.82 ceiling
  - But Ali's low weight = auxiliary only, doesn't bind student to 0.82

This NB adds:
- Perch logit-level KL distillation (Hinton style with T=1)
- ALPHA_LOGIT = 0.1 (slightly lower)
- Mapped species only (~80 of 234 that exist in Perch vocab)
- Keeps existing Tucker MSE embedding distill (ALPHA_EMB=1.0)

Training cost: 5-6h on Colab L4 (eca_nfnet_l0, 25 ep, single fold).
Expected: R1-Ali LB ~0.90-0.92 (vs original R1 ~0.89), cascading benefit through R2/R3 if continued.
"""
import json, io, sys, shutil
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp017\notebook\nb_train.ipynb")
DST = Path(__file__).with_name("nb_train_ali.ipynb")

print(f"[clone] {SRC.name} -> {DST.name}")
shutil.copy(SRC, DST)

nb = json.loads(DST.read_text(encoding="utf-8"))

# ============================================================
# PATCH 1: Header (insert new markdown cell at top, or modify existing)
# ============================================================
HEADER_MD = """# exp094 — R1 with Ali Ozan style KL distillation (T=1, low alpha)

## Concept

BC2026 discussion #694479 (Tucker SED Baseline) で **Ali Ozan (3rd place)** が報告:
> "very good results today when I used **temperature 1** and a slightly **lower KL divergence weight** directly from the first iteration"

Tucker は **MSE embedding distillation** (LB 0.90 ceiling) を支持、Ali の logit KL に懐疑的:
> "Perch logits raw LB 0.82、embedding clustering 0.90"

但し Ali は **3rd place 圏で実証**、Tucker は **803rd**。実績ベースで Ali path は valid。

## 改修内容 (exp017 R1 base + Ali 流追加)

```python
# 既存 (Tucker 流、保持):
total_loss = BCE_loss + ALPHA_EMB * MSE(student_emb, perch_emb)

# ★ NEW (Ali 流追加):
+ ALPHA_LOGIT * KL(student_logit/T, perch_logit/T)  # T=1, ALPHA_LOGIT=0.1
```

### 設定
- `T_DISTILL = 1.0` (no temperature scaling、Ali spec)
- `ALPHA_LOGIT = 0.1` (slightly lower KL weight、Ali spec)
- `ALPHA_EMB = 1.0` (既存維持)
- Mapped 種 (Perch logit 持つ ~80 種) のみ logit KL 適用

### 理論的根拠
- R1 の main signal は **train.csv (ground truth)** = 0.82 ceiling 問題なし
- Perch logit は **auxiliary signal (low weight)** = 軽い参考意見
- Main BCE が dominant、Perch logit は calibration の補助
- → 0.82 ceiling に縛られず、Perch dark knowledge を取り入れる

## 出力

- ckpt: `/content/drive/MyDrive/kaggle/birdclef2026/output/exp094/fold0/`
- Kaggle Dataset push: `birdclef2026-exp094-ali-r1` (training 後)

## 想定
- ~5-6h on Colab L4 (25 ep、batch=192)
- 期待 LB: 0.90-0.92 (vs orig R1 ~0.89)
- 続いて R2-Ali、R3-Ali を NS iter で派生可能 (cascading効果)

---
"""

# Insert as new cell 0 (header markdown)
new_header_cell = {
    "cell_type": "markdown",
    "id": "hdr_ali",
    "metadata": {},
    "source": HEADER_MD.splitlines(keepends=True),
}
nb["cells"].insert(0, new_header_cell)
print("[OK] Header markdown cell inserted")

# ============================================================
# PATCH 2: Config cell - Add Ali params + Mapped indices
# ============================================================
# Insert these definitions after ALPHA_DISTILL = 1.0
CONFIG_OLD = "ALPHA_DISTILL = 1.0"
CONFIG_NEW = """ALPHA_DISTILL = 1.0
# ★ exp094: Ali Ozan style KL distillation (BC2026 discussion #694479)
USE_ALI_KD     = True       # Enable Hinton-style KL distillation
T_DISTILL      = 1.0        # Ali spec: temperature=1 (no scaling)
ALPHA_LOGIT    = 0.1        # Ali spec: slightly lower KL weight
PERCH_META_DIR = LOCAL_DATA / "perch-meta"   # for lbl_to_bc.csv (Mapped index map)"""

# ============================================================
# PATCH 3: Config cell - Add lbl_to_bc loading + MAPPED_POS construction
# ============================================================
# Insert after backbone print
SETUP_OLD = 'print(f"Max runtime: {MAX_RUNTIME_SEC/3600:.1f}h")'
SETUP_NEW = '''print(f"Max runtime: {MAX_RUNTIME_SEC/3600:.1f}h")

# === exp094: Mapped/Unmapped species indices (for Ali KL distillation) ===
if USE_ALI_KD:
    _lbl2bc_path_candidates = [
        PERCH_META_DIR / "lbl_to_bc.csv",
        LOCAL_DATA / "perch-meta-bc2026" / "lbl_to_bc.csv",
    ]
    _lbl2bc_path = next((p for p in _lbl2bc_path_candidates if p.exists()), None)
    assert _lbl2bc_path is not None, f"lbl_to_bc.csv not found in: {_lbl2bc_path_candidates}"
    print(f"[ALI KD] lbl_to_bc.csv: {_lbl2bc_path}")
    _lbl2bc_df = pd.read_csv(_lbl2bc_path)
    # Expect columns: primary_label, bc_index (or label, bc_idx)
    if "primary_label" in _lbl2bc_df.columns and "bc_index" in _lbl2bc_df.columns:
        _lbl2bc = _lbl2bc_df.set_index("primary_label")["bc_index"]
    else:
        _lbl2bc = _lbl2bc_df.set_index(_lbl2bc_df.columns[0])[_lbl2bc_df.columns[1]]
    NO_LABEL = -1
    # PRIMARY_LABELS will be set later (after taxonomy load) - we build BC_INDICES then
    print(f"[ALI KD] T_DISTILL={T_DISTILL}, ALPHA_LOGIT={ALPHA_LOGIT}")'''

# ============================================================
# PATCH 4: PerchTeacher class - Add predict_logits method
# ============================================================
TEACHER_OLD = '''    @torch.no_grad()
    def embed(self, waveforms_5s):
        wav_np = waveforms_5s.cpu().numpy().astype(np.float32)
        results = self.session.run(None, {self.input_name: wav_np})
        return torch.from_numpy(results[self._embed_idx]).float()'''

TEACHER_NEW = '''    @torch.no_grad()
    def embed(self, waveforms_5s):
        wav_np = waveforms_5s.cpu().numpy().astype(np.float32)
        results = self.session.run(None, {self.input_name: wav_np})
        return torch.from_numpy(results[self._embed_idx]).float()

    @torch.no_grad()
    def embed_and_logits(self, waveforms_5s):
        """★ exp094 NEW: Return both embedding (1536) and logits (14795).

        For Ali Ozan style KL distillation: need Perch logit head output
        for Mapped species (~80 of 234 that exist in Perch vocab).
        """
        wav_np = waveforms_5s.cpu().numpy().astype(np.float32)
        results = self.session.run(None, {self.input_name: wav_np})
        emb = torch.from_numpy(results[self._embed_idx]).float()
        # Find label head output (14795-d, the one that's NOT 1536 nor matches input shape)
        logits = None
        for i, arr in enumerate(results):
            if i == self._embed_idx: continue
            if hasattr(arr, "shape") and len(arr.shape) == 2 and arr.shape[-1] > 10000:
                logits = torch.from_numpy(arr).float()
                break
        if logits is None:
            # Fallback: try last output
            logits = torch.from_numpy(results[-1]).float()
        return emb, logits'''

# ============================================================
# PATCH 5: Cell that defines PRIMARY_LABELS - add BC_INDICES build after it
# ============================================================
# We need to find where PRIMARY_LABELS / l2i is built (from taxonomy)
PRIMARY_OLD = "l2i = {l: i for i, l in enumerate(PRIMARY_LABELS)}"
PRIMARY_NEW = '''l2i = {l: i for i, l in enumerate(PRIMARY_LABELS)}

# === exp094: Build BC_INDICES / MAPPED_POS after PRIMARY_LABELS is defined ===
if USE_ALI_KD:
    BC_INDICES = np.array([
        int(_lbl2bc.loc[c]) if c in _lbl2bc.index else NO_LABEL
        for c in PRIMARY_LABELS
    ], dtype=np.int32)
    MAPPED_MASK = BC_INDICES != NO_LABEL
    MAPPED_POS_NP = np.where(MAPPED_MASK)[0].astype(np.int64)
    MAPPED_BC_IDX_NP = BC_INDICES[MAPPED_MASK].astype(np.int64)
    MAPPED_POS = torch.from_numpy(MAPPED_POS_NP).long().to(device)
    MAPPED_BC_IDX = torch.from_numpy(MAPPED_BC_IDX_NP).long().to(device)
    print(f"[ALI KD] Mapped: {MAPPED_MASK.sum()} / {NUM_CLASSES} species have a Perch logit")
    print(f"[ALI KD] MAPPED_POS shape: {MAPPED_POS.shape}, MAPPED_BC_IDX shape: {MAPPED_BC_IDX.shape}")'''

# ============================================================
# PATCH 6: Training loop - replace embed() call with embed_and_logits, add KL loss
# ============================================================
LOOP_OLD = '''                    perch_emb = perch_teacher.embed(wav_5s).to(device)
                distill_loss = F.mse_loss(distill_emb, perch_emb)
                loss = cls_loss + ALPHA_DISTILL * distill_loss
            else:
                distill_loss = torch.tensor(0.0, device=device)
                loss = cls_loss'''

LOOP_NEW = '''                    if USE_ALI_KD:
                        perch_emb, perch_logits = perch_teacher.embed_and_logits(wav_5s)
                        perch_emb = perch_emb.to(device)
                        perch_logits = perch_logits.to(device)
                    else:
                        perch_emb = perch_teacher.embed(wav_5s).to(device)
                        perch_logits = None

                # === Tucker style: MSE on embedding (既存維持) ===
                distill_loss = F.mse_loss(distill_emb, perch_emb)

                # === ★ exp094: Ali style KL distillation on Perch logits ===
                if USE_ALI_KD and perch_logits is not None:
                    # Soft target from Perch (T=1, no scaling per Ali spec)
                    perch_mapped_logits = perch_logits.index_select(1, MAPPED_BC_IDX)   # (B, n_mapped)
                    soft_target = torch.sigmoid(perch_mapped_logits / T_DISTILL).detach()
                    # Student logits for mapped species (T=1 too)
                    student_mapped_logits = clip_logits.index_select(1, MAPPED_POS)     # (B, n_mapped)
                    soft_pred_logits = student_mapped_logits / T_DISTILL
                    # Multi-label KL = BCE(sigmoid(student/T), sigmoid(perch/T)) * T^2
                    distill_logit_loss = F.binary_cross_entropy_with_logits(
                        soft_pred_logits, soft_target
                    ) * (T_DISTILL * T_DISTILL)
                else:
                    distill_logit_loss = torch.tensor(0.0, device=device)

                loss = (cls_loss
                        + ALPHA_DISTILL * distill_loss
                        + ALPHA_LOGIT * distill_logit_loss)
            else:
                distill_loss = torch.tensor(0.0, device=device)
                distill_logit_loss = torch.tensor(0.0, device=device)
                loss = cls_loss'''

# ============================================================
# PATCH 7: Training loop print - add ali_kd loss
# ============================================================
PRINT_OLD = '''            print(f"    ep{epoch+1:02d} batch {batch_idx:4d}/{N_STEPS_PER_EP}  "
                  f"loss={loss.item():.4f}  cls={cls_loss.item():.4f}  "
                  f"dist={distill_loss.item():.4f}  lr={cur_lr:.2e}", flush=True)'''

PRINT_NEW = '''            print(f"    ep{epoch+1:02d} batch {batch_idx:4d}/{N_STEPS_PER_EP}  "
                  f"loss={loss.item():.4f}  cls={cls_loss.item():.4f}  "
                  f"emb={distill_loss.item():.4f}  "
                  f"kd={distill_logit_loss.item():.4f}  "
                  f"lr={cur_lr:.2e}", flush=True)'''

# ============================================================
# PATCH 8: Drive output path - rename to exp094
# ============================================================
# These patches will rename Drive paths from exp017 to exp094 to avoid overwrite
DRIVE_PATCHES = [
    ('DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/kaggle/birdclef2026/output/exp017")',
     'DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/kaggle/birdclef2026/output/exp094")'),
    ('"birdclef2026 exp017"', '"birdclef2026 exp094 ali r1"'),
    ('"birdclef2026-exp017-r1"', '"birdclef2026-exp094-ali-r1"'),
]

# ============================================================
# Apply all patches
# ============================================================
patches = [
    (CONFIG_OLD, CONFIG_NEW),
    (SETUP_OLD, SETUP_NEW),
    (TEACHER_OLD, TEACHER_NEW),
    (PRIMARY_OLD, PRIMARY_NEW),
    (LOOP_OLD, LOOP_NEW),
    (PRINT_OLD, PRINT_NEW),
] + DRIVE_PATCHES

print(f"\nApplying {len(patches)} code patches...")
all_src_before = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
applied = 0
missed = []
for old, new in patches:
    found_in = 0
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if old in src:
            new_src = src.replace(old, new)
            c["source"] = new_src.splitlines(keepends=True)
            found_in += 1
            break
    if found_in > 0:
        applied += 1
        label = old[:60].replace("\n", " | ")
        print(f"  [OK] {label!r}")
    else:
        missed.append(old[:80])
        print(f"  [MISS] {old[:60]!r}")

print(f"\nTotal applied: {applied}/{len(patches)}, missed: {len(missed)}")

# ============================================================
# Save
# ============================================================
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nSaved: {DST}")

# ============================================================
# Verify
# ============================================================
all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
print("\n=== Verification ===")
checks = [
    ("Header (exp094 + Ali)", "exp094 — R1 with Ali Ozan" in all_src),
    ("USE_ALI_KD = True", "USE_ALI_KD     = True" in all_src),
    ("T_DISTILL = 1.0", "T_DISTILL      = 1.0" in all_src),
    ("ALPHA_LOGIT = 0.1", "ALPHA_LOGIT    = 0.1" in all_src),
    ("embed_and_logits method", "def embed_and_logits" in all_src),
    ("MAPPED_POS construction", "MAPPED_POS = torch.from_numpy" in all_src),
    ("MAPPED_BC_IDX construction", "MAPPED_BC_IDX = torch.from_numpy" in all_src),
    ("KL distill loss in training", "distill_logit_loss = F.binary_cross_entropy_with_logits" in all_src),
    ("Loss combined (3 terms)", "+ ALPHA_LOGIT * distill_logit_loss" in all_src),
    ("KL printed in training log", "kd={distill_logit_loss.item()" in all_src),
    ("Drive path -> exp094", "/output/exp094" in all_src),
    ("Existing MSE emb distill intact", "F.mse_loss(distill_emb, perch_emb)" in all_src),
    ("Existing ALPHA_DISTILL = 1.0 intact", "ALPHA_DISTILL = 1.0" in all_src),
    ("Existing BACKBONE eca_nfnet_l0 intact", 'BACKBONE = "eca_nfnet_l0"' in all_src),
]
all_ok = True
for name, ok in checks:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")
    if not ok: all_ok = False

# ============================================================
# Compile check
# ============================================================
print("\n=== Compile check ===")
n_ok = n_err = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                       for ln in src.splitlines())
    try:
        compile(clean, f"<cell-{i}>", "exec")
        n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  [FAIL] cell {i}: {e}")
print(f"Result: {n_ok} OK, {n_err} ERRORS")

if all_ok and n_err == 0 and applied == len(patches):
    print("\n[OK] exp094 NB ready for Colab L4 training")
else:
    print(f"\n[WARN] Issues - review missed patches: {missed[:3]}")
