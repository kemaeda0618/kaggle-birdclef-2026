"""Build exp130: exp110 stack + amphibian SPECIALIST surgical rank-blend (35 amphibian cols only).

Derived from exp110 (LB 0.952). Two surgical edits:
  1. Insert amphib_b0 (efficientnet_b0, torch) inference cell AFTER 'e29-infer'
     -> amph_df (row_id-indexed, 35 amphibian presence scores), aligned via shared audio_cache.
  2. Patch the 'blend' cell: right before submission.to_csv, rank-blend ONLY the 35 amphibian
     columns with amphib_b0 (alpha=0.5). Assert the 199 non-amphibian columns are bit-identical.

amphib mel == e29 mel (32k/2048/512/256/20/16000) so input is consistent. Uses torch+timm only
(no openvino) -> compatible with submit NB enable_internet=False.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path("experiment/exp110/notebook/nb_exp110_blend_30_40_30.ipynb")
DST = Path("experiment/exp130/notebook/nb_exp130_amphib_surgical.ipynb")
DST.parent.mkdir(parents=True, exist_ok=True)

nb = json.loads(SRC.read_text(encoding="utf-8"))

AMPH_CELL = """# === exp130: amphibian specialist (effb0, torch) inference — aligned with audio_cache ===
import json as _json, timm as _timm
_AMPH_DIR = next((Path(p) for p in [
    "/kaggle/input/birdclef2026-amphib-b0-ov",
    "/kaggle/input/datasets/maekeso/birdclef2026-amphib-b0-ov"] if Path(p).exists()), None)
print("[exp130] AMPH_DIR:", _AMPH_DIR)
_amph_meta = _json.load(open(_AMPH_DIR / "amphib_meta.json"))
AMP_LABELS = _amph_meta["labels"]                       # 35 labels, model output order
_amph_model = _timm.create_model("efficientnet_b0", pretrained=False, in_chans=1,
                                 num_classes=len(AMP_LABELS))
_amph_model.load_state_dict(torch.load(str(_AMPH_DIR / "amphib_b0.pth"), map_location="cpu"))
_amph_model.eval()
_amp_mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=2048, hop_length=512,
            n_mels=256, f_min=20, f_max=16000, power=2.0)
_amp_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
_amph_rows = []
with torch.no_grad():
    for _raw_60s in audio_cache:
        _ch = _raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
        _m = _amp_db(_amp_mel(torch.from_numpy(_ch)))            # [12,256,T]
        _mu = _m.mean((1, 2), keepdim=True); _sd = _m.std((1, 2), keepdim=True) + 1e-6
        _m = ((_m - _mu) / _sd).unsqueeze(1)                     # [12,1,256,T]
        _o = torch.sigmoid(_amph_model(_m)).cpu().numpy()        # [12,35]
        _amph_rows.append(_o.astype(np.float32))
_amph_flat = np.concatenate(_amph_rows, axis=0)                  # [n_rows,35] in all_row_ids order
amph_df = pd.DataFrame(_amph_flat, columns=AMP_LABELS)
amph_df["row_id"] = all_row_ids
amph_df = amph_df.set_index("row_id")
print(f"[exp130] amphib inference done: {amph_df.shape}, labels={len(AMP_LABELS)}")
"""

# 1) insert amphib cell after 'e29-infer'
ids = [c.get("id") for c in nb["cells"]]
assert "e29-infer" in ids, ids
pos = ids.index("e29-infer") + 1
amph_cell = {"cell_type": "code", "id": "exp130_amphib", "metadata": {},
             "execution_count": None, "outputs": [], "source": AMPH_CELL.splitlines(keepends=True)}
nb["cells"].insert(pos, amph_cell)

# 2) patch blend cell: inject surgical blend right before submission.to_csv
TOCSV = 'submission.to_csv("submission.csv", index=False)'
BLEND = '''# === exp130: amphibian surgical rank-blend (35 amphibian cols only; 199 untouched) ===
from scipy.stats import rankdata as _rd
ALPHA_AMPH = 0.5
_nonamp_cols = [c for c in PRIMARY_LABELS if c not in set(AMP_LABELS)]
_pre_nonamp = submission[_nonamp_cols].values.copy()
_amph_al = amph_df.reindex(submission["row_id"]).fillna(0.5).values   # [n_rows,35] submission order
_nrow = len(submission); _nb = 0
for _j, _lbl in enumerate(AMP_LABELS):
    if _lbl not in submission.columns:
        continue
    _rb = _rd(submission[_lbl].values) / _nrow
    _ra = _rd(_amph_al[:, _j]) / _nrow
    submission[_lbl] = ((1 - ALPHA_AMPH) * _rb + ALPHA_AMPH * _ra).astype(np.float32)
    _nb += 1
assert np.array_equal(submission[_nonamp_cols].values, _pre_nonamp), "exp130: non-amphibian cols changed!"
print(f"[exp130] amphibian surgical blend: {_nb} cols (alpha={ALPHA_AMPH}); {len(_nonamp_cols)} non-amph cols verified unchanged")
''' + TOCSV

patched = False
for c in nb["cells"]:
    if c.get("id") != "blend":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    assert TOCSV in src, "to_csv anchor not found in blend cell"
    src = src.replace(TOCSV, BLEND)
    c["source"] = src.splitlines(keepends=True)
    patched = True
    break
assert patched, "blend cell not found"

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {DST} ({DST.stat().st_size} bytes)")

# verify: compile the two edited/new cells
chk = json.loads(DST.read_text(encoding="utf-8"))
for c in chk["cells"]:
    if c.get("id") in ("exp130_amphib", "blend"):
        src = "".join(c["source"])
        clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln for ln in src.splitlines())
        try:
            compile(clean, f"<{c['id']}>", "exec"); print(f"  [OK] {c['id']} ({len(src.splitlines())} lines)")
        except SyntaxError as e:
            print(f"  [FAIL] {c['id']}: {e}")
print("cells:", len(chk["cells"]), "| amph cell at idx", [c.get("id") for c in chk["cells"]].index("exp130_amphib"))
