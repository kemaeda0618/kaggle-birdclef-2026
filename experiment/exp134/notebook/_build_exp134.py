"""Build exp134: exp110 stack + amphibian v2 (exp131) surgical rank-blend. Same design as exp130 but v2 model."""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path("experiment/exp110/notebook/nb_exp110_blend_30_40_30.ipynb")
DST = Path("experiment/exp134/notebook/nb_exp134_amphib_v2_surgical.ipynb")
DST.parent.mkdir(parents=True, exist_ok=True)
nb = json.loads(SRC.read_text(encoding="utf-8"))

AMPH_CELL = """# === exp134: amphibian specialist V2 (effb0, torch) inference — aligned with audio_cache ===
import json as _json, timm as _timm
_AMPH_DIR = next((Path(p) for p in [
    "/kaggle/input/birdclef2026-amphib-b0-v2",
    "/kaggle/input/datasets/maekeso/birdclef2026-amphib-b0-v2"] if Path(p).exists()), None)
print("[exp134] AMPH_DIR:", _AMPH_DIR)
_amph_meta = _json.load(open(_AMPH_DIR / "amphib_v2_meta.json"))
AMP_LABELS = _amph_meta["labels"]
_amph_model = _timm.create_model("efficientnet_b0", pretrained=False, in_chans=1, num_classes=len(AMP_LABELS))
_amph_model.load_state_dict(torch.load(str(_AMPH_DIR / "amphib_b0_v2.pth"), map_location="cpu"))
_amph_model.eval()
_amp_mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=2048, hop_length=512,
            n_mels=256, f_min=20, f_max=16000, power=2.0)
_amp_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
_amph_rows = []
with torch.no_grad():
    for _raw_60s in audio_cache:
        _ch = _raw_60s.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
        _m = _amp_db(_amp_mel(torch.from_numpy(_ch)))
        _mu = _m.mean((1, 2), keepdim=True); _sd = _m.std((1, 2), keepdim=True) + 1e-6
        _m = ((_m - _mu) / _sd).unsqueeze(1)
        _o = torch.sigmoid(_amph_model(_m)).cpu().numpy()
        _amph_rows.append(_o.astype(np.float32))
_amph_flat = np.concatenate(_amph_rows, axis=0)
amph_df = pd.DataFrame(_amph_flat, columns=AMP_LABELS)
amph_df["row_id"] = all_row_ids
amph_df = amph_df.set_index("row_id")
print(f"[exp134] amphib v2 inference done: {amph_df.shape}, labels={len(AMP_LABELS)}")
"""

ids = [c.get("id") for c in nb["cells"]]
assert "e29-infer" in ids, ids
nb["cells"].insert(ids.index("e29-infer") + 1,
    {"cell_type": "code", "id": "exp134_amphib", "metadata": {}, "execution_count": None, "outputs": [],
     "source": AMPH_CELL.splitlines(keepends=True)})

TOCSV = 'submission.to_csv("submission.csv", index=False)'
BLEND = '''# === exp134: amphibian V2 surgical rank-blend (35 amphibian cols only; 199 untouched) ===
from scipy.stats import rankdata as _rd
ALPHA_AMPH = 0.5
_nonamp_cols = [c for c in PRIMARY_LABELS if c not in set(AMP_LABELS)]
_pre_nonamp = submission[_nonamp_cols].values.copy()
_amph_al = amph_df.reindex(submission["row_id"]).fillna(0.5).values
_nrow = len(submission); _nb = 0
for _j, _lbl in enumerate(AMP_LABELS):
    if _lbl not in submission.columns:
        continue
    _rb = _rd(submission[_lbl].values) / _nrow
    _ra = _rd(_amph_al[:, _j]) / _nrow
    submission[_lbl] = ((1 - ALPHA_AMPH) * _rb + ALPHA_AMPH * _ra).astype(np.float32)
    _nb += 1
assert np.array_equal(submission[_nonamp_cols].values, _pre_nonamp), "exp134: non-amphibian cols changed!"
print(f"[exp134] amphibian V2 surgical blend: {_nb} cols (alpha={ALPHA_AMPH}); {len(_nonamp_cols)} non-amph unchanged")
''' + TOCSV

patched = False
for c in nb["cells"]:
    if c.get("id") != "blend": continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    assert TOCSV in src, "to_csv anchor not found"
    c["source"] = src.replace(TOCSV, BLEND).splitlines(keepends=True); patched = True; break
assert patched

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {DST}")
chk = json.loads(DST.read_text(encoding="utf-8"))
for c in chk["cells"]:
    if c.get("id") in ("exp134_amphib", "blend"):
        src = "".join(c["source"]); clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln for ln in src.splitlines())
        try: compile(clean, f"<{c['id']}>", "exec"); print(f"  [OK] {c['id']}")
        except SyntaxError as e: print(f"  [FAIL] {c['id']}: {e}")
