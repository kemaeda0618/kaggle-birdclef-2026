"""exp127: Amphibia-ONLY full diagnostic — run EVERY Amphibia XC clip (no cap).

Difference vs exp126:
  - cell1 glob narrowed to /kaggle/input/datasets/** (XC datasets only) -> fast (skip 9-min full walk)
  - cell3 keeps ONLY taxon==Amphibia, ALL clips (no MAX_PER_SP, no common cap)
  - keeps in_train flag so we see BOTH true-OOF and memorized (in-train) clips, split in report
  - cell6 prints per-species for ALL Amphibia species, OOF vs in-train side by side

Amphibia clean (OOF) = 35 clips / 11 species (from exp126 log); plus in-train clips for completeness.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path("experiment/exp127/notebook/nb_exp127_amphibia_all.ipynb")
NB.parent.mkdir(parents=True, exist_ok=True)


def cell(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}


C1 = """# === exp127 cell1: imports + FAST glob (XC datasets only) ===
import os, sys, glob, re, time
from pathlib import Path
import numpy as np, pandas as pd

WHEEL = sorted(glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True))
try:
    import openvino as ov
except ImportError:
    wd = str(Path(WHEEL[0]).parent)
    !pip install -q --no-deps {wd}/openvino-*.whl {wd}/openvino_telemetry-*.whl
    import openvino as ov
print("openvino", ov.__version__)

# XC datasets mount under /kaggle/input/datasets/... and are all .mp3 -> narrow glob (fast)
xc_audio = glob.glob("/kaggle/input/datasets/**/*.mp3", recursive=True)
if not xc_audio:  # fallback: any mp3 anywhere
    xc_audio = glob.glob("/kaggle/input/**/*.mp3", recursive=True)
print(f"XC mp3 files: {len(xc_audio)}")
print("sample:", xc_audio[0].replace("/kaggle/input/", "") if xc_audio else "NONE")
"""

C2 = """# === exp127 cell2: map file -> species + in_train flag ===
def ff(c, m):
    for p in c:
        p = Path(p)
        if p.exists() and (list(p.rglob(m)) or (p / m).exists()):
            return p
    return None
COMP = ff(["/kaggle/input/competitions/birdclef-2026", "/kaggle/input/birdclef-2026"], "taxonomy.csv")
print("COMP:", COMP)

tax = pd.read_csv(COMP / "taxonomy.csv")
train = pd.read_csv(COMP / "train.csv")
ss = pd.read_csv(COMP / "sample_submission.csv")
LABELS = ss.columns[1:].tolist(); L2I = {l: i for i, l in enumerate(LABELS)}; NC = len(LABELS)

label_set = set(str(x) for x in tax["primary_label"])
l2t = dict(zip(tax["primary_label"].astype(str), tax["class_name"].astype(str)))
l2sci = dict(zip(tax["primary_label"].astype(str), tax["scientific_name"].astype(str)))

def norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
sci2label = {norm(r["scientific_name"]): str(r["primary_label"]) for _, r in tax.iterrows()}
com2label = {norm(r["common_name"]): str(r["primary_label"]) for _, r in tax.iterrows()}

def xc_id(s):
    m = re.search(r"XC\\s*0*([0-9]{3,})", str(s), flags=re.I)
    return m.group(1) if m else None
train_xc_ids = set(filter(None, (xc_id(f) for f in train["filename"])))

def infer_species(path):
    parts = [Path(path).stem] + list(Path(path).parts)
    for raw in parts:
        if str(raw) in label_set: return str(raw)
        n = norm(raw)
        if n in sci2label: return sci2label[n]
        if n in com2label: return com2label[n]
    pp = Path(path).parts
    for a, b in zip(pp, pp[1:]):
        n = norm(a + b)
        if n in sci2label: return sci2label[n]
    return None

rows = []
for p in xc_audio:
    sp = infer_species(p); rid = xc_id(p)
    if sp is None: continue
    rows.append({"path": p, "species": sp, "rec_id": rid,
                 "in_train": (rid is not None and rid in train_xc_ids),
                 "taxon": l2t.get(sp, "?")})
xc = pd.DataFrame(rows)
print(f"matched: {len(xc)} clips, species {xc['species'].nunique()}")
print("taxon counts:\\n", xc.groupby("taxon")["path"].count())
"""

C3 = """# === exp127 cell3: keep ONLY Amphibia, ALL clips (no cap) ===
train["primary_label"] = train["primary_label"].astype(str)
train_count = train["primary_label"].value_counts().to_dict()
def tier(n):
    if n == 0: return "ghost"
    if n <= 5: return "very_rare"
    if n <= 20: return "rare"
    if n <= 50: return "low"
    return "common"

amp = xc[xc["taxon"] == "Amphibia"].copy()
amp["train_count"] = amp["species"].map(lambda s: train_count.get(s, 0))
amp["tier"] = amp["train_count"].map(tier)
amp["sci"] = amp["species"].map(l2sci)
DIAG = amp.sort_values(["species", "in_train"]).reset_index(drop=True)
print(f"=== ALL Amphibia clips: {len(DIAG)} (OOF {int((~DIAG['in_train']).sum())} + in_train {int(DIAG['in_train'].sum())}) ===")
print(f"Amphibia species in XC: {DIAG['species'].nunique()}")
print("\\nper-species clip counts (OOF / in_train):")
g = DIAG.groupby(["species", "sci", "tier", "train_count"])["in_train"].agg(
    n_total="count", n_in_train="sum")
g["n_oof"] = g["n_total"] - g["n_in_train"]
print(g.reset_index()[["species","sci","tier","train_count","n_oof","n_in_train"]].to_string(index=False))
"""

C4 = """# === exp127 cell4: load OV streams ===
import torch, torchaudio, soundfile as sf, librosa
SR=32000; N_MELS=256; N_FFT=2048; HOP=512; FMIN=20; FMAX=16000; TOP_DB=80
WIN=SR*5; N_WIN_CAP=8
def sig(x): return 1/(1+np.exp(-np.clip(x, -50, 50)))

TUCK = ff(["/kaggle/input/notebooks/maekeso/birdclef2026-tucker-sed-ov",
           "/kaggle/input/birdclef2026-tucker-sed-ov"], "sed_fold0.xml")
E106 = ff(["/kaggle/input/notebooks/maekeso/birdclef2026-e106-3fold-ov",
           "/kaggle/input/birdclef2026-e106-3fold-ov"], "exp106_fold0.xml")
core = ov.Core()
tuck_models = [core.compile_model(str(p), "CPU") for p in sorted(TUCK.glob("sed_fold*.xml"))]
e106_models = [core.compile_model(str(E106 / f"exp106_fold{f}.xml"), "CPU") for f in [0, 1, 2]]
print("tucker", len(tuck_models), "e106", len(e106_models))

mt = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=N_FFT, hop_length=HOP,
                                          n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0)
dt = torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)
def mel_torch(chs):
    w = torch.from_numpy(np.stack(chs).astype(np.float32)); m = dt(mt(w))
    mu = m.mean((1, 2), keepdim=True); sd = m.std((1, 2), keepdim=True) + 1e-6
    return ((m - mu) / sd).unsqueeze(1).numpy().astype(np.float32)
def mel_lib(chs):
    out = []
    for x in chs:
        s = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                            n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
        s = librosa.power_to_db(s, top_db=TOP_DB); s = (s - s.mean()) / (s.std() + 1e-6)
        out.append(s)
    return np.stack(out)[:, None].astype(np.float32)
def run(model, mels):
    out = []
    for b in range(0, len(mels), 24):
        o = model(mels[b:b+24]); clip = o[model.outputs[0]]; fr = o[model.outputs[1]].max(1)
        out.append((0.5*sig(clip) + 0.5*sig(fr)).astype(np.float32))
    return np.concatenate(out)
def load_windows(path):
    try:
        w, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        w, sr = librosa.load(str(path), sr=SR, mono=True)
    if getattr(w, "ndim", 1) > 1: w = w.mean(1)
    if sr != SR: w = librosa.resample(w, orig_sr=sr, target_sr=SR)
    if len(w) < WIN:
        w = np.concatenate([w, np.zeros(WIN - len(w), dtype=np.float32)])
    nwin = min(N_WIN_CAP, max(1, len(w) // WIN))
    idx = np.linspace(0, max(0, len(w) - WIN), nwin).astype(int)
    return np.stack([w[i:i+WIN] for i in idx]).astype(np.float32)
"""

C5 = """# === exp127 cell5: inference on every Amphibia clip ===
recs = []; t0 = time.time()
for k, r in DIAG.iterrows():
    try:
        chs = load_windows(r["path"])
    except Exception:
        continue
    ML = mel_lib(chs); MT = mel_torch(chs)
    pt = np.mean([run(m, ML) for m in tuck_models], axis=0)
    pe = np.mean([run(m, MT) for m in e106_models], axis=0)
    pc = (0.5 * pt + 0.5 * pe).max(0)
    ci = L2I[r["species"]]
    rank = int((pc > pc[ci]).sum()) + 1
    recs.append({"species": r["species"], "sci": r["sci"], "tier": r["tier"],
                 "train_count": int(r["train_count"]), "in_train": bool(r["in_train"]),
                 "rec_id": r["rec_id"], "p_true": float(pc[ci]), "rank": rank,
                 "top1": int(rank == 1), "top5": int(rank <= 5),
                 "argmax_label": LABELS[int(pc.argmax())]})
    if (k + 1) % 25 == 0: print(f"  {k+1}/{len(DIAG)} ({time.time()-t0:.0f}s)")
D = pd.DataFrame(recs)
D.to_csv("/kaggle/working/exp127_amphibia_all.csv", index=False)
print(f"scored {len(D)} clips in {time.time()-t0:.0f}s")
"""

C6 = """# === exp127 cell6: per-species report (OOF vs in_train) ===
pd.set_option("display.max_rows", 200); pd.set_option("display.width", 200)

def summ(df):
    return df.groupby(["species","sci","tier","train_count"]).agg(
        n=("rank","count"), top1=("top1","mean"), top5=("top5","mean"),
        p_true=("p_true","mean"), med_rank=("rank","median")).reset_index()

print("=== ALL Amphibia, OOF (true generalization) — per species ===")
oof = summ(D[~D["in_train"]]).sort_values(["top5","p_true"])
print(oof.to_string(index=False))

print("\\n=== ALL Amphibia, IN-TRAIN (memorized, inflated) — per species ===")
itr = summ(D[D["in_train"]]).sort_values(["top5","p_true"])
print(itr.to_string(index=False) if len(itr) else "  (none)")

print("\\n=== OOF vs in_train summary ===")
for name, sub in [("OOF", D[~D["in_train"]]), ("in_train", D[D["in_train"]])]:
    if len(sub):
        print(f"  {name:9s}: clips={len(sub):4d} species={sub['species'].nunique():3d} "
              f"top1={sub['top1'].mean():.3f} top5={sub['top5'].mean():.3f} "
              f"p_true={sub['p_true'].mean():.3f} med_rank={sub['rank'].median():.0f}")

print("\\n=== every OOF clip (full dump, worst rank first) ===")
cols = ["species","sci","tier","train_count","in_train","p_true","rank","argmax_label"]
print(D[~D["in_train"]].sort_values("rank", ascending=False)[cols].to_string(index=False))
"""

CELLS = [(C1, "glob"), (C2, "map"), (C3, "amp"), (C4, "load"), (C5, "infer"), (C6, "report")]
nb = {"cells": [cell(s, c) for s, c in CELLS],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NB}")
for s, c in CELLS:
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln for ln in s.splitlines())
    try:
        compile(clean, f"<{c}>", "exec"); print(f"  [OK] {c}")
    except SyntaxError as e:
        print(f"  [FAIL] {c}: {e}")
