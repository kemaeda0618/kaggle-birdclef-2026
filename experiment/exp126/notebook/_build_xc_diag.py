"""exp126: External XC diagnostic — can our model discriminate rare species on UNSEEN data?

True OOF diagnostic:
  - Mount 3 XC API-download datasets (maekeso/birdclef2026-xc-api-dl-part1..3)
  - Self-discover their structure on Kaggle (cell 1) -> we don't know layout from here
  - Map each XC file -> target species (taxon_id / scientific_name match)
  - EXCLUDE any XC recording already in train.csv (contamination guard -> true OOF only)
  - Run Tucker 5-fold OV + exp106 3-fold OV on the clips
  - For each clip: rank of the TRUE species among 234, top-1 hit, p_true
  - Aggregate by signal tier (ghost / very_rare / rare / common)
  -> answers: does the model GENERALIZE to rare species, or only memorize train?

Note: NB4/Perch-proto stream is EXCLUDED (speed). It is the stream that most helps
ghost/rare via Perch embeddings, so if rare recall is low here, NB4 may still rescue.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path("experiment/exp126/notebook/nb_exp126_xc_diag.ipynb")
NB.parent.mkdir(parents=True, exist_ok=True)


def cell(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}


C1 = """# === exp126 cell1: imports + DISCOVER the 3 XC datasets ===
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

AUDIO_EXT = (".ogg", ".mp3", ".wav", ".flac", ".m4a")

# glob every audio file under /kaggle/input that looks like an XC download dataset
all_audio = []
for ext in AUDIO_EXT:
    all_audio += glob.glob(f"/kaggle/input/**/*{ext}", recursive=True)
# keep only files whose path mentions an xc dataset (defensive: also keep anything not in competition dir)
xc_audio = [p for p in all_audio if "xc-api-dl" in p or "xc_api_dl" in p.lower()]
if not xc_audio:
    # fallback: any audio NOT inside the competition dir
    xc_audio = [p for p in all_audio if "birdclef-2026/train" not in p and "competitions" not in p]
print(f"total audio under /kaggle/input: {len(all_audio)}")
print(f"XC-dataset audio files: {len(xc_audio)}")

# show the top-level input dirs and a few sample paths so we LEARN the structure
roots = sorted(set(p.split('/kaggle/input/')[1].split('/')[0] for p in xc_audio))
print("\\ntop-level input dirs holding XC audio:", roots)
print("\\nsample XC paths:")
for p in xc_audio[:25]:
    print("  ", p.replace("/kaggle/input/", ""))
print("\\nfile extensions:", sorted(set(Path(p).suffix.lower() for p in xc_audio)))
"""

C2 = """# === exp126 cell2: map XC file -> species + EXCLUDE train-contaminated recordings ===
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

def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())
sci2label = {norm(r["scientific_name"]): str(r["primary_label"]) for _, r in tax.iterrows()}
com2label = {norm(r["common_name"]): str(r["primary_label"]) for _, r in tax.iterrows()}

# XC recording ids already used in training (filename like 'taxon/XC123456.ogg' or 'taxon/iNat..')
def xc_id(s):
    m = re.search(r"XC\\s*0*([0-9]{3,})", str(s), flags=re.I)
    return m.group(1) if m else None
train_xc_ids = set(filter(None, (xc_id(f) for f in train["filename"])))
print("train XC recording ids:", len(train_xc_ids))

def infer_species(path):
    parts = [Path(path).stem] + list(Path(path).parts)
    for raw in parts:
        if str(raw) in label_set:
            return str(raw)                       # taxon_id directly in path
        n = norm(raw)
        if n in sci2label:
            return sci2label[n]                   # scientific name dir
        if n in com2label:
            return com2label[n]
    # try last 2 dir names joined (Genus species)
    pp = Path(path).parts
    for a, b in zip(pp, pp[1:]):
        n = norm(a + b)
        if n in sci2label:
            return sci2label[n]
    return None

rows = []
for p in xc_audio:
    sp = infer_species(p)
    rid = xc_id(p)
    rows.append({"path": p, "species": sp, "rec_id": rid,
                 "in_train": (rid is not None and rid in train_xc_ids)})
xc = pd.DataFrame(rows)
matched = xc[xc["species"].notna()].copy()
print(f"\\nmatched to a target species: {len(matched)}/{len(xc)}")
print(f"unmatched (species inference failed): {len(xc)-len(matched)}")
if len(xc) - len(matched):
    print("  sample unmatched paths:")
    for p in xc[xc["species"].isna()]["path"].head(10):
        print("   ", p.replace('/kaggle/input/', ''))

clean = matched[~matched["in_train"]].copy()        # TRUE OOF only
clean["taxon"] = clean["species"].map(l2t)
print(f"\\nafter dropping train-contaminated recordings: {len(clean)} clips")
print(f"  dropped as in_train: {int(matched['in_train'].sum())}")
print(f"  species covered (clean): {clean['species'].nunique()} / {NC}")
"""

C3 = """# === exp126 cell3: tier by train_audio count -> pick diagnostic targets ===
train["primary_label"] = train["primary_label"].astype(str)
train_count = train["primary_label"].value_counts().to_dict()

def tier(n):
    if n == 0: return "ghost"
    if n <= 5: return "very_rare"
    if n <= 20: return "rare"
    if n <= 50: return "low"
    return "common"

clean["train_count"] = clean["species"].map(lambda s: train_count.get(s, 0))
clean["tier"] = clean["train_count"].map(tier)

print("=== clean XC clips by tier (train_audio count) ===")
print(clean.groupby("tier").agg(clips=("path", "count"),
                                species=("species", "nunique")).reindex(
      ["ghost", "very_rare", "rare", "low", "common"]).fillna(0).astype(int))
print("\\n=== by taxon ===")
print(clean.groupby("taxon").agg(clips=("path", "count"), species=("species", "nunique")))

# diagnostic sampling: prioritize rare, cap per species + total to bound runtime
MAX_PER_SP = 5
rng = np.random.RandomState(0)
pick = []
order = {"ghost": 0, "very_rare": 1, "rare": 2, "low": 3, "common": 4}
for sp, g in clean.groupby("species"):
    g = g.sample(min(len(g), MAX_PER_SP), random_state=0)
    pick.append(g)
picked = pd.concat(pick).reset_index(drop=True)
# cap common to a baseline sample so rare dominate
rare_part = picked[picked["tier"].isin(["ghost", "very_rare", "rare", "low"])]
comm_part = picked[picked["tier"] == "common"]
if len(comm_part) > 150:
    comm_part = comm_part.sample(150, random_state=0)
DIAG = pd.concat([rare_part, comm_part]).reset_index(drop=True)
DIAG = DIAG.sort_values("path").reset_index(drop=True)
print(f"\\n=== diagnostic set: {len(DIAG)} clips, {DIAG['species'].nunique()} species ===")
print(DIAG.groupby("tier")["path"].count().reindex(
      ["ghost", "very_rare", "rare", "low", "common"]).fillna(0).astype(int))
"""

C4 = """# === exp126 cell4: load OV streams (Tucker 5-fold + exp106 3-fold) ===
import torch, torchaudio, soundfile as sf, librosa
SR=32000; N_MELS=256; N_FFT=2048; HOP=512; FMIN=20; FMAX=16000; TOP_DB=80
WIN=SR*5; N_WIN_CAP=8
def sig(x): return 1/(1+np.exp(-np.clip(x, -50, 50)))

TUCK = ff(["/kaggle/input/notebooks/maekeso/birdclef2026-tucker-sed-ov",
           "/kaggle/input/birdclef2026-tucker-sed-ov"], "sed_fold0.xml")
E106 = ff(["/kaggle/input/notebooks/maekeso/birdclef2026-e106-3fold-ov",
           "/kaggle/input/birdclef2026-e106-3fold-ov"], "exp106_fold0.xml")
print("TUCK", TUCK, "\\nE106", E106)

core = ov.Core()
tuck_models = [core.compile_model(str(p), "CPU") for p in sorted(TUCK.glob("sed_fold*.xml"))]
e106_models = [core.compile_model(str(E106 / f"exp106_fold{f}.xml"), "CPU") for f in [0, 1, 2]]
print("tucker folds", len(tuck_models), "e106 folds", len(e106_models))

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

C5 = """# === exp126 cell5: inference + per-species diagnostic ===
recs = []
t0 = time.time()
for k, r in DIAG.iterrows():
    try:
        chs = load_windows(r["path"])
    except Exception as e:
        continue
    ML = mel_lib(chs); MT = mel_torch(chs)
    pt = np.mean([run(m, ML) for m in tuck_models], axis=0)
    pe = np.mean([run(m, MT) for m in e106_models], axis=0)
    pb = 0.5 * pt + 0.5 * pe                  # [nwin, NC]
    pc = pb.max(0)                            # presence score per class
    ci = L2I[r["species"]]
    rank = int((pc > pc[ci]).sum()) + 1       # 1 = best
    recs.append({"species": r["species"], "taxon": r["taxon"], "tier": r["tier"],
                 "train_count": int(r["train_count"]), "p_true": float(pc[ci]),
                 "rank": rank, "top1": int(rank == 1), "top5": int(rank <= 5),
                 "argmax_label": LABELS[int(pc.argmax())]})
    if (k + 1) % 50 == 0:
        print(f"  {k+1}/{len(DIAG)}  ({time.time()-t0:.0f}s)")
D = pd.DataFrame(recs)
D.to_csv("/kaggle/working/exp126_xc_diag.csv", index=False)
print(f"\\nscored {len(D)} clips in {time.time()-t0:.0f}s")
"""

C6 = """# === exp126 cell6: report ===
print("=== diagnostic by tier (does model identify the TRUE species on unseen XC?) ===")
order = ["ghost", "very_rare", "rare", "low", "common"]
agg = D.groupby("tier").agg(clips=("rank", "count"), species=("species", "nunique"),
                            top1=("top1", "mean"), top5=("top5", "mean"),
                            p_true=("p_true", "mean"), med_rank=("rank", "median")).reindex(order).dropna(how="all")
print(agg.round(3))

print("\\n=== by taxon ===")
print(D.groupby("taxon").agg(clips=("rank", "count"), top1=("top1", "mean"),
                             top5=("top5", "mean"), p_true=("p_true", "mean"),
                             med_rank=("rank", "median")).round(3))

print("\\n=== per-species (rare/very_rare/ghost), worst first ===")
sp = D[D["tier"].isin(["ghost", "very_rare", "rare"])].groupby(
     ["species", "taxon", "tier"]).agg(
     clips=("rank", "count"), top1=("top1", "mean"), top5=("top5", "mean"),
     p_true=("p_true", "mean"), med_rank=("rank", "median")).reset_index()
sp = sp.sort_values(["top5", "p_true"])
print(f"{'species':>12} {'taxon':>9} {'tier':>10} {'n':>3} {'top1':>5} {'top5':>5} {'p_true':>7} {'medR':>5}")
for _, r in sp.head(40).iterrows():
    print(f"{r['species']:>12} {r['taxon']:>9} {r['tier']:>10} {int(r['clips']):>3} "
          f"{r['top1']:>5.2f} {r['top5']:>5.2f} {r['p_true']:>7.3f} {int(r['med_rank']):>5}")

print("\\n=== INTERPRETATION ===")
rare = D[D['tier'].isin(['ghost','very_rare','rare'])]
comm = D[D['tier'] == 'common']
if len(rare) and len(comm):
    print(f"rare top5 = {rare['top5'].mean():.3f}  vs  common top5 = {comm['top5'].mean():.3f}")
    print(f"rare p_true= {rare['p_true'].mean():.3f}  vs  common p_true= {comm['p_true'].mean():.3f}")
    print("-> if rare top5 << common top5: model does NOT generalize to rare species (data axis gap)")
    print("-> note: NB4/Perch stream excluded; may lift ghost/rare further")
"""

CELLS = [(C1, "disc"), (C2, "map"), (C3, "tier"), (C4, "load"), (C5, "infer"), (C6, "report")]
nb = {"cells": [cell(s, c) for s, c in CELLS],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {NB}")
# syntax check (strip ! / % lines for compile)
for s, c in CELLS:
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln for ln in s.splitlines())
    try:
        compile(clean, f"<{c}>", "exec"); print(f"  [OK] {c}")
    except SyntaxError as e:
        print(f"  [FAIL] {c}: {e}")
