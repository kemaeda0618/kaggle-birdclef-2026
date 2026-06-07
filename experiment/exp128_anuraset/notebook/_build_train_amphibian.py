"""exp128: Amphibia SPECIALIST training NB (Kaggle T4x2).

Train an amphibian-focused multi-label classifier on:
  - AnuraSet (denden12/anuraset-bc26-32k-mono-ogg) — SOUNDSCAPE recordings, domain-matched to test
  - train_audio amphibian clips (35 Amphibia primary_labels)
Output: 35-way (our Amphibia label order) probabilities.
Backbone: efficientnet_b0 (clean OpenVINO conversion, fast). mel identical to main pipeline.
Export: amphib_b0.pth + OpenVINO IR (amphib_b0.xml/.bin) to /kaggle/working.

Surgical use later: blend ONLY the 35 amphibian columns into the 234-way ensemble (Aves untouched).
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path("experiment/exp128_anuraset/notebook/nb_exp128_train_amphibian.ipynb")
NB.parent.mkdir(parents=True, exist_ok=True)


def cell(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}


C1 = """# === exp128 cell1: setup ===
import os, sys, glob, re, time, random, json
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torchaudio
import soundfile as sf, librosa

!pip install -q timm openvino
import timm
print("torch", torch.__version__, "timm", timm.__version__, "cuda", torch.cuda.is_available())

SR=32000; N_FFT=2048; HOP=512; N_MELS=256; FMIN=20; FMAX=16000; TOP_DB=80
DUR=5; WIN=SR*DUR
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS=14; BS=64; LR=1e-3; WD=1e-2; BACKBONE="efficientnet_b0"
"""

C2 = """# === exp128 cell2: paths + 35 amphibian label space + AnuraSet mapping ===
def ff(c, m):
    for p in c:
        p = Path(p)
        if p.exists() and (list(p.rglob(m)) or (p / m).exists()): return p
    return None
COMP = ff(["/kaggle/input/competitions/birdclef-2026", "/kaggle/input/birdclef-2026"], "taxonomy.csv")
ANU = ff(["/kaggle/input/anuraset-bc26-32k-mono-ogg",
          "/kaggle/input/datasets/denden12/anuraset-bc26-32k-mono-ogg"], "labels.csv")
print("COMP", COMP, "\\nANU", ANU)

tax = pd.read_csv(COMP/"taxonomy.csv")
AMP = tax[tax["class_name"]=="Amphibia"]["primary_label"].astype(str).tolist()
AMP = sorted(AMP)
A2I = {a:i for i,a in enumerate(AMP)}; NC=len(AMP)
print("amphibian classes:", NC)

train = pd.read_csv(COMP/"train.csv"); train["primary_label"]=train["primary_label"].astype(str)
sci2id = dict(zip(tax["scientific_name"], tax["primary_label"].astype(str)))

# AnuraSet SPECIES_<code> -> our amphibian idx (direct+synonym only)
amap = pd.read_csv(ANU/"bc26_mapping.csv")
code2idx = {}
for _,r in amap.iterrows():
    if pd.notna(r["bc26_primary_label"]) and r["match_type"] in ("direct","synonym"):
        bcid = sci2id.get(r["bc26_primary_label"])
        if bcid in A2I:
            code2idx["SPECIES_"+r["anuraset_code"]] = A2I[bcid]
print("AnuraSet columns mapped to our amphibians:", len(code2idx))
"""

C3 = """# === exp128 cell3: assemble training items (path, multi-hot label) ===
items = []  # (path, label_vec float32[NC], is_soundscape)

# --- AnuraSet 1-min soundscape files ---
lab = pd.read_csv(ANU/"labels.csv")
anu_audio = {Path(p).stem: p for p in glob.glob(str(ANU/"audio/**/*.ogg"), recursive=True)}
print("AnuraSet ogg files:", len(anu_audio))
mapped_cols = [c for c in code2idx if c in lab.columns]
n_anu=0
for _,r in lab.iterrows():
    stem = str(r["AUDIO_FILE_ID"])
    p = anu_audio.get(stem)
    if p is None:
        # try fuzzy: some ids may differ in zero padding
        cand = [v for k,v in anu_audio.items() if k.endswith(stem) or stem.endswith(k)]
        p = cand[0] if cand else None
    if p is None: continue
    y = np.zeros(NC, dtype=np.float32)
    for c in mapped_cols:
        if r[c] > 0: y[code2idx[c]] = 1.0
    items.append((p, y, True))   # keep even all-zero (negatives are useful soundscape context)
    n_anu+=1
print("AnuraSet items:", n_anu)

# --- train_audio amphibian focal clips ---
ta_root = COMP/"train_audio"
amp_set = set(AMP)
def sec_labels(s):
    return [t for t in re.findall(r"[A-Za-z0-9]+", str(s)) if t in amp_set]
n_ta=0
for _,r in train[train["primary_label"].isin(amp_set)].iterrows():
    p = ta_root/r["filename"]
    if not p.exists(): continue
    y = np.zeros(NC, dtype=np.float32); y[A2I[r["primary_label"]]] = 1.0
    for s in sec_labels(r.get("secondary_labels","")): y[A2I[s]] = 1.0
    items.append((str(p), y, False)); n_ta+=1
print("train_audio amphibian items:", n_ta, "| total items:", len(items))

# train/val split
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(items)); nval = int(len(items)*0.15)
val_idx = set(idx[:nval].tolist())
train_items = [it for i,it in enumerate(items) if i not in val_idx]
val_items   = [it for i,it in enumerate(items) if i in val_idx]
print("train", len(train_items), "val", len(val_items))
pos_per_class = np.stack([y for _,y,_ in items]).sum(0)
print("positives per class (min/median/max):", int(pos_per_class.min()), int(np.median(pos_per_class)), int(pos_per_class.max()))
"""

C4 = """# === exp128 cell4: dataset / mel ===
melspec = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=N_FFT, hop_length=HOP,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0)
to_db = torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)

def load_crop(path, train=True):
    try:
        w, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        w, sr = librosa.load(path, sr=SR, mono=True)
    if getattr(w,"ndim",1) > 1: w = w.mean(1)
    if sr != SR: w = librosa.resample(w, orig_sr=sr, target_sr=SR)
    if len(w) < WIN:
        w = np.pad(w, (0, WIN-len(w)))
    if train:
        st = random.randint(0, len(w)-WIN)
    else:
        st = max(0, (len(w)-WIN)//2)
    return w[st:st+WIN].astype(np.float32)

class DS(torch.utils.data.Dataset):
    def __init__(self, items, train=True): self.items=items; self.train=train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y, _ = self.items[i]
        w = load_crop(p, self.train)
        return torch.from_numpy(w), torch.from_numpy(y)

def collate(batch):
    ws = torch.stack([b[0] for b in batch]); ys = torch.stack([b[1] for b in batch])
    m = to_db(melspec(ws))
    mu = m.mean((1,2),keepdim=True); sd = m.std((1,2),keepdim=True)+1e-6
    m = ((m-mu)/sd).unsqueeze(1)   # [B,1,N_MELS,T]
    return m, ys

tl = torch.utils.data.DataLoader(DS(train_items,True), batch_size=BS, shuffle=True,
        num_workers=2, collate_fn=collate, drop_last=True, pin_memory=True)
vl = torch.utils.data.DataLoader(DS(val_items,False), batch_size=BS, shuffle=False,
        num_workers=2, collate_fn=collate, pin_memory=True)
print("batches/epoch", len(tl))
"""

C5 = """# === exp128 cell5: train ===
from sklearn.metrics import roc_auc_score
model = timm.create_model(BACKBONE, pretrained=True, in_chans=1, num_classes=NC).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS*len(tl))
scaler = torch.cuda.amp.GradScaler()
crit = nn.BCEWithLogitsLoss()

def evaluate():
    model.eval(); P=[]; Y=[]
    with torch.no_grad():
        for m,y in vl:
            with torch.cuda.amp.autocast():
                o = model(m.to(DEV))
            P.append(torch.sigmoid(o).float().cpu().numpy()); Y.append(y.numpy())
    P=np.concatenate(P); Y=np.concatenate(Y)
    aucs=[]
    for c in range(NC):
        if 0 < Y[:,c].sum() < len(Y):
            try: aucs.append(roc_auc_score(Y[:,c], P[:,c]))
            except: pass
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)

best=-1
for ep in range(EPOCHS):
    model.train(); t0=time.time(); tot=0
    for bi,(m,y) in enumerate(tl):
        m=m.to(DEV); y=y.to(DEV)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            loss = crit(model(m), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot+=loss.item()
        if bi % 50 == 0:
            print(f"  ep{ep} step{bi}/{len(tl)} loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
    va, nev = evaluate()
    print(f"[ep{ep}] train_loss={tot/len(tl):.4f} val_macro_auc={va:.4f} (evaluable {nev}/{NC}) {time.time()-t0:.0f}s", flush=True)
    if va > best:
        best=va; torch.save(model.state_dict(), "/kaggle/working/amphib_b0.pth")
        print(f"  ** BEST {best:.4f} saved")
print("best val_macro_auc:", best)
"""

C6 = """# === exp128 cell6: export OpenVINO IR + label order ===
import openvino as ov
model.load_state_dict(torch.load("/kaggle/working/amphib_b0.pth", map_location=DEV)); model.eval().to(DEV)
# frame count for 5s
T = melspec(torch.zeros(WIN)).shape[-1]
ex = torch.zeros(1,1,N_MELS,T)
model_cpu = model.cpu().eval()
ovm = ov.convert_model(model_cpu, example_input=ex, input=[1,1,N_MELS,T])
# allow dynamic batch
ovm.reshape({0: ov.PartialShape([-1,1,N_MELS,T])})
ov.save_model(ovm, "/kaggle/working/amphib_b0.xml")
json.dump({"labels": AMP, "n_mels": N_MELS, "T": int(T), "sr": SR, "hop": HOP, "n_fft": N_FFT,
           "fmin": FMIN, "fmax": FMAX, "top_db": TOP_DB, "win": WIN, "backbone": BACKBONE},
          open("/kaggle/working/amphib_meta.json","w"), indent=2)
print("exported amphib_b0.pth / amphib_b0.xml / amphib_meta.json")
print("AMP label order:", AMP)
import os
for f in sorted(os.listdir("/kaggle/working")):
    print("  ", f, os.path.getsize("/kaggle/working/"+f))
"""

CELLS=[(C1,"setup"),(C2,"paths"),(C3,"data"),(C4,"ds"),(C5,"train"),(C6,"export")]
nb={"cells":[cell(s,c) for s,c in CELLS],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},
    "nbformat":4,"nbformat_minor":5}
NB.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding="utf-8")
print(f"Wrote {NB}")
for s,c in CELLS:
    clean="\n".join("# "+ln if ln.lstrip().startswith(("!","%")) else ln for ln in s.splitlines())
    try: compile(clean,f"<{c}>","exec"); print(f"  [OK] {c}")
    except SyntaxError as e: print(f"  [FAIL] {c}: {e}")
