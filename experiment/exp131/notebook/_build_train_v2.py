"""exp131: Amphibia SPECIALIST v2 — AnuraSet + train_audio + iNat, + SpecAug/mixup/oversample.

Improvements over exp128:
  + iNat (shadowdude/train-recordings) Amphibia dirs matched to our 35 -> covers AnuraSet-missing
    17 species incl true ghosts (1491113 Adenomera guarani, 25073 Chiasmocleis mehelyi).
  + SpecAugment (freq/time mask) + waveform gain/noise + light mixup -> generalization.
  + WeightedRandomSampler (oversample rare amphibians).
  + 20 epochs.
Backbone effb0, 35-way, mel identical to main pipeline. Export amphib_b0_v2.pth + meta.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path("experiment/exp131/notebook/nb_exp131_train_v2.ipynb")
NB.parent.mkdir(parents=True, exist_ok=True)


def cell(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}


C1 = """# === exp131 cell1: setup ===
import os, sys, glob, re, time, random, json
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torchaudio, soundfile as sf, librosa
!pip install -q timm openvino
import timm
SR=32000; N_FFT=2048; HOP=512; N_MELS=256; FMIN=20; FMAX=16000; TOP_DB=80
DUR=5; WIN=SR*DUR; SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV="cuda" if torch.cuda.is_available() else "cpu"
EPOCHS=20; BS=64; LR=1e-3; WD=1e-2; BACKBONE="efficientnet_b0"
INAT_CAP=60; ANU_OK=True
MIXUP_P=0.5; MIXUP_A=0.2
print("torch",torch.__version__,"cuda",torch.cuda.is_available())
"""

C2 = """# === exp131 cell2: paths + 35 labels + AnuraSet mapping ===
def ff(c,m):
    for p in c:
        p=Path(p)
        if p.exists() and (list(p.rglob(m)) or (p/m).exists()): return p
    return None
COMP=ff(["/kaggle/input/competitions/birdclef-2026","/kaggle/input/birdclef-2026"],"taxonomy.csv")
ANU=ff(["/kaggle/input/anuraset-bc26-32k-mono-ogg","/kaggle/input/datasets/denden12/anuraset-bc26-32k-mono-ogg"],"labels.csv")
INAT=ff(["/kaggle/input/train-recordings","/kaggle/input/datasets/shadowdude/train-recordings"],"train")
print("COMP",COMP,"\\nANU",ANU,"\\nINAT",INAT)

tax=pd.read_csv(COMP/"taxonomy.csv")
AMP=sorted(tax[tax["class_name"]=="Amphibia"]["primary_label"].astype(str).tolist())
A2I={a:i for i,a in enumerate(AMP)}; NC=len(AMP); print("amphibian classes",NC)
train=pd.read_csv(COMP/"train.csv"); train["primary_label"]=train["primary_label"].astype(str)
def norm(s): return re.sub(r"[^a-z0-9]","",str(s).lower())
# our amphibian scientific name (normalized) -> label
amp_sci2label={norm(r["scientific_name"]):str(r["primary_label"]) for _,r in tax.iterrows()
               if r["class_name"]=="Amphibia"}
sci2label_all={norm(r["scientific_name"]):str(r["primary_label"]) for _,r in tax.iterrows()}

amap=pd.read_csv(ANU/"bc26_mapping.csv")
code2idx={}
for _,r in amap.iterrows():
    if pd.notna(r["bc26_primary_label"]) and r["match_type"] in ("direct","synonym"):
        bid=sci2label_all.get(norm(r["bc26_primary_label"]))
        if bid in A2I: code2idx["SPECIES_"+r["anuraset_code"]]=A2I[bid]
print("AnuraSet mapped cols",len(code2idx))
"""

C3 = """# === exp131 cell3: assemble items (AnuraSet + train_audio + iNat) ===
items=[]  # (path, label_vec[NC], source)

# AnuraSet
lab=pd.read_csv(ANU/"labels.csv")
anu_audio={Path(p).stem:p for p in glob.glob(str(ANU/"audio/**/*.ogg"),recursive=True)}
mapped=[c for c in code2idx if c in lab.columns]
for _,r in lab.iterrows():
    p=anu_audio.get(str(r["AUDIO_FILE_ID"]))
    if p is None: continue
    y=np.zeros(NC,dtype=np.float32)
    for c in mapped:
        if r[c]>0: y[code2idx[c]]=1.0
    items.append((p,y,"anuraset"))
n_anu=len(items)

# train_audio amphibians
amp_set=set(AMP); ta=COMP/"train_audio"
def sec(s): return [t for t in re.findall(r"[A-Za-z0-9]+",str(s)) if t in amp_set]
for _,r in train[train["primary_label"].isin(amp_set)].iterrows():
    p=ta/r["filename"]
    if not p.exists(): continue
    y=np.zeros(NC,dtype=np.float32); y[A2I[r["primary_label"]]]=1.0
    for s in sec(r.get("secondary_labels","")): y[A2I[s]]=1.0
    items.append((str(p),y,"train_audio"))
n_ta=len(items)-n_anu

# iNat (shadowdude/train-recordings): match Amphibia dirs -> our 35
n_inat=0; inat_cov=set()
if INAT is not None:
    itrain=INAT/"train" if (INAT/"train").exists() else INAT
    for d in itrain.iterdir():
        if not d.is_dir() or "_Amphibia_" not in d.name: continue
        parts=d.name.split("_")
        sci=norm(parts[-2]+parts[-1]) if len(parts)>=2 else ""
        lbl=amp_sci2label.get(sci)
        if lbl is None: continue
        af=[]
        for e in ("*.ogg","*.mp3","*.wav","*.flac"): af+=list(d.rglob(e))
        for a in af[:INAT_CAP]:
            y=np.zeros(NC,dtype=np.float32); y[A2I[lbl]]=1.0
            items.append((str(a),y,"inat")); n_inat+=1; inat_cov.add(lbl)
print(f"items: anuraset={n_anu} train_audio={n_ta} inat={n_inat} (cov {len(inat_cov)} sp) total={len(items)}")

# split
rng=np.random.RandomState(SEED); idx=rng.permutation(len(items)); nval=int(len(items)*0.15)
vset=set(idx[:nval].tolist())
train_items=[it for i,it in enumerate(items) if i not in vset]
val_items=[it for i,it in enumerate(items) if i in vset]
Y=np.stack([y for _,y,_ in items]); pos=Y.sum(0)
print("train",len(train_items),"val",len(val_items),"| pos/class min/med/max",int(pos.min()),int(np.median(pos)),int(pos.max()))
# sampler weights (oversample rare): item weight = max over positive classes of 1/sqrt(freq)
cls_w=1.0/np.sqrt(np.clip(pos,1,None))
tr_w=[]
for _,y,_ in train_items:
    w=cls_w[y>0].max() if (y>0).any() else cls_w.min()
    tr_w.append(float(w))
tr_w=np.array(tr_w,dtype=np.float64)
"""

C4 = """# === exp131 cell4: dataset + aug ===
melspec=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,f_min=FMIN,f_max=FMAX,power=2.0)
to_db=torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)
freq_mask=torchaudio.transforms.FrequencyMasking(freq_mask_param=24)
time_mask=torchaudio.transforms.TimeMasking(time_mask_param=32)

def load_crop(path,train=True):
    try: w,sr=sf.read(path,dtype="float32",always_2d=False)
    except Exception: w,sr=librosa.load(path,sr=SR,mono=True)
    if getattr(w,"ndim",1)>1: w=w.mean(1)
    if sr!=SR: w=librosa.resample(w,orig_sr=sr,target_sr=SR)
    if len(w)<WIN: w=np.pad(w,(0,WIN-len(w)))
    st=random.randint(0,len(w)-WIN) if train else max(0,(len(w)-WIN)//2)
    w=w[st:st+WIN].astype(np.float32)
    if train:
        w=w*np.float32(10**(random.uniform(-6,6)/20))           # random gain +-6dB
        if random.random()<0.3: w=w+np.random.randn(WIN).astype(np.float32)*0.003*np.std(w)
    return w

class DS(torch.utils.data.Dataset):
    def __init__(self,items,train=True): self.items=items; self.train=train
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,y,_=self.items[i]
        return torch.from_numpy(load_crop(p,self.train)), torch.from_numpy(y)

def collate_train(batch):
    ws=torch.stack([b[0] for b in batch]); ys=torch.stack([b[1] for b in batch])
    m=to_db(melspec(ws)); mu=m.mean((1,2),keepdim=True); sd=m.std((1,2),keepdim=True)+1e-6
    m=((m-mu)/sd).unsqueeze(1)
    m=time_mask(freq_mask(m))                                    # SpecAugment
    return m,ys
def collate_val(batch):
    ws=torch.stack([b[0] for b in batch]); ys=torch.stack([b[1] for b in batch])
    m=to_db(melspec(ws)); mu=m.mean((1,2),keepdim=True); sd=m.std((1,2),keepdim=True)+1e-6
    return ((m-mu)/sd).unsqueeze(1),ys

sampler=torch.utils.data.WeightedRandomSampler(tr_w, num_samples=len(train_items), replacement=True)
tl=torch.utils.data.DataLoader(DS(train_items,True),batch_size=BS,sampler=sampler,num_workers=2,collate_fn=collate_train,drop_last=True,pin_memory=True)
vl=torch.utils.data.DataLoader(DS(val_items,False),batch_size=BS,shuffle=False,num_workers=2,collate_fn=collate_val,pin_memory=True)
print("batches/epoch",len(tl))
"""

C5 = """# === exp131 cell5: train (mixup + BCE) ===
from sklearn.metrics import roc_auc_score
model=timm.create_model(BACKBONE,pretrained=True,in_chans=1,num_classes=NC).to(DEV)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS*len(tl))
scaler=torch.amp.GradScaler('cuda')
crit=nn.BCEWithLogitsLoss()

def evaluate():
    model.eval(); P=[];Yv=[]
    with torch.no_grad():
        for m,y in vl:
            with torch.amp.autocast('cuda'): o=model(m.to(DEV))
            P.append(torch.sigmoid(o).float().cpu().numpy()); Yv.append(y.numpy())
    P=np.concatenate(P); Yv=np.concatenate(Yv); a=[]
    for c in range(NC):
        if 0<Yv[:,c].sum()<len(Yv):
            try: a.append(roc_auc_score(Yv[:,c],P[:,c]))
            except: pass
    return float(np.mean(a)) if a else float("nan"), len(a)

best=-1
for ep in range(EPOCHS):
    model.train(); t0=time.time(); tot=0
    for bi,(m,y) in enumerate(tl):
        m=m.to(DEV); y=y.to(DEV)
        if random.random()<MIXUP_P:
            lam=np.random.beta(MIXUP_A,MIXUP_A); perm=torch.randperm(m.size(0),device=DEV)
            m=lam*m+(1-lam)*m[perm]; y=lam*y+(1-lam)*y[perm]
        opt.zero_grad()
        with torch.amp.autocast('cuda'): loss=crit(model(m),y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot+=loss.item()
        if bi%50==0: print(f"  ep{ep} step{bi}/{len(tl)} loss={loss.item():.4f}",flush=True)
    va,nev=evaluate()
    print(f"[ep{ep}] loss={tot/len(tl):.4f} val_macro_auc={va:.4f} ({nev}/{NC}) {time.time()-t0:.0f}s",flush=True)
    if va>best: best=va; torch.save(model.state_dict(),"/kaggle/working/amphib_b0_v2.pth"); print(f"  ** BEST {best:.4f}")
print("best",best)
"""

C6 = """# === exp131 cell6: export ===
model.load_state_dict(torch.load("/kaggle/working/amphib_b0_v2.pth",map_location=DEV)); model.eval()
T=1+WIN//HOP
meta={"labels":AMP,"n_mels":N_MELS,"T":int(T),"sr":SR,"hop":HOP,"n_fft":N_FFT,"fmin":FMIN,"fmax":FMAX,"top_db":TOP_DB,"win":WIN,"backbone":BACKBONE,"best_val":float(best)}
json.dump(meta,open("/kaggle/working/amphib_v2_meta.json","w"),indent=2)
print("exported amphib_b0_v2.pth + amphib_v2_meta.json; best val",best)
for f in sorted(os.listdir("/kaggle/working")): print("  ",f)
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
