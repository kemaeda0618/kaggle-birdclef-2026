"""exp132: Amphibia SPECIALIST v3 — STRONG-labeled AnuraSet (93k 3s samples) + train_audio.

Key upgrade over exp128/131 (weak 1-min labels):
  mismaresenka/anuraset-preprocessed = 93,378 3-second samples with STRONG per-window labels
  (1 iff the species actually calls in that 3s window). Cleaner supervision -> fewer false
  positives -> better column AUC (the LB metric). Addresses exp130's main FP risk at the data level.

Data: AnuraSet-strong (18 of our amphibians, via embedded code map) + train_audio amphibians (35).
22050Hz -> 32000 resample, 3s -> 5s pad. effb0 35-way. SpecAug/mixup/oversample. Export v3 pth.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path("experiment/exp132/notebook/nb_exp132_train_v3.ipynb")
NB.parent.mkdir(parents=True, exist_ok=True)


def cell(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}


C1 = """# === exp132 cell1: setup ===
import os, sys, glob, re, time, random, json
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torchaudio, soundfile as sf, librosa
!pip install -q timm
import timm
SR=32000; N_FFT=2048; HOP=512; N_MELS=256; FMIN=20; FMAX=16000; TOP_DB=80
DUR=5; WIN=SR*DUR; SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV="cuda" if torch.cuda.is_available() else "cpu"
EPOCHS=15; BS=64; LR=1e-3; WD=1e-2; BACKBONE="efficientnet_b0"
EPOCH_SAMPLES=40000; MIXUP_P=0.5; MIXUP_A=0.2
print("torch",torch.__version__,"cuda",torch.cuda.is_available())
"""

C2 = """# === exp132 cell2: paths + 35 labels + AnuraSet code->label (embedded, 18 direct) ===
def ff(c,m):
    for p in c:
        p=Path(p)
        if p.exists() and (list(p.rglob(m)) or (p/m).exists()): return p
    return None
COMP=ff(["/kaggle/input/competitions/birdclef-2026","/kaggle/input/birdclef-2026"],"taxonomy.csv")
_anu_base=ff(["/kaggle/input/anuraset-preprocessed","/kaggle/input/datasets/mismaresenka/anuraset-preprocessed"],"metadata.csv")
ANU=next(iter(Path(_anu_base).rglob("metadata.csv"))).parent   # dir actually containing metadata.csv (robust to mount)
print("COMP",COMP,"\\nANU",ANU)

tax=pd.read_csv(COMP/"taxonomy.csv")
AMP=sorted(tax[tax["class_name"]=="Amphibia"]["primary_label"].astype(str).tolist())
A2I={a:i for i,a in enumerate(AMP)}; NC=len(AMP); print("amphibian classes",NC)

# AnuraSet species code -> our BC26 primary_label (direct + synonym), from bc26_mapping (embedded)
CODE2LABEL = {
 "AMEPIC":"64898","BOALUN":"555123","BOARAN":"555146","DENMIN":"65377","DENNAN":"65380",
 "ELABIC":"25092","LEPELE":"22967","LEPFUS":"22973","LEPLAB":"22983","LEPLAT":"1176823",
 "LEPPOD":"22961","PHYALB":"23158","PHYNAT":"476521","PHYSAU":"23724","PITAZU":"517063",
 "SCIFUS":"24287","SCIFUV":"24285","SCINAS":"24279",
}
CODE2IDX={c:A2I[l] for c,l in CODE2LABEL.items() if l in A2I}
print("AnuraSet codes mapped:",len(CODE2IDX))
train=pd.read_csv(COMP/"train.csv"); train["primary_label"]=train["primary_label"].astype(str)
"""

C3 = """# === exp132 cell3: assemble items (AnuraSet STRONG + train_audio) ===
items=[]  # (path, label_vec[NC], source, is_test)
meta=pd.read_csv(ANU/"metadata.csv")
print("AnuraSet metadata rows:",len(meta),"| subset:",meta["subset"].value_counts().to_dict())
anu_audio={Path(p).stem:p for p in glob.glob(str(ANU/"**/*.wav"),recursive=True)}
print("AnuraSet wavs found:",len(anu_audio))
code_cols=[c for c in CODE2IDX if c in meta.columns]
n_anu=0; n_anu_pos=0
for _,r in meta.iterrows():
    key=f"{r['fname']}_{int(r['min_t'])}_{int(r['max_t'])}"   # actual file = {fname}_{min_t}_{max_t}.wav
    p=anu_audio.get(key)
    if p is None: continue
    y=np.zeros(NC,dtype=np.float32)
    for c in code_cols:
        if r[c]>0: y[CODE2IDX[c]]=1.0
    items.append((p,y,"anuraset",str(r.get("subset",""))=="test")); n_anu+=1
    if y.sum()>0: n_anu_pos+=1
print(f"AnuraSet items {n_anu} (pos for our species {n_anu_pos})")
assert n_anu > 80000, f"FAIL-FAST: AnuraSet matched only {n_anu} (expected ~93378). key construction broken."

# train_audio amphibians (focal, all 35)
amp_set=set(AMP); ta=COMP/"train_audio"
def sec(s): return [t for t in re.findall(r"[A-Za-z0-9]+",str(s)) if t in amp_set]
n_ta=0
for _,r in train[train["primary_label"].isin(amp_set)].iterrows():
    p=ta/r["filename"]
    if not p.exists(): continue
    y=np.zeros(NC,dtype=np.float32); y[A2I[r["primary_label"]]]=1.0
    for s in sec(r.get("secondary_labels","")): y[A2I[s]]=1.0
    items.append((str(p),y,"train_audio",False)); n_ta+=1
print(f"train_audio items {n_ta} | total {len(items)}")

# split: AnuraSet test subset -> val; rest -> train (+ train_audio all train)
train_items=[it for it in items if not it[3]]
val_items=[it for it in items if it[3]]
if len(val_items)<200:   # fallback random val
    rng=np.random.RandomState(SEED); idx=rng.permutation(len(items)); nv=int(len(items)*0.12)
    vs=set(idx[:nv].tolist()); train_items=[it for i,it in enumerate(items) if i not in vs]; val_items=[it for i,it in enumerate(items) if i in vs]
Y=np.stack([y for _,y,_,_ in train_items]); pos=Y.sum(0)
print("train",len(train_items),"val",len(val_items),"| pos/class min/med/max",int(pos.min()),int(np.median(pos)),int(pos.max()))
cls_w=1.0/np.sqrt(np.clip(pos,1,None))
tr_w=np.array([(cls_w[y>0].max() if (y>0).any() else cls_w.min()) for _,y,_,_ in train_items],dtype=np.float64)
"""

C4 = """# === exp132 cell4: dataset + aug ===
melspec=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,f_min=FMIN,f_max=FMAX,power=2.0)
to_db=torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)
freq_mask=torchaudio.transforms.FrequencyMasking(freq_mask_param=24)
time_mask=torchaudio.transforms.TimeMasking(time_mask_param=32)
def load_crop(path,train=True):
    try: w,sr=sf.read(path,dtype="float32",always_2d=False)
    except Exception: w,sr=librosa.load(path,sr=SR,mono=True)
    if getattr(w,"ndim",1)>1: w=w.mean(1)
    if sr!=SR: w=librosa.resample(w,orig_sr=sr,target_sr=SR)
    if len(w)<WIN:
        w=np.tile(w,int(np.ceil(WIN/max(1,len(w)))))[:WIN]   # tile short clips (3s->5s) to fill window, no silence pad
    else:
        st=random.randint(0,len(w)-WIN) if train else max(0,(len(w)-WIN)//2); w=w[st:st+WIN]
    w=w[:WIN].astype(np.float32)
    if train:
        w=w*np.float32(10**(random.uniform(-6,6)/20))
        if random.random()<0.3: w=w+np.random.randn(WIN).astype(np.float32)*0.003*(np.std(w)+1e-6)
    return w
class DS(torch.utils.data.Dataset):
    def __init__(s,it,tr=True): s.it=it; s.tr=tr
    def __len__(s): return len(s.it)
    def __getitem__(s,i):
        p,y,_,_=s.it[i]; return torch.from_numpy(load_crop(p,s.tr)), torch.from_numpy(y)
def col_tr(b):
    ws=torch.stack([x[0] for x in b]); ys=torch.stack([x[1] for x in b])
    m=to_db(melspec(ws)); mu=m.mean((1,2),keepdim=True); sd=m.std((1,2),keepdim=True)+1e-6
    m=((m-mu)/sd).unsqueeze(1); return time_mask(freq_mask(m)),ys
def col_va(b):
    ws=torch.stack([x[0] for x in b]); ys=torch.stack([x[1] for x in b])
    m=to_db(melspec(ws)); mu=m.mean((1,2),keepdim=True); sd=m.std((1,2),keepdim=True)+1e-6
    return ((m-mu)/sd).unsqueeze(1),ys
sampler=torch.utils.data.WeightedRandomSampler(tr_w,num_samples=min(EPOCH_SAMPLES,len(train_items)),replacement=True)
tl=torch.utils.data.DataLoader(DS(train_items,True),batch_size=BS,sampler=sampler,num_workers=4,collate_fn=col_tr,drop_last=True,pin_memory=True)
vl=torch.utils.data.DataLoader(DS(val_items,False),batch_size=BS,shuffle=False,num_workers=4,collate_fn=col_va,pin_memory=True)
print("batches/epoch",len(tl))
"""

C5 = """# === exp132 cell5: train ===
from sklearn.metrics import roc_auc_score
model=timm.create_model(BACKBONE,pretrained=True,in_chans=1,num_classes=NC).to(DEV)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS*len(tl))
scaler=torch.amp.GradScaler('cuda'); crit=nn.BCEWithLogitsLoss()
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
            lam=np.random.beta(MIXUP_A,MIXUP_A); pm=torch.randperm(m.size(0),device=DEV)
            m=lam*m+(1-lam)*m[pm]; y=lam*y+(1-lam)*y[pm]
        opt.zero_grad()
        with torch.amp.autocast('cuda'): loss=crit(model(m),y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tot+=loss.item()
        if bi%100==0: print(f"  ep{ep} step{bi}/{len(tl)} loss={loss.item():.4f}",flush=True)
    va,nev=evaluate()
    print(f"[ep{ep}] loss={tot/len(tl):.4f} val_macro_auc={va:.4f} ({nev}/{NC}) {time.time()-t0:.0f}s",flush=True)
    if va>best: best=va; torch.save(model.state_dict(),"/kaggle/working/amphib_b0_v3.pth"); print(f"  ** BEST {best:.4f}")
print("best",best)
"""

C6 = """# === exp132 cell6: export ===
T=1+WIN//HOP
meta_out={"labels":AMP,"n_mels":N_MELS,"T":int(T),"sr":SR,"hop":HOP,"n_fft":N_FFT,"fmin":FMIN,"fmax":FMAX,"top_db":TOP_DB,"win":WIN,"backbone":BACKBONE,"best_val":float(best),"source":"anuraset-strong+train_audio"}
json.dump(meta_out,open("/kaggle/working/amphib_v3_meta.json","w"),indent=2)
print("exported amphib_b0_v3.pth + amphib_v3_meta.json; best val",best)
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
