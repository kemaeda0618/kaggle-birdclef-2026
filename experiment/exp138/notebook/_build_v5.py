"""exp138: amphibian specialist v5 = v4 recipe + mismaresenka STRONG-label AnuraSet (replaces denden12 weak).

Same as v4 (Pantanal labeled-SC primary + train_audio + Pantanal gate + clean recipe),
but AnuraSet breadth uses STRONG labels (mismaresenka, 3-sec accurate) instead of weak (denden12).
AnuraSet subsampled to keep Pantanal primary. Compare gate col-AUC vs v4 (denden12) to settle which is right.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
NB = Path("experiment/exp138/notebook/nb_exp138_v5.ipynb")
NB.parent.mkdir(parents=True, exist_ok=True)

def cell(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

C1 = """# === exp138 cell1: setup ===
import os, sys, glob, re, time, random, json
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torchaudio, soundfile as sf, librosa
from sklearn.metrics import roc_auc_score
!pip install -q timm
import timm
SR=32000; N_FFT=2048; HOP=512; N_MELS=256; FMIN=20; FMAX=16000; TOP_DB=80
WIN=SR*5; SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV="cuda" if torch.cuda.is_available() else "cpu"
EPOCHS=16; BS=64; LR=1e-3; WD=1e-2; BACKBONE="efficientnet_b0"; ANU_CAP=3000
print("torch",torch.__version__,"cuda",torch.cuda.is_available())
"""

C2 = """# === exp138 cell2: paths + 35 labels + AnuraSet code map ===
def ff(c,m):
    for p in c:
        p=Path(p)
        if p.exists() and (list(p.rglob(m)) or (p/m).exists()): return p
    return None
COMP=ff(["/kaggle/input/competitions/birdclef-2026","/kaggle/input/birdclef-2026"],"taxonomy.csv")
_ab=ff(["/kaggle/input/anuraset-preprocessed","/kaggle/input/datasets/mismaresenka/anuraset-preprocessed"],"metadata.csv")
ANU=next(iter(Path(_ab).rglob("metadata.csv"))).parent   # robust to mount
print("COMP",COMP,"\\nANU(strong)",ANU)
tax=pd.read_csv(COMP/"taxonomy.csv")
AMP=sorted(tax[tax["class_name"]=="Amphibia"]["primary_label"].astype(str).tolist())
A2I={a:i for i,a in enumerate(AMP)}; NC=len(AMP); print("amphibian classes",NC)
train=pd.read_csv(COMP/"train.csv"); train["primary_label"]=train["primary_label"].astype(str)
CODE2IDX={c:A2I[l] for c,l in {
 "AMEPIC":"64898","BOALUN":"555123","BOARAN":"555146","DENMIN":"65377","DENNAN":"65380",
 "ELABIC":"25092","LEPELE":"22967","LEPFUS":"22973","LEPLAB":"22983","LEPLAT":"1176823",
 "LEPPOD":"22961","PHYALB":"23158","PHYNAT":"476521","PHYSAU":"23724","PITAZU":"517063",
 "SCIFUS":"24287","SCIFUV":"24285","SCINAS":"24279"}.items() if l in A2I}
"""

C3 = """# === exp138 cell3: Pantanal primary + STRONG AnuraSet (subsampled) ===
amp_set=set(AMP)
def t2s(v):
    s=str(v).strip()
    if ":" in s:
        p=[float(x) for x in s.split(":")]; return int(round(p[0]*3600+p[1]*60+p[2] if len(p)==3 else p[0]*60+p[1]))
    return int(round(float(v)))
# labeled-SC split by file (gate = held-out)
SCdir=COMP/"train_soundscapes"; ldf=pd.read_csv(COMP/"train_soundscapes_labels.csv")
sc_files=sorted(set(ldf["filename"])); rng=np.random.RandomState(SEED); perm=rng.permutation(len(sc_files))
n_hold=max(8,int(len(sc_files)*0.2)); hold_files=set(np.array(sc_files)[perm[:n_hold]])
sc_path={p.name:p for p in SCdir.glob("*.ogg")}; sc_cache={}
def sc_audio(fn):
    if fn not in sc_cache:
        w,sr=sf.read(str(sc_path[fn]),dtype="float32",always_2d=False)
        if getattr(w,'ndim',1)>1: w=w.mean(1)
        if sr!=SR: w=librosa.resample(w,orig_sr=sr,target_sr=SR)
        sc_cache[fn]=w
    return sc_cache[fn]
sc_train=[]; gate_items=[]
for _,r in ldf.iterrows():
    fn=str(r["filename"])
    if fn not in sc_path: continue
    st=t2s(r["start"]); y=np.zeros(NC,dtype=np.float32)
    for t in re.split(r"[;,]",str(r["primary_label"])):
        t=t.strip()
        if t in amp_set: y[A2I[t]]=1.0
    (gate_items if fn in hold_files else sc_train).append((fn,st,y))
print(f"SC: train {len(sc_train)} / gate {len(gate_items)}")
# train_audio amphibians
ta=COMP/"train_audio"
def sec(s): return [t for t in re.findall(r"[A-Za-z0-9]+",str(s)) if t in amp_set]
ta_items=[]
for _,r in train[train["primary_label"].isin(amp_set)].iterrows():
    p=ta/r["filename"]
    if not p.exists(): continue
    y=np.zeros(NC,dtype=np.float32); y[A2I[r["primary_label"]]]=1.0
    for s in sec(r.get("secondary_labels","")): y[A2I[s]]=1.0
    ta_items.append((str(p),y))
# AnuraSet STRONG (mismaresenka), train subset, fname-key (verified), subsample
meta=pd.read_csv(ANU/"metadata.csv"); anu_aud={Path(p).stem:p for p in glob.glob(str(ANU/"**/*.wav"),recursive=True)}
print("AnuraSet strong wavs:",len(anu_aud))
code_cols=[c for c in CODE2IDX if c in meta.columns]; anu_pos=[]; anu_neg=[]
for _,r in meta.iterrows():
    if str(r.get("subset",""))=="test": continue
    key=f"{r['fname']}_{int(r['min_t'])}_{int(r['max_t'])}"; p=anu_aud.get(key)
    if p is None: continue
    y=np.zeros(NC,dtype=np.float32)
    for c in code_cols:
        if r[c]>0: y[CODE2IDX[c]]=1.0
    (anu_pos if y.sum()>0 else anu_neg).append((p,y))
random.shuffle(anu_pos); random.shuffle(anu_neg)
anu_items=anu_pos[:ANU_CAP]+anu_neg[:ANU_CAP//2]
print(f"train_audio {len(ta_items)} | AnuraSet strong pos {len(anu_pos)} neg {len(anu_neg)} -> used {len(anu_items)}")
assert len(sc_train)>500 and len(gate_items)>100 and len(anu_pos)>1000, "FAIL-FAST: data load problem"
Yg=np.stack([y for _,_,y in gate_items]); gate_eval=[j for j in range(NC) if 0<Yg[:,j].sum()<len(Yg)]
print("gate evaluable amphibians:",len(gate_eval))
"""

C4 = """# === exp138 cell4: dataset (clean) ===
melspec=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,f_min=FMIN,f_max=FMAX,power=2.0)
to_db=torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)
def crop(w,train=True):
    if len(w)<WIN: w=np.tile(w,int(np.ceil(WIN/max(1,len(w)))))[:WIN]
    else:
        st=random.randint(0,len(w)-WIN) if train else max(0,(len(w)-WIN)//2); w=w[st:st+WIN]
    return w[:WIN].astype(np.float32)
def load_path(p):
    try: w,sr=sf.read(p,dtype="float32",always_2d=False)
    except Exception: w,sr=librosa.load(p,sr=SR,mono=True)
    if getattr(w,'ndim',1)>1: w=w.mean(1)
    if sr!=SR: w=librosa.resample(w,orig_sr=sr,target_sr=SR)
    return crop(w,True)
class TrainDS(torch.utils.data.Dataset):
    def __init__(s,items): s.items=items
    def __len__(s): return len(s.items)
    def __getitem__(s,i):
        it=s.items[i]
        if it[0]=='sc':
            w=sc_audio(it[1]); a=it[2]*SR; ch=w[a:a+WIN]
            if len(ch)<WIN: ch=np.pad(ch,(0,WIN-len(ch)))
            x=ch[:WIN].astype(np.float32); y=it[3]
        else: x=load_path(it[1]); y=it[2]
        return torch.from_numpy(x),torch.from_numpy(y)
train_items=[('sc',fn,st,y) for (fn,st,y) in sc_train]+[('p',p,y) for (p,y) in ta_items]+[('p',p,y) for (p,y) in anu_items]
def collate(b):
    ws=torch.stack([x[0] for x in b]); ys=torch.stack([x[1] for x in b])
    m=to_db(melspec(ws)); mu=m.mean((1,2),keepdim=True); sd=m.std((1,2),keepdim=True)+1e-6
    return ((m-mu)/sd).unsqueeze(1),ys
tl=torch.utils.data.DataLoader(TrainDS(train_items),batch_size=BS,shuffle=True,num_workers=4,collate_fn=collate,drop_last=True,pin_memory=True)
gate_wavs=[]
for fn,st,y in gate_items:
    w=sc_audio(fn); a=st*SR; ch=w[a:a+WIN]
    if len(ch)<WIN: ch=np.pad(ch,(0,WIN-len(ch)))
    gate_wavs.append(ch[:WIN].astype(np.float32))
print("total train",len(train_items),"gate",len(gate_wavs),"batches/ep",len(tl))
"""

C5 = """# === exp138 cell5: train, best by held-out labeled-SC col-AUC ===
model=timm.create_model(BACKBONE,pretrained=True,in_chans=1,num_classes=NC).to(DEV)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS*len(tl))
scaler=torch.amp.GradScaler('cuda'); crit=nn.BCEWithLogitsLoss()
def gate_auc():
    model.eval(); P=[]
    with torch.no_grad():
        for b in range(0,len(gate_wavs),64):
            ws=torch.from_numpy(np.stack(gate_wavs[b:b+64])); m=to_db(melspec(ws))
            mu=m.mean((1,2),keepdim=True); sd=m.std((1,2),keepdim=True)+1e-6
            with torch.amp.autocast('cuda'): o=model(((m-mu)/sd).unsqueeze(1).to(DEV))
            P.append(torch.sigmoid(o).float().cpu().numpy())
    P=np.concatenate(P); return float(np.mean([roc_auc_score(Yg[:,j],P[:,j]) for j in gate_eval]))
best=-1
for ep in range(EPOCHS):
    model.train(); t0=time.time(); tot=0
    for bi,(m,y) in enumerate(tl):
        m=m.to(DEV); y=y.to(DEV); opt.zero_grad()
        with torch.amp.autocast('cuda'): loss=crit(model(m),y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tot+=loss.item()
        if bi%100==0: print(f"  ep{ep} step{bi}/{len(tl)} loss={loss.item():.4f}",flush=True)
    g=gate_auc(); print(f"[ep{ep}] loss={tot/len(tl):.4f} GATE labeled-SC col-AUC={g:.4f} (v1=0.70) {time.time()-t0:.0f}s",flush=True)
    if g>best: best=g; torch.save(model.state_dict(),"/kaggle/working/amphib_b0_v5.pth"); print(f"  ** BEST {best:.4f}")
print("best gate col-AUC:",best)
"""

C6 = """# === exp138 cell6: export ===
T=1+WIN//HOP
json.dump({"labels":AMP,"n_mels":N_MELS,"T":int(T),"sr":SR,"hop":HOP,"n_fft":N_FFT,"fmin":FMIN,"fmax":FMAX,"top_db":TOP_DB,"win":WIN,"backbone":BACKBONE,"best_gate_auc":float(best)},open("/kaggle/working/amphib_v5_meta.json","w"),indent=2)
print("exported amphib_b0_v5.pth; best held-out labeled-SC col-AUC =",best,"(v1=0.70, denden12-v4 for compare)")
"""

CELLS=[(C1,"setup"),(C2,"paths"),(C3,"data"),(C4,"ds"),(C5,"train"),(C6,"export")]
nb={"cells":[cell(s,c) for s,c in CELLS],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
NB.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding="utf-8")
print(f"Wrote {NB}")
for s,c in CELLS:
    clean="\n".join("# "+ln if ln.lstrip().startswith(("!","%")) else ln for ln in s.splitlines())
    try: compile(clean,f"<{c}>","exec"); print(f"  [OK] {c}")
    except SyntaxError as e: print(f"  [FAIL] {c}: {e}")
