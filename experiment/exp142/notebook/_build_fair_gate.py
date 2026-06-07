"""exp142: FAIR FP gate — v1/v3/v5 evaluated on the SAME held-out labeled-SC files (v5's 20% split).

Fixes the apples-to-oranges issue: v5's 0.798 was on held-out 20%, v1's 0.70 (exp136) was on FULL set.
Here all three specialists are scored on the IDENTICAL held-out 20% files (replicating v5/v4 split, SEED=42).
All three are CLEAN on these files: v1/v3 never trained on labeled-SC; v5 trained on the other 80%.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
NB = Path("experiment/exp142/notebook/nb_exp142_fair_gate.ipynb")
NB.parent.mkdir(parents=True, exist_ok=True)

def cell(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

C1 = """# === exp142 cell1: imports + models ===
import os, re, json, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torchaudio, soundfile as sf, librosa, timm
from sklearn.metrics import roc_auc_score
SR=32000; N_FFT=2048; HOP=512; N_MELS=256; FMIN=20; FMAX=16000; TOP_DB=80; WIN=SR*5; SEED=42
def ff(c,m):
    for p in c:
        p=Path(p)
        if p.exists() and (list(p.rglob(m)) or (p/m).exists()): return p
    return None
COMP=ff(["/kaggle/input/competitions/birdclef-2026","/kaggle/input/birdclef-2026"],"taxonomy.csv")
V1=ff(["/kaggle/input/birdclef2026-amphib-b0-ov","/kaggle/input/datasets/maekeso/birdclef2026-amphib-b0-ov"],"amphib_b0.pth")
V3=ff(["/kaggle/input/birdclef2026-amphib-b0-v3","/kaggle/input/datasets/maekeso/birdclef2026-amphib-b0-v3"],"amphib_b0_v3.pth")
V5=ff(["/kaggle/input/birdclef2026-amphib-b0-v5","/kaggle/input/datasets/maekeso/birdclef2026-amphib-b0-v5"],"amphib_b0_v5.pth")
print("V1",V1,"\\nV3",V3,"\\nV5",V5)
tax=pd.read_csv(COMP/"taxonomy.csv")
AMP=sorted(tax[tax["class_name"]=="Amphibia"]["primary_label"].astype(str).tolist()); A2I={a:i for i,a in enumerate(AMP)}; NC=len(AMP)
def load_model(d,fn):
    m=timm.create_model("efficientnet_b0",pretrained=False,in_chans=1,num_classes=NC)
    m.load_state_dict(torch.load(str(d/fn),map_location="cpu")); m.eval(); return m
models={"v1":load_model(V1,"amphib_b0.pth"),"v3":load_model(V3,"amphib_b0_v3.pth"),"v5":load_model(V5,"amphib_b0_v5.pth")}
print("models loaded:",list(models))
"""

C2 = """# === exp142 cell2: replicate v5 held-out split + load held-out segments ===
SCdir=COMP/"train_soundscapes"; ldf=pd.read_csv(COMP/"train_soundscapes_labels.csv")
sc_files=sorted(set(ldf["filename"])); rng=np.random.RandomState(SEED); perm=rng.permutation(len(sc_files))
n_hold=max(8,int(len(sc_files)*0.2)); hold_files=set(np.array(sc_files)[perm[:n_hold]])   # ★ same as v4/v5
print(f"held-out files: {len(hold_files)} / {len(sc_files)}")
sc_path={p.name:p for p in SCdir.glob("*.ogg")}; cache={}
def t2s(v):
    s=str(v).strip()
    if ":" in s:
        p=[float(x) for x in s.split(":")]; return int(round(p[0]*3600+p[1]*60+p[2] if len(p)==3 else p[0]*60+p[1]))
    return int(round(float(v)))
def aud(fn):
    if fn not in cache:
        w,sr=sf.read(str(sc_path[fn]),dtype="float32",always_2d=False)
        if getattr(w,'ndim',1)>1: w=w.mean(1)
        if sr!=SR: w=librosa.resample(w,orig_sr=sr,target_sr=SR)
        cache[fn]=w
    return cache[fn]
segs=[]; Y=[]; amp_set=set(AMP)
for _,r in ldf.iterrows():
    fn=str(r["filename"])
    if fn not in hold_files or fn not in sc_path: continue
    st=t2s(r["start"]); w=aud(fn); ch=w[st*SR:st*SR+WIN]
    if len(ch)<WIN: ch=np.concatenate([ch,np.zeros(WIN-len(ch),dtype=np.float32)])
    y=np.zeros(NC,dtype=np.float32)
    for t in re.split(r"[;,]",str(r["primary_label"])):
        t=t.strip()
        if t in amp_set: y[A2I[t]]=1.0
    segs.append(ch[:WIN].astype(np.float32)); Y.append(y)
Y=np.stack(Y); evaluable=[j for j in range(NC) if 0<Y[:,j].sum()<len(Y)]
print(f"held-out segments {len(segs)} | evaluable amphibians {len(evaluable)} | pos rows {int((Y.sum(1)>0).sum())} neg {int((Y.sum(1)==0).sum())}")
assert len(segs)>100, "FAIL-FAST"
"""

C3 = """# === exp142 cell3: run all + per-species + macro col-AUC ===
mt=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,f_min=FMIN,f_max=FMAX,power=2.0)
dt=torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)
def run(model):
    P=[]
    with torch.no_grad():
        for b in range(0,len(segs),64):
            ws=torch.from_numpy(np.stack(segs[b:b+64])); m=dt(mt(ws))
            mu=m.mean((1,2),keepdim=True); sd=m.std((1,2),keepdim=True)+1e-6
            P.append(torch.sigmoid(model(((m-mu)/sd).unsqueeze(1))).numpy())
    return np.concatenate(P)
P={k:run(m) for k,m in models.items()}
rows=[]
for j in evaluable:
    rows.append({"species":AMP[j],"pos":int(Y[:,j].sum()),
                 "v1":roc_auc_score(Y[:,j],P["v1"][:,j]),
                 "v3":roc_auc_score(Y[:,j],P["v3"][:,j]),
                 "v5":roc_auc_score(Y[:,j],P["v5"][:,j])})
R=pd.DataFrame(rows).sort_values("pos",ascending=False)
print("=== FAIR comparison: same held-out labeled-SC files, column AUC ===")
print(f"{'species':>10} {'pos':>5} {'v1':>6} {'v3':>6} {'v5':>6}")
for _,r in R.iterrows():
    print(f"{r['species']:>10} {int(r['pos']):>5} {r['v1']:>6.3f} {r['v3']:>6.3f} {r['v5']:>6.3f}")
print(f"\\nMACRO col-AUC (same held-out set): v1={R['v1'].mean():.4f}  v3={R['v3'].mean():.4f}  v5={R['v5'].mean():.4f}")
print(f"v5 beats v1 on {(R['v5']>R['v1']).sum()}/{len(R)} species")
R.to_csv("/kaggle/working/exp142_fair_gate.csv",index=False)
"""

CELLS=[(C1,"model"),(C2,"seg"),(C3,"report")]
nb={"cells":[cell(s,c) for s,c in CELLS],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
NB.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding="utf-8")
print(f"Wrote {NB}")
for s,c in CELLS:
    try: compile(s,f"<{c}>","exec"); print(f"  [OK] {c}")
    except SyntaxError as e: print(f"  [FAIL] {c}: {e}")
