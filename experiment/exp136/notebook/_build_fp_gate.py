"""exp136: FP-aware GATE — amphibian specialist STANDALONE column-AUC on labeled train_soundscapes.

Clean for the specialist (it trained on AnuraSet+train_audio, NOT train_soundscapes).
labeled-SC has 1478 segments incl 346 negatives -> measures column AUC = LB metric incl false positives.
Evaluates v1 (amphib-b0-ov) AND v3 (amphib-b0-v3). If v1 low -> confirms exp130 0.925 collapse.
If v3 high -> strong labels fixed FP -> safe to blend.
Base ensemble NOT used (it is contaminated on labeled-SC).
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
NB = Path("experiment/exp136/notebook/nb_exp136_fp_gate.ipynb")
NB.parent.mkdir(parents=True, exist_ok=True)

def cell(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

C1 = """# === exp136 cell1: imports ===
import os, sys, glob, re, json, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torchaudio, soundfile as sf, librosa, timm
from sklearn.metrics import roc_auc_score
SR=32000; N_FFT=2048; HOP=512; N_MELS=256; FMIN=20; FMAX=16000; TOP_DB=80; WIN=SR*5
print("timm", timm.__version__)
"""

C2 = """# === exp136 cell2: paths + models ===
def ff(c,m):
    for p in c:
        p=Path(p)
        if p.exists() and (list(p.rglob(m)) or (p/m).exists()): return p
    return None
COMP=ff(["/kaggle/input/competitions/birdclef-2026","/kaggle/input/birdclef-2026"],"taxonomy.csv")
V1=ff(["/kaggle/input/birdclef2026-amphib-b0-ov","/kaggle/input/datasets/maekeso/birdclef2026-amphib-b0-ov"],"amphib_b0.pth")
V3=ff(["/kaggle/input/birdclef2026-amphib-b0-v3","/kaggle/input/datasets/maekeso/birdclef2026-amphib-b0-v3"],"amphib_b0_v3.pth")
print("COMP",COMP,"\\nV1",V1,"\\nV3",V3)
m1=json.load(open(V1/"amphib_meta.json")); m3=json.load(open(V3/"amphib_v3_meta.json"))
AMP=m1["labels"]
assert AMP==m3["labels"], "v1/v3 label order mismatch!"   # fail-fast
A2I={a:i for i,a in enumerate(AMP)}; NC=len(AMP); print("amphib labels",NC)

def load_model(d, fn):
    mdl=timm.create_model("efficientnet_b0",pretrained=False,in_chans=1,num_classes=NC)
    mdl.load_state_dict(torch.load(str(d/fn),map_location="cpu")); mdl.eval(); return mdl
model_v1=load_model(V1,"amphib_b0.pth"); model_v3=load_model(V3,"amphib_b0_v3.pth")
print("models loaded")
"""

C3 = """# === exp136 cell3: labeled train_soundscapes -> segments + GT (amphibian multi-hot) ===
SC=COMP/"train_soundscapes"; LAB=COMP/"train_soundscapes_labels.csv"
ldf=pd.read_csv(LAB); print("labels rows",len(ldf),"cols",list(ldf.columns))
def t2s(v):
    s=str(v).strip()
    if ":" in s:
        p=[float(x) for x in s.split(":")]; return int(round(p[0]*3600+p[1]*60+p[2] if len(p)==3 else p[0]*60+p[1]))
    return int(round(float(s)))
amp_set=set(AMP)
# audio cache per file
sc_files={p.name:p for p in SC.glob("*.ogg")}
print("SC ogg files:",len(sc_files))
cache={}
def get_audio(fn):
    if fn not in cache:
        p=sc_files.get(fn) or sc_files.get(fn if fn.endswith('.ogg') else fn+'.ogg')
        w,sr=sf.read(str(p),dtype="float32",always_2d=False)
        if getattr(w,'ndim',1)>1: w=w.mean(1)
        if sr!=SR: w=librosa.resample(w,orig_sr=sr,target_sr=SR)
        cache[fn]=w
    return cache[fn]

segs=[]; Y=[]
for _,r in ldf.iterrows():
    fn=str(r["filename"]);
    if fn not in sc_files and (fn+'.ogg') not in sc_files and not fn.endswith('.ogg'): pass
    st=t2s(r["start"]);
    try: w=get_audio(fn)
    except Exception: continue
    a=st*SR; chunk=w[a:a+WIN]
    if len(chunk)<WIN: chunk=np.concatenate([chunk,np.zeros(WIN-len(chunk),dtype=np.float32)])
    y=np.zeros(NC,dtype=np.float32)
    for t in re.split(r"[;,]",str(r["primary_label"])):
        t=t.strip()
        if t in amp_set: y[A2I[t]]=1.0
    segs.append(chunk.astype(np.float32)); Y.append(y)
Y=np.stack(Y); print("segments",len(segs),"| amphibian-positive rows",int((Y.sum(1)>0).sum()),"| negatives",int((Y.sum(1)==0).sum()))
assert len(segs)>1000 and Y.sum()>0, "FAIL-FAST: SC segments/GT not loaded"
pos_per=Y.sum(0); evaluable=[j for j in range(NC) if 0<pos_per[j]<len(Y)]
print("evaluable amphibians (have positives):",len(evaluable))
"""

C4 = """# === exp136 cell4: run specialists standalone -> column AUC ===
mt=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,f_min=FMIN,f_max=FMAX,power=2.0)
dt=torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)
def to_mel(batch):
    w=torch.from_numpy(np.stack(batch)); m=dt(mt(w)); mu=m.mean((1,2),keepdim=True); sd=m.std((1,2),keepdim=True)+1e-6
    return ((m-mu)/sd).unsqueeze(1)
def run(model):
    out=[]
    with torch.no_grad():
        for b in range(0,len(segs),64):
            mel=to_mel(segs[b:b+64]); o=torch.sigmoid(model(mel)).numpy(); out.append(o.astype(np.float32))
    return np.concatenate(out)
t0=time.time(); P1=run(model_v1); P3=run(model_v3); print(f"inference done {time.time()-t0:.0f}s, shapes {P1.shape} {P3.shape}")
"""

C5 = """# === exp136 cell5: column-AUC report (v1 vs v3, FP-aware) ===
rows=[]
for j in evaluable:
    a1=roc_auc_score(Y[:,j],P1[:,j]); a3=roc_auc_score(Y[:,j],P3[:,j])
    rows.append({"species":AMP[j],"pos":int(Y[:,j].sum()),"auc_v1":a1,"auc_v3":a3,"delta":a3-a1})
R=pd.DataFrame(rows).sort_values("pos",ascending=False)
print("=== per-amphibian column AUC on labeled-SC (FP-aware, specialist standalone) ===")
print(f"{'species':>10} {'pos':>5} {'auc_v1':>7} {'auc_v3':>7} {'v3-v1':>6}")
for _,r in R.iterrows():
    print(f"{r['species']:>10} {int(r['pos']):>5} {r['auc_v1']:>7.3f} {r['auc_v3']:>7.3f} {r['delta']:>+6.3f}")
print(f"\\nMACRO amphibian column AUC:  v1={R['auc_v1'].mean():.4f}   v3={R['auc_v3'].mean():.4f}")
print(f"\\n=== VERDICT ===")
print(f"  v1 macro {R['auc_v1'].mean():.3f}: low (~0.5-0.7) => FP collapse (explains exp130 0.925)")
print(f"  v3 macro {R['auc_v3'].mean():.3f}: high (>~0.85) => strong labels fixed FP => safe to blend")
print(f"  v3 beats v1 on {(R['delta']>0).sum()}/{len(R)} species")
R.to_csv("/kaggle/working/exp136_fp_gate.csv",index=False)
"""

CELLS=[(C1,"imp"),(C2,"model"),(C3,"sc"),(C4,"run"),(C5,"report")]
nb={"cells":[cell(s,c) for s,c in CELLS],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},
    "nbformat":4,"nbformat_minor":5}
NB.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding="utf-8")
print(f"Wrote {NB}")
for s,c in CELLS:
    clean="\n".join("# "+ln if ln.lstrip().startswith(("!","%")) else ln for ln in s.splitlines())
    try: compile(clean,f"<{c}>","exec"); print(f"  [OK] {c}")
    except SyntaxError as e: print(f"  [FAIL] {c}: {e}")
