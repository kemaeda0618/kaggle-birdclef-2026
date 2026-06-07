"""exp133: Phase-2 GATE for v2 — does amphib_b0_v2 (torch) improve OOF amphibian rank vs base?
Same as exp129 but amphib model = v2 (.pth via torch+timm) instead of v1 OV. Compare to exp129's v1 numbers."""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
NB = Path("experiment/exp133/notebook/nb_exp133_verify_v2.ipynb")
NB.parent.mkdir(parents=True, exist_ok=True)

def cell(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

C1 = """# === exp133 cell1: imports + OV (for Tucker/e106) ===
import os, sys, glob, re, time, json
from pathlib import Path
import numpy as np, pandas as pd
import torch, torchaudio, soundfile as sf, librosa, timm
WHEEL = sorted(glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True))
try:
    import openvino as ov
except ImportError:
    wd = str(Path(WHEEL[0]).parent)
    !pip install -q --no-deps {wd}/openvino-*.whl {wd}/openvino_telemetry-*.whl
    import openvino as ov
print("openvino", ov.__version__, "timm", timm.__version__)
SR=32000; N_FFT=2048; HOP=512; N_MELS=256; FMIN=20; FMAX=16000; TOP_DB=80
WIN=SR*5; N_WIN_CAP=8
def sig(x): return 1/(1+np.exp(-np.clip(x,-50,50)))
"""

C2 = """# === exp133 cell2: paths + labels + amphib v2 meta ===
def ff(c,m):
    for p in c:
        p=Path(p)
        if p.exists() and (list(p.rglob(m)) or (p/m).exists()): return p
    return None
COMP=ff(["/kaggle/input/competitions/birdclef-2026","/kaggle/input/birdclef-2026"],"taxonomy.csv")
TUCK=ff(["/kaggle/input/notebooks/maekeso/birdclef2026-tucker-sed-ov","/kaggle/input/birdclef2026-tucker-sed-ov"],"sed_fold0.xml")
E106=ff(["/kaggle/input/notebooks/maekeso/birdclef2026-e106-3fold-ov","/kaggle/input/birdclef2026-e106-3fold-ov"],"exp106_fold0.xml")
AMPH=ff(["/kaggle/input/birdclef2026-amphib-b0-v2","/kaggle/input/datasets/maekeso/birdclef2026-amphib-b0-v2"],"amphib_b0_v2.pth")
print("COMP",COMP,"\\nTUCK",TUCK,"\\nE106",E106,"\\nAMPH",AMPH)
ss=pd.read_csv(COMP/"sample_submission.csv"); LABELS=ss.columns[1:].tolist(); L2I={l:i for i,l in enumerate(LABELS)}; NC=len(LABELS)
tax=pd.read_csv(COMP/"taxonomy.csv")
amphib_meta=json.load(open(AMPH/"amphib_v2_meta.json")); AMP=amphib_meta["labels"]
AMP_COLS=np.array([L2I[a] for a in AMP])
print("amphib v2 labels",len(AMP))
"""

C3 = """# === exp133 cell3: OOF amphibian XC clips (train-deduped) ===
xc=glob.glob("/kaggle/input/datasets/**/*.mp3",recursive=True) or glob.glob("/kaggle/input/**/*.mp3",recursive=True)
train=pd.read_csv(COMP/"train.csv"); train["primary_label"]=train["primary_label"].astype(str)
label_set=set(str(x) for x in tax["primary_label"])
def norm(s): return re.sub(r"[^a-z0-9]","",str(s).lower())
sci2label={norm(r["scientific_name"]):str(r["primary_label"]) for _,r in tax.iterrows()}
def xcid(s):
    m=re.search(r"XC\\s*0*([0-9]{3,})",str(s),flags=re.I); return m.group(1) if m else None
train_ids=set(filter(None,(xcid(f) for f in train["filename"]))); amp_set=set(AMP)
def infer(path):
    for raw in [Path(path).stem]+list(Path(path).parts):
        if str(raw) in label_set: return str(raw)
        n=norm(raw)
        if n in sci2label: return sci2label[n]
    return None
rows=[]
for p in xc:
    sp=infer(p); rid=xcid(p)
    if sp in amp_set and not (rid and rid in train_ids):
        rows.append({"path":p,"species":sp,"rec_id":rid})
DIAG=pd.DataFrame(rows).drop_duplicates("path").reset_index(drop=True)
DIAG["train_count"]=DIAG["species"].map(train["primary_label"].value_counts().to_dict()).fillna(0).astype(int)
print("OOF amphibian clips",len(DIAG),"species",DIAG["species"].nunique())
"""

C4 = """# === exp133 cell4: models (Tucker/e106 OV + amphib v2 torch) + mel ===
core=ov.Core()
tuck=[core.compile_model(str(p),"CPU") for p in sorted(TUCK.glob("sed_fold*.xml"))]
e106=[core.compile_model(str(E106/f"exp106_fold{f}.xml"),"CPU") for f in [0,1,2]]
amphib=timm.create_model("efficientnet_b0",pretrained=False,in_chans=1,num_classes=len(AMP))
amphib.load_state_dict(torch.load(str(AMPH/"amphib_b0_v2.pth"),map_location="cpu")); amphib.eval()
print("tuck",len(tuck),"e106",len(e106),"amphib v2 ok")
mt=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,f_min=FMIN,f_max=FMAX,power=2.0)
dt=torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)
def mel_torch(chs):
    w=torch.from_numpy(np.stack(chs).astype(np.float32)); m=dt(mt(w))
    mu=m.mean((1,2),keepdim=True); sd=m.std((1,2),keepdim=True)+1e-6
    return ((m-mu)/sd).unsqueeze(1).numpy().astype(np.float32)
def mel_lib(chs):
    out=[]
    for x in chs:
        s=librosa.feature.melspectrogram(y=x,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX,power=2.0)
        s=librosa.power_to_db(s,top_db=TOP_DB); s=(s-s.mean())/(s.std()+1e-6); out.append(s)
    return np.stack(out)[:,None].astype(np.float32)
def runm(model,mels):
    out=[]
    for b in range(0,len(mels),24):
        o=model(mels[b:b+24]); clip=o[model.outputs[0]]; fr=o[model.outputs[1]].max(1)
        out.append((0.5*sig(clip)+0.5*sig(fr)).astype(np.float32))
    return np.concatenate(out)
def run_amphib(MT):
    with torch.no_grad():
        out=[]
        for b in range(0,len(MT),24):
            o=amphib(torch.from_numpy(MT[b:b+24])); out.append(torch.sigmoid(o).numpy().astype(np.float32))
    return np.concatenate(out)
def load_windows(path):
    try: w,sr=sf.read(str(path),dtype="float32",always_2d=False)
    except Exception: w,sr=librosa.load(str(path),sr=SR,mono=True)
    if getattr(w,"ndim",1)>1: w=w.mean(1)
    if sr!=SR: w=librosa.resample(w,orig_sr=sr,target_sr=SR)
    if len(w)<WIN: w=np.concatenate([w,np.zeros(WIN-len(w),dtype=np.float32)])
    n=min(N_WIN_CAP,max(1,len(w)//WIN)); idx=np.linspace(0,max(0,len(w)-WIN),n).astype(int)
    return np.stack([w[i:i+WIN] for i in idx]).astype(np.float32)
"""

C5 = """# === exp133 cell5: blend sweep + rank compare ===
ALPHAS=[0.0,0.3,0.5,0.7,1.0]; recs=[]; t0=time.time()
for k,r in DIAG.iterrows():
    try: chs=load_windows(r["path"])
    except Exception: continue
    ML=mel_lib(chs); MT=mel_torch(chs)
    pt=np.mean([runm(m,ML) for m in tuck],axis=0)
    pe=np.mean([runm(m,MT) for m in e106],axis=0)
    base=(0.5*pt+0.5*pe).max(0)
    q=run_amphib(MT).max(0)
    ci=L2I[r["species"]]; row={"species":r["species"],"train_count":int(r["train_count"])}
    for a in ALPHAS:
        p=base.copy(); p[AMP_COLS]=(1-a)*base[AMP_COLS]+a*q
        row[f"rank_a{a}"]=int((p>p[ci]).sum())+1; row[f"ptrue_a{a}"]=float(p[ci])
    recs.append(row)
D=pd.DataFrame(recs); D.to_csv("/kaggle/working/exp133_verify_v2.csv",index=False)
print(f"scored {len(D)} OOF amphibian clips in {time.time()-t0:.0f}s")
"""

C6 = """# === exp133 cell6: GATE report (compare to exp129 v1: top5 0.714 @ a=0.5) ===
ALPHAS=[0.0,0.3,0.5,0.7,1.0]
print("=== v2 rank by alpha (a=0 = base) — lower rank / higher top5 is better ===")
print(f"{'alpha':6} {'med_rank':9} {'mean_rank':10} {'top1':6} {'top5':6} {'mean_ptrue':10}")
for a in ALPHAS:
    rk=D[f"rank_a{a}"]; pt=D[f"ptrue_a{a}"]
    print(f"{a:<6} {rk.median():9.1f} {rk.mean():10.1f} {(rk==1).mean():6.3f} {(rk<=5).mean():6.3f} {pt.mean():10.3f}")
print("\\n=== compare: v1 (exp129) was top5=0.714 @ a=0.5. v2 GO if top5 >= that ===")
b=D['rank_a0.0'].median()
for a in [0.3,0.5,0.7]:
    print(f"  v2 alpha={a}: top5={ (D[f'rank_a{a}']<=5).mean():.3f}  median rank {b:.0f}->{D[f'rank_a{a}'].median():.0f}")
"""

CELLS=[(C1,"imp"),(C2,"path"),(C3,"oof"),(C4,"model"),(C5,"sweep"),(C6,"report")]
nb={"cells":[cell(s,c) for s,c in CELLS],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},
    "nbformat":4,"nbformat_minor":5}
NB.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding="utf-8")
print(f"Wrote {NB}")
for s,c in CELLS:
    clean="\n".join("# "+ln if ln.lstrip().startswith(("!","%")) else ln for ln in s.splitlines())
    try: compile(clean,f"<{c}>","exec"); print(f"  [OK] {c}")
    except SyntaxError as e: print(f"  [FAIL] {c}: {e}")
