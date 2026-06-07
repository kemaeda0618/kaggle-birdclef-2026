"""exp129: Phase-2 GATE — does amphib_b0 (surgical on amphibian cols) improve OOF amphibian rank?

On exp127 OOF amphibian XC clips (true OOF, train-deduped):
  - current ensemble (Tucker 5-fold + e106 3-fold) -> P_base (234-way)
  - amphib_b0 (35-way) -> Q, mapped into the 35 amphibian columns
  - blend P[:,amph] = (1-a)*P_base + a*Q  for a in {0,.3,.5,.7,1.0}
  - re-rank true species in 234, compare rank_base vs rank_blend
GATE: if blend lowers (improves) median rank of weak amphibians -> GO for surgical integration.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path("experiment/exp129/notebook/nb_exp129_verify.ipynb")
NB.parent.mkdir(parents=True, exist_ok=True)


def cell(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}


C1 = """# === exp129 cell1: imports + OV ===
import os, sys, glob, re, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torchaudio, soundfile as sf, librosa
WHEEL = sorted(glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True))
try:
    import openvino as ov
except ImportError:
    wd = str(Path(WHEEL[0]).parent)
    !pip install -q --no-deps {wd}/openvino-*.whl {wd}/openvino_telemetry-*.whl
    import openvino as ov
print("openvino", ov.__version__)
SR=32000; N_FFT=2048; HOP=512; N_MELS=256; FMIN=20; FMAX=16000; TOP_DB=80
WIN=SR*5; N_WIN_CAP=8
def sig(x): return 1/(1+np.exp(-np.clip(x,-50,50)))
"""

C2 = """# === exp129 cell2: paths + labels + amphib meta ===
def ff(c,m):
    for p in c:
        p=Path(p)
        if p.exists() and (list(p.rglob(m)) or (p/m).exists()): return p
    return None
COMP = ff(["/kaggle/input/competitions/birdclef-2026","/kaggle/input/birdclef-2026"],"taxonomy.csv")
TUCK = ff(["/kaggle/input/notebooks/maekeso/birdclef2026-tucker-sed-ov","/kaggle/input/birdclef2026-tucker-sed-ov"],"sed_fold0.xml")
E106 = ff(["/kaggle/input/notebooks/maekeso/birdclef2026-e106-3fold-ov","/kaggle/input/birdclef2026-e106-3fold-ov"],"exp106_fold0.xml")
AMPH = ff(["/kaggle/input/birdclef2026-amphib-b0-ov","/kaggle/input/datasets/maekeso/birdclef2026-amphib-b0-ov"],"amphib_b0.xml")
print("COMP",COMP,"\\nTUCK",TUCK,"\\nE106",E106,"\\nAMPH",AMPH)

ss=pd.read_csv(COMP/"sample_submission.csv"); LABELS=ss.columns[1:].tolist(); L2I={l:i for i,l in enumerate(LABELS)}; NC=len(LABELS)
tax=pd.read_csv(COMP/"taxonomy.csv"); l2t=dict(zip(tax["primary_label"].astype(str),tax["class_name"].astype(str)))
amphib_meta=json.load(open(AMPH/"amphib_meta.json")); AMP=amphib_meta["labels"]
# map amphib output idx -> 234 column idx
AMP_COLS=np.array([L2I[a] for a in AMP])   # length 35
print("amphib labels:",len(AMP),"T",amphib_meta["T"])
"""

C3 = """# === exp129 cell3: OOF amphibian XC clips (train-deduped) ===
xc=glob.glob("/kaggle/input/datasets/**/*.mp3",recursive=True) or glob.glob("/kaggle/input/**/*.mp3",recursive=True)
train=pd.read_csv(COMP/"train.csv"); train["primary_label"]=train["primary_label"].astype(str)
label_set=set(str(x) for x in tax["primary_label"])
def norm(s): return re.sub(r"[^a-z0-9]","",str(s).lower())
sci2label={norm(r["scientific_name"]):str(r["primary_label"]) for _,r in tax.iterrows()}
def xcid(s):
    m=re.search(r"XC\\s*0*([0-9]{3,})",str(s),flags=re.I); return m.group(1) if m else None
train_ids=set(filter(None,(xcid(f) for f in train["filename"])))
amp_set=set(AMP)
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
print("OOF amphibian clips:",len(DIAG),"species",DIAG["species"].nunique())
"""

C4 = """# === exp129 cell4: models + mel ===
core=ov.Core()
tuck=[core.compile_model(str(p),"CPU") for p in sorted(TUCK.glob("sed_fold*.xml"))]
e106=[core.compile_model(str(E106/f"exp106_fold{f}.xml"),"CPU") for f in [0,1,2]]
amphib=core.compile_model(str(AMPH/"amphib_b0.xml"),"CPU")
print("tuck",len(tuck),"e106",len(e106),"amphib ok")

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
def run_amphib(mels):
    out=[]
    for b in range(0,len(mels),24):
        o=amphib(mels[b:b+24]); out.append(sig(o[amphib.outputs[0]]).astype(np.float32))
    return np.concatenate(out)   # [n,35]
def load_windows(path):
    try: w,sr=sf.read(str(path),dtype="float32",always_2d=False)
    except Exception: w,sr=librosa.load(str(path),sr=SR,mono=True)
    if getattr(w,"ndim",1)>1: w=w.mean(1)
    if sr!=SR: w=librosa.resample(w,orig_sr=sr,target_sr=SR)
    if len(w)<WIN: w=np.concatenate([w,np.zeros(WIN-len(w),dtype=np.float32)])
    n=min(N_WIN_CAP,max(1,len(w)//WIN)); idx=np.linspace(0,max(0,len(w)-WIN),n).astype(int)
    return np.stack([w[i:i+WIN] for i in idx]).astype(np.float32)
"""

C5 = """# === exp129 cell5: blend sweep + rank compare ===
ALPHAS=[0.0,0.3,0.5,0.7,1.0]
recs=[]; t0=time.time()
for k,r in DIAG.iterrows():
    try: chs=load_windows(r["path"])
    except Exception: continue
    ML=mel_lib(chs); MT=mel_torch(chs)
    pt=np.mean([runm(m,ML) for m in tuck],axis=0)
    pe=np.mean([runm(m,MT) for m in e106],axis=0)
    base=(0.5*pt+0.5*pe).max(0)           # 234 presence
    q=run_amphib(MT).max(0)               # 35 presence (amphib output order = AMP)
    ci=L2I[r["species"]]
    row={"species":r["species"],"train_count":int(r["train_count"])}
    for a in ALPHAS:
        p=base.copy(); p[AMP_COLS]=(1-a)*base[AMP_COLS]+a*q
        row[f"rank_a{a}"]=int((p>p[ci]).sum())+1
        row[f"ptrue_a{a}"]=float(p[ci])
    recs.append(row)
D=pd.DataFrame(recs); D.to_csv("/kaggle/working/exp129_verify.csv",index=False)
print(f"scored {len(D)} OOF amphibian clips in {time.time()-t0:.0f}s")
"""

C6 = """# === exp129 cell6: GATE report ===
ALPHAS=[0.0,0.3,0.5,0.7,1.0]
print("=== rank by alpha (a=0 is current ensemble baseline) — lower is better ===")
print(f"{'alpha':6} {'med_rank':9} {'mean_rank':10} {'top1':6} {'top5':6} {'mean_ptrue':10}")
for a in ALPHAS:
    rk=D[f"rank_a{a}"]; pt=D[f"ptrue_a{a}"]
    print(f"{a:<6} {rk.median():9.1f} {rk.mean():10.1f} {(rk==1).mean():6.3f} {(rk<=5).mean():6.3f} {pt.mean():10.3f}")

print("\\n=== per-species: rank a=0 (base) vs best alpha ===")
best_a=0.5
g=D.groupby(["species","train_count"]).agg(n=("rank_a0.0","count"),
    base_med=("rank_a0.0","median"), blend_med=(f"rank_a{best_a}","median")).reset_index()
g["delta"]=g["base_med"]-g["blend_med"]   # positive = improved
g=g.sort_values("delta",ascending=False)
print(f"(blend alpha={best_a}; delta>0 = improved)")
print(f"{'species':10} {'train':6} {'n':3} {'base_med':9} {'blend_med':10} {'delta':6}")
for _,r in g.iterrows():
    print(f"{r['species']:10} {int(r['train_count']):6} {int(r['n']):3} {r['base_med']:9.0f} {r['blend_med']:10.0f} {r['delta']:+6.0f}")

print("\\n=== VERDICT ===")
base_med=D['rank_a0.0'].median()
for a in [0.3,0.5,0.7,1.0]:
    d=base_med - D[f'rank_a{a}'].median()
    print(f"  alpha={a}: median rank {base_med:.0f} -> {D[f'rank_a{a}'].median():.0f} (improve {d:+.0f})")
print("  GO if any alpha clearly lowers median rank / raises top5 vs alpha=0")
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
