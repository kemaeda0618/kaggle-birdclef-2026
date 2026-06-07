"""exp124 v2: CLEAN per-class weakness analysis (no ground-truth-AUC saturation trap).

Approach (avoids the OV5b/OV5c saturation failure):
  - Run Tucker SED 5-fold OV + exp106 3-fold OV on labeled train_soundscapes (66 files)
  - For EACH class, compute:
      (a) prediction level: max / mean / p90 over the 739 windows  -> "collapsed" species (~0)
      (b) Tucker-vs-e106 rank correlation -> species where streams DISAGREE (low confidence)
      (c) ground-truth AUC per class (informative even if inflated; we read RELATIVE)
  - Aggregate by taxon (Aves/Insecta/Amphibia/Mammalia/Reptilia)
  - Output: which species are weak/collapsed/disagreed -> targets for per-class boost

This is ground-truth-LIGHT (level + disagreement need no labels) so saturation doesn't hide it.
Light: only 2 OV streams (no heavy NB4). ~10 min run.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB = Path("experiment/exp124/notebook/nb_exp124_clean.ipynb")
NB.parent.mkdir(parents=True, exist_ok=True)


def cell(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}


C1 = """# === exp124: clean per-class weakness analysis ===
import os, sys, glob, re, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torchaudio, soundfile as sf, librosa
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

WHEEL = sorted(glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True))
try:
    import openvino as ov
except ImportError:
    wd = str(Path(WHEEL[0]).parent)
    !pip install -q --no-deps {wd}/openvino-*.whl {wd}/openvino_telemetry-*.whl
    import openvino as ov
print("openvino", ov.__version__)

SR=32000; N_MELS=256; N_FFT=2048; HOP=512; FMIN=20; FMAX=16000; TOP_DB=80
N_WINDOWS=12; WIN=SR*5
def sig(x): return 1/(1+np.exp(-np.clip(x,-50,50)))
"""

C2 = """# paths
def ff(c,m):
    for p in c:
        p=Path(p)
        if p.exists() and (list(p.rglob(m)) or (p/m).exists()): return p
    return None
COMP = ff(["/kaggle/input/competitions/birdclef-2026","/kaggle/input/birdclef-2026"],"taxonomy.csv")
SC = COMP/"train_soundscapes"; LAB = COMP/"train_soundscapes_labels.csv"; SS=COMP/"sample_submission.csv"
TUCK = ff(["/kaggle/input/notebooks/maekeso/birdclef2026-tucker-sed-ov","/kaggle/input/birdclef2026-tucker-sed-ov"],"sed_fold0.xml")
E106 = ff(["/kaggle/input/notebooks/maekeso/birdclef2026-e106-3fold-ov","/kaggle/input/birdclef2026-e106-3fold-ov"],"exp106_fold0.xml")
print("COMP",COMP,"\\nTUCK",TUCK,"\\nE106",E106)

ss=pd.read_csv(SS); LABELS=ss.columns[1:].tolist(); L2I={l:i for i,l in enumerate(LABELS)}; NC=len(LABELS)
tax=pd.read_csv(COMP/"taxonomy.csv"); l2t=dict(zip(tax["primary_label"].astype(str),tax["class_name"].astype(str)))
"""

C3 = """# build ground truth + load labeled SC windows
ldf=pd.read_csv(LAB)
def t2s(v):
    s=str(v).strip()
    if ":" in s:
        p=[float(x) for x in s.split(":")]; return p[0]*3600+p[1]*60+p[2] if len(p)==3 else p[0]*60+p[1]
    return float(s)
ec="end" if "end" in ldf.columns else "end_time"
seg={}
for _,r in ldf.iterrows():
    st=Path(str(r["filename"])).stem; es=int(round(t2s(r[ec])))
    sp=[x.strip() for x in str(r["primary_label"]).replace(",",";").split(";") if x.strip() and x.strip()!="nan"]
    seg.setdefault((st,es),set()).update(sp)
sc_map={p.stem:p for p in SC.glob("*.ogg")}
files=sorted(set(k[0] for k in seg)&set(sc_map))
print("labeled files:",len(files))

def load(p):
    w,sr=sf.read(str(p),dtype="float32",always_2d=False)
    if w.ndim>1: w=w.mean(1)
    if sr!=SR: w=librosa.resample(w,orig_sr=sr,target_sr=SR)
    t=N_WINDOWS*WIN; w=np.concatenate([w,np.zeros(t-len(w),dtype=np.float32)]) if len(w)<t else w[:t]
    return w.reshape(N_WINDOWS,WIN)

chunks=[]; Y=[]
for st in files:
    cs=load(sc_map[st])
    for wi in range(N_WINDOWS):
        es=(wi+1)*5; y=np.zeros(NC,dtype=np.float32)
        for s in seg.get((st,es),set()):
            if s in L2I: y[L2I[s]]=1.0
        chunks.append(cs[wi]); Y.append(y)
Y=np.stack(Y); print("windows",len(chunks),"pos",int(Y.sum()))
"""

C4 = """# mels
mt=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,f_min=FMIN,f_max=FMAX,power=2.0)
dt=torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)
def mel_t(chs):
    w=torch.from_numpy(np.stack(chs).astype(np.float32)); m=dt(mt(w))
    mu=m.mean((1,2),keepdim=True); sd=m.std((1,2),keepdim=True)+1e-6
    return ((m-mu)/sd).unsqueeze(1).numpy().astype(np.float32)
def mel_l(chs):
    out=[]
    for x in chs:
        s=librosa.feature.melspectrogram(y=x,sr=SR,n_fft=N_FFT,hop_length=HOP,n_mels=N_MELS,fmin=FMIN,fmax=FMAX,power=2.0)
        s=librosa.power_to_db(s,top_db=TOP_DB); s=(s-s.mean())/(s.std()+1e-6); out.append(s)
    return np.stack(out)[:,None].astype(np.float32)
MT=mel_t(chunks); ML=mel_l(chunks); print("mels",MT.shape)
"""

C5 = """# run streams
core=ov.Core()
def run(xml, mels):
    c=core.compile_model(str(xml),"CPU"); out=[]
    for b in range(0,len(mels),24):
        o=c(mels[b:b+24]); clip=o[c.outputs[0]]; fr=o[c.outputs[1]].max(1)
        out.append((0.5*sig(clip)+0.5*sig(fr)).astype(np.float32))
    del c; return np.concatenate(out)

# Tucker 5-fold avg (librosa mel)
tuck=[run(p,ML) for p in sorted(TUCK.glob("sed_fold*.xml"))]
P_tuck=np.mean(tuck,axis=0)
# e106 3-fold avg (torch mel)
e106=[run(E106/f"exp106_fold{f}.xml",MT) for f in [0,1,2]]
P_e106=np.mean(e106,axis=0)
print("P_tuck",P_tuck.shape,"P_e106",P_e106.shape)
P_blend=0.5*P_tuck+0.5*P_e106
"""

C6 = """# === per-class weakness report (ground-truth-light) ===
rows=[]
for ci,lbl in enumerate(LABELS):
    pos=int(Y[:,ci].sum())
    pmax=float(P_blend[:,ci].max()); pmean=float(P_blend[:,ci].mean())
    # stream disagreement (rank corr between tucker & e106 for this class)
    if np.ptp(P_tuck[:,ci])>1e-9 and np.ptp(P_e106[:,ci])>1e-9:
        rc,_=spearmanr(P_tuck[:,ci],P_e106[:,ci]); rc=float(rc) if not np.isnan(rc) else np.nan
    else: rc=np.nan
    # AUC (inflated but relative)
    auc=np.nan
    if 0<pos<len(Y):
        try: auc=float(roc_auc_score(Y[:,ci],P_blend[:,ci]))
        except: pass
    rows.append({"label":lbl,"taxon":l2t.get(lbl,"?"),"pos":pos,"pmax":pmax,"pmean":pmean,"stream_corr":rc,"auc":auc})
df=pd.DataFrame(rows)

print("\\n=== COLLAPSED species (pmax < 0.3) = model barely fires ===")
col=df[df["pmax"]<0.3].sort_values("pmax")
print(f"  count: {len(col)}")
for _,r in col.head(30).iterrows():
    print(f"  {r['label']:14s} {r['taxon']:10s} pos={int(r['pos']):3d} pmax={r['pmax']:.3f} corr={r['stream_corr']:.2f}")

print("\\n=== STREAM DISAGREEMENT (corr < 0.3, pos>0) = low confidence ===")
dis=df[(df["stream_corr"]<0.3)&(df["pos"]>0)].sort_values("stream_corr")
for _,r in dis.head(20).iterrows():
    print(f"  {r['label']:14s} {r['taxon']:10s} pos={int(r['pos']):3d} pmax={r['pmax']:.3f} corr={r['stream_corr']:.2f} auc={r['auc']:.3f}")

print("\\n=== per-taxon (evaluable, pos>0) ===")
ev=df[df["pos"]>0]
print(ev.groupby("taxon").agg(n=("label","count"),auc=("auc","mean"),pmax=("pmax","mean"),corr=("stream_corr","mean")).round(3))

print("\\n=== WORST 25 by AUC (pos>0) ===")
for _,r in ev[ev["auc"].notna()].sort_values("auc").head(25).iterrows():
    print(f"  {r['label']:14s} {r['taxon']:10s} pos={int(r['pos']):3d} auc={r['auc']:.3f} pmax={r['pmax']:.3f} corr={r['stream_corr']:.2f}")

df.to_csv("/kaggle/working/exp124_perclass.csv",index=False)
print("\\nsaved exp124_perclass.csv")
"""

CELLS=[(C1,"imp"),(C2,"path"),(C3,"gt"),(C4,"mel"),(C5,"run"),(C6,"report")]
nb={"cells":[cell(s,c) for s,c in CELLS],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},
    "nbformat":4,"nbformat_minor":5}
NB.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding="utf-8")
print(f"Wrote {NB}")
for s,c in CELLS:
    if c=="imp": continue
    clean="\n".join("# "+ln if ln.lstrip().startswith(("!","%")) else ln for ln in s.splitlines())
    try: compile(clean,f"<{c}>","exec"); print(f"  [OK] {c}")
    except SyntaxError as e: print(f"  [FAIL] {c}: {e}")
