"""OV5c: LIGHTWEIGHT INT8 AUC check — e106 ONLY (3 folds), robust against OV5b hang.

OV5b hung in the AUC cell (8 models + 720 windows -> memory pressure / OV thread leak).
OV5c fixes:
  - e106 ONLY (3 folds) — Tucker stays FP32 in exp112 (light enough), so only e106 INT8 matters
  - eval reduced 720 -> 360 windows (30 files)
  - calib 200 -> 96
  - per-fold: compute + print AUC delta IMMEDIATELY with flush=True (partial results survive hang)
  - explicit del + gc after each model; del compiled models right after use
  - ensemble AUC at end (fast)
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path("experiment/exp112/notebook/nb_ov5c.ipynb")


def cell(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


CELL1 = """!pip install -q openvino nncf
import openvino as ov, nncf
print(f"openvino={ov.__version__}, nncf={nncf.__version__}", flush=True)
"""

CELL2 = """import os, sys, time, glob, re, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torchaudio, soundfile as sf, librosa
from sklearn.metrics import roc_auc_score

SR=32_000; N_MELS=256; N_FFT=2048; HOP=512; FMIN=20; FMAX=16000; TOP_DB=80
N_WINDOWS=12; WINDOW_SAMPLES=SR*5

def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-50,50)))

def find_first(cands, marker):
    for p in cands:
        p=Path(p)
        if p.exists() and (list(p.rglob(marker)) or (p/marker).exists()):
            return p
    return None

COMP=find_first(["/kaggle/input/competitions/birdclef-2026","/kaggle/input/birdclef-2026"],"taxonomy.csv")
assert COMP, "comp not found"
SC_DIR=COMP/"train_soundscapes"; LABELS=COMP/"train_soundscapes_labels.csv"; SS=COMP/"sample_submission.csv"

E106_DIR=find_first(["/kaggle/input/notebooks/maekeso/birdclef2026-e106-3fold-ov",
                     "/kaggle/input/birdclef2026-e106-3fold-ov"],"exp106_fold0.xml")
if E106_DIR is None:
    E106_DIR=sorted(Path("/kaggle/input").rglob("exp106_fold0.xml"))[0].parent
print(f"COMP={COMP}\\nE106_DIR={E106_DIR}", flush=True)
"""

CELL3 = """# === ground truth Y from labels ===
sample_sub=pd.read_csv(SS); PRIMARY=sample_sub.columns[1:].tolist()
L2I={l:i for i,l in enumerate(PRIMARY)}; N_CLASSES=len(PRIMARY)
ldf=pd.read_csv(LABELS)

def t2s(v):
    s=str(v).strip()
    if ":" in s:
        p=[float(x) for x in s.split(":")]
        return p[0]*3600+p[1]*60+p[2] if len(p)==3 else p[0]*60+p[1]
    return float(s)

end_col="end" if "end" in ldf.columns else None
seg={}
for _,r in ldf.iterrows():
    stem=Path(str(r["filename"])).stem
    es=int(round(t2s(r[end_col])))
    sp=[x.strip() for x in str(r["primary_label"]).replace(",",";").split(";") if x.strip() and x.strip()!="nan"]
    seg.setdefault((stem,es),set()).update(sp)

sc_ogg={p.stem:p for p in SC_DIR.glob("*.ogg")}
labeled=sorted(set(k[0] for k in seg)&set(sc_ogg))[:30]   # ★ 30 files = 360 windows
print(f"eval files: {len(labeled)}", flush=True)

def load_ch(path,n=N_WINDOWS):
    w,sr=sf.read(str(path),dtype="float32",always_2d=False)
    if w.ndim>1: w=w.mean(1)
    if sr!=SR: w=librosa.resample(w,orig_sr=sr,target_sr=SR)
    tg=n*WINDOW_SAMPLES
    w=np.concatenate([w,np.zeros(tg-len(w),dtype=np.float32)]) if len(w)<tg else w[:tg]
    return w.reshape(n,WINDOW_SAMPLES)

eval_ch=[]; eval_Y=[]
for stem in labeled:
    chs=load_ch(sc_ogg[stem])
    for wi in range(N_WINDOWS):
        es=(wi+1)*5; species=seg.get((stem,es),set())
        y=np.zeros(N_CLASSES,dtype=np.float32)
        for sp in species:
            if sp in L2I: y[L2I[sp]]=1.0
        eval_ch.append(chs[wi]); eval_Y.append(y)
eval_Y=np.stack(eval_Y)
print(f"eval windows={len(eval_ch)} pos={int(eval_Y.sum())} cls+={int((eval_Y.sum(0)>0).sum())}", flush=True)

# calib: other files
calib_files=[p for p in sorted(SC_DIR.glob("*.ogg")) if p.stem not in set(labeled)][:8]
calib_ch=[]
for p in calib_files: calib_ch.extend(load_ch(p))
calib_ch=calib_ch[:96]
print(f"calib windows={len(calib_ch)}", flush=True)
"""

CELL4 = """# === torch mel (e106) ===
_mtf=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=N_FFT,hop_length=HOP,
        n_mels=N_MELS,f_min=FMIN,f_max=FMAX,power=2.0)
_dtf=torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)
def mel_t(chunks):
    w=torch.from_numpy(np.stack(chunks).astype(np.float32))
    m=_dtf(_mtf(w))
    mu=m.mean((1,2),keepdim=True); sd=m.std((1,2),keepdim=True)+1e-6
    return ((m-mu)/sd).unsqueeze(1).numpy().astype(np.float32)
t0=time.time()
eval_mel=mel_t(eval_ch); calib_mel=mel_t(calib_ch)
print(f"mels: eval={eval_mel.shape} ({time.time()-t0:.0f}s)", flush=True)
"""

CELL5 = """# === per-fold: quantize + AUC delta (immediate flush) ===
core=ov.Core(); OUT=Path("/kaggle/working")

def run(comp,mels):
    out=[]
    for b in range(0,len(mels),24):
        o=comp(mels[b:b+24])
        clip=o[comp.outputs[0]]; fr=o[comp.outputs[1]].max(1)
        out.append((0.5*sigmoid(clip)+0.5*sigmoid(fr)).astype(np.float32))
    return np.concatenate(out)

def macro_auc(Y,P):
    a=[]
    for c in range(Y.shape[1]):
        p=Y[:,c].sum()
        if 0<p<len(Y):
            try: a.append(roc_auc_score(Y[:,c],P[:,c]))
            except: pass
    return (float(np.mean(a)) if a else float("nan")), len(a)

fp32_preds={}; int8_preds={}
print(f"\\n{'fold':>6} {'FP32_AUC':>10} {'INT8_AUC':>10} {'delta':>9} {'ncls':>5} {'s':>5}", flush=True)
for f in [0,1,2]:
    t0=time.time()
    xml=E106_DIR/f"exp106_fold{f}.xml"
    m=core.read_model(str(xml))
    cf=core.compile_model(m,"CPU")
    pf=run(cf,eval_mel); fp32_preds[f]=pf
    del cf; gc.collect()
    q=nncf.quantize(m, nncf.Dataset(list(calib_mel), lambda s:s[None]),
                    subset_size=min(96,len(calib_mel)), preset=nncf.QuantizationPreset.PERFORMANCE)
    ov.save_model(q, str(OUT/f"exp106_fold{f}_int8.xml"))
    cq=core.compile_model(q,"CPU")
    pq=run(cq,eval_mel); int8_preds[f]=pq
    del cq,m,q; gc.collect()
    a_fp,n=macro_auc(eval_Y,pf); a_i8,_=macro_auc(eval_Y,pq)
    print(f"{f:>6} {a_fp:>10.5f} {a_i8:>10.5f} {a_i8-a_fp:>+9.5f} {n:>5} {time.time()-t0:>5.0f}", flush=True)

# ensemble
e_fp=np.mean([fp32_preds[f] for f in [0,1,2]],axis=0)
e_i8=np.mean([int8_preds[f] for f in [0,1,2]],axis=0)
a_fp,n=macro_auc(eval_Y,e_fp); a_i8,_=macro_auc(eval_Y,e_i8)
print(f"\\n{'='*50}", flush=True)
print(f"e106 3-fold ENSEMBLE: FP32={a_fp:.5f} INT8={a_i8:.5f} delta={a_i8-a_fp:+.5f} (ncls={n})", flush=True)
d=a_i8-a_fp
v="SAFE (>-0.002)" if d>-0.002 else ("ACCEPTABLE (-0.005..-0.002)" if d>-0.005 else "RISKY (<-0.005)")
print(f">>> INT8 verdict (e106): {v}", flush=True)
print(f"    NOTE: eval contaminated (trained on labeled SS) -> DELTA matters, not abs AUC", flush=True)
"""

CELLS=[(CELL1,"install"),(CELL2,"setup"),(CELL3,"data"),(CELL4,"mel"),(CELL5,"quant")]
nb={"cells":[cell(s,c) for s,c in CELLS],
    "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python"}},
    "nbformat":4,"nbformat_minor":5}
NB_PATH.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding="utf-8")
print(f"Wrote {NB_PATH}")
for s,c in CELLS:
    if c=="install": continue
    clean="\n".join("# "+ln if ln.lstrip().startswith(("!","%")) else ln for ln in s.splitlines())
    try: compile(clean,f"<{c}>","exec"); print(f"  [OK] {c}")
    except SyntaxError as e: print(f"  [FAIL] {c}: {e}")
