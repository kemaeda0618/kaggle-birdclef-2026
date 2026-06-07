import json, os, io, sys, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle"/"kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
REF = "maekeso/birdclef2026-exp141-amphib-v5-surgical"
while True:
    try: st = json.loads(api.kernels_status(REF))
    except Exception as e: print("err",e); time.sleep(60); continue
    s = st.get("status"); print(time.strftime("%H:%M:%S"), s, flush=True)
    if s not in ("RUNNING","QUEUED"): print("FINAL:",json.dumps(st)); break
    time.sleep(60)
out = Path("experiment/exp141/notebook/_out"); out.mkdir(exist_ok=True)
try:
    api.kernels_output(REF,str(out)); print("dl:",[p.name for p in sorted(out.glob('*'))])
    d=json.load(open(out/"birdclef2026-exp141-amphib-v5-surgical.log",encoding="utf-8"))
    [print(e.get("data","").rstrip()) for e in d if any(k in e.get("data","") for k in ["exp141","Submission","assert","Error","Traceback","min)"])]
except Exception as e: print("err",e)
