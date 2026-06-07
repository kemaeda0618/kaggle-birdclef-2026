"""Poll exp134 v2 submit NB local verification run; pull log to confirm assert + timing."""
import json, os, io, sys, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"; os.execv(sys.executable, [sys.executable] + sys.argv)
os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle"/"kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
REF = "maekeso/birdclef2026-exp134-amphib-v2-surgical"
while True:
    try: st = json.loads(api.kernels_status(REF))
    except Exception as e: print("err",e); time.sleep(60); continue
    s = st.get("status"); print(time.strftime("%H:%M:%S"), s, flush=True)
    if s not in ("RUNNING","QUEUED"): print("FINAL:",json.dumps(st)); break
    time.sleep(60)
out = Path("experiment/exp134/notebook/_out"); out.mkdir(exist_ok=True)
try: api.kernels_output(REF,str(out)); print("dl:",[p.name for p in sorted(out.glob('*'))])
except Exception as e: print("err",e)
