"""Poll birdclef-2026 competition submissions until the newest one gets a public score."""
import json, os, io, sys, time
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"; os.execv(sys.executable, [sys.executable] + sys.argv)
os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home()/".kaggle"/"kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

COMP = "birdclef-2026"
for _ in range(70):
    try:
        subs = api.competition_submissions(COMP)
    except Exception as e:
        print("err:", str(e)[:120]); time.sleep(120); continue
    if subs:
        s = subs[0]  # newest first
        desc = getattr(s, "description", "") or getattr(s, "fileName", "")
        status = getattr(s, "status", "")
        pub = getattr(s, "publicScore", None)
        print(time.strftime("%H:%M:%S"), "status=", status, "score=", pub, "|", str(desc)[:50], flush=True)
        if pub not in (None, "", "None"):
            print("FINAL public score:", pub)
            # print top few for context
            for s2 in subs[:5]:
                print("  ", getattr(s2, "date", ""), getattr(s2, "publicScore", None), str(getattr(s2, "description", ""))[:40])
            break
    time.sleep(120)
