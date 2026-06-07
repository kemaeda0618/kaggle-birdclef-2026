import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

for name, p in [("exp116","experiment/exp116/notebook/nb_exp116_fold01.ipynb"),
                ("exp117","experiment/exp117/notebook/nb_exp117_fold02.ipynb")]:
    nb = json.loads(Path(p).read_text(encoding="utf-8"))
    print(f"===== {name} =====")
    for c in nb["cells"]:
        cid = c.get("id")
        s = "".join(c.get("source", []))
        for ln_no, ln in enumerate(s.split("\n"), 1):
            if "exp111" in ln or "exp102" in ln:
                print(f"  [{cid}:{ln_no}] {ln.strip()[:140]}")
