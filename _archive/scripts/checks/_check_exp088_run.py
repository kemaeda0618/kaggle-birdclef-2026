"""Fetch exp088 kernel output (script 322652434) and audit run."""
import json, os, tempfile, re
from pathlib import Path

if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp088-4model-window-babych"
with tempfile.TemporaryDirectory() as td:
    api.kernels_output(slug, path=td, quiet=True)
    log = next(Path(td).rglob("*.log"))
    content = log.read_text(encoding="utf-8", errors="replace")
    if '"stream_name"' in content:
        try:
            entries = json.loads(content)
            for e in entries:
                if isinstance(e, dict) and 'data' in e:
                    s = e['data'].rstrip('\n')
                    if s.strip():
                        print(s)
        except Exception:
            for m in re.finditer(r'"data":"([^"]*(?:\\.[^"]*)*)"', content):
                s = m.group(1).encode().decode('unicode_escape').rstrip('\n')
                if s.strip():
                    print(s)
    else:
        print(content)
