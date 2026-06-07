"""Read exp085 kernel log fully, extract all stdout content."""
import json, os, tempfile, re
from pathlib import Path

if not os.environ.get("KAGGLE_API_TOKEN"):
    os.environ["KAGGLE_API_TOKEN"] = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

slug = "maekeso/birdclef2026-exp085-4model-blend"
with tempfile.TemporaryDirectory() as td:
    api.kernels_output(slug, path=td, quiet=True)
    log = next(Path(td).rglob("*.log"))
    content = log.read_text(encoding="utf-8", errors="replace")
    # Strip JSON entry format if present, extract data field
    if content.startswith("[") or '"stream_name"' in content:
        # JSON-array format: extract data fields
        try:
            entries = json.loads(content)
            for e in entries:
                if isinstance(e, dict) and 'data' in e:
                    s = e['data'].rstrip('\n')
                    if s.strip():
                        print(s)
        except Exception:
            # Fallback: regex extraction
            for m in re.finditer(r'"data":"([^"]*(?:\\.[^"]*)*)"', content):
                s = m.group(1).encode().decode('unicode_escape').rstrip('\n')
                if s.strip():
                    print(s)
    else:
        print(content)
