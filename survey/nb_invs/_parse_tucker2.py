"""Tucker NB content is in 'blob' (notebook JSON encoded). Decode."""
import json, base64
from pathlib import Path

p = Path(r"C:\Users\maeke\.claude\projects\C--Users-maeke-work-kaggle-birdclef-2026\10b3c073-565f-4aa3-897d-438b8fd4776a\tool-results\mcp-kaggle-get_notebook_info-1779582937669.txt")
data = json.loads(p.read_text(encoding="utf-8"))
blob = data["blob"]
print(f"blob type: {type(blob)}, keys (if dict): {list(blob.keys()) if isinstance(blob, dict) else 'N/A'}")
print(f"blob length: {len(blob) if isinstance(blob, str) else 'dict'}")
if isinstance(blob, dict):
    for k, v in blob.items():
        print(f"  {k}: {type(v)} - {str(v)[:100]}")
elif isinstance(blob, str):
    print(f"blob preview: {blob[:500]}")
