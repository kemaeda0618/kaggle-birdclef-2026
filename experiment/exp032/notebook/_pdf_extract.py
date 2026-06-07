"""Download + extract text from a list of PDFs.

Usage: python _pdf_extract.py URL [URL ...]
"""
import sys
import urllib.request
import fitz  # pymupdf
from pathlib import Path
import tempfile

URLS = sys.argv[1:] if len(sys.argv) > 1 else [
    "https://ceur-ws.org/Vol-4038/paper_254.pdf",
    "https://ceur-ws.org/Vol-4038/paper_232.pdf",
    "https://arxiv.org/pdf/2507.08236",
    "https://ceur-ws.org/Vol-4038/paper_256.pdf",
]

for url in URLS:
    print(f"\n{'='*78}\nFetching: {url}\n{'='*78}")
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                tf.write(resp.read())
            path = tf.name
        doc = fitz.open(path)
        print(f"pages={len(doc)}")
        for pi in range(len(doc)):
            page = doc[pi]
            text = page.get_text()
            print(f"\n--- page {pi+1} ---")
            print(text)
        doc.close()
        Path(path).unlink(missing_ok=True)
    except Exception as e:
        print(f"FAIL: {e}")
