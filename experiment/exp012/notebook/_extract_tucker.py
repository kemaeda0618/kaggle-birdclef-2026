"""Extract Tucker SED notebook source into a usable .ipynb file.

Reads the JSON dump from kaggle_get_notebook_info MCP tool and writes a clean
.ipynb that can be opened locally.
"""
import json, sys, io
from pathlib import Path

if __name__ == "__main__":
    src = Path(__file__).with_name("_reference_tucker_raw.json")
    dst_ipynb = Path(__file__).with_name("_reference_tucker.ipynb")
    dst_py    = Path(__file__).with_name("_reference_tucker.py")

    raw = json.loads(src.read_text(encoding="utf-8"))
    inner = json.loads(raw[0]["text"])  # {"metadata": ..., "blob": ...}
    src_text = inner["blob"]["source"]
    nb = json.loads(src_text)
    print("Cells:", len(nb["cells"]))
    print("Languages:", nb["metadata"].get("language_info", {}).get("name"))
    print("Kaggle data sources:")
    for d in nb["metadata"].get("kaggle", {}).get("dataSources", []):
        print(f"  {d.get('sourceType')}: id={d.get('sourceId')} db={d.get('databundleVersionId')}")

    dst_ipynb.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {dst_ipynb} ({dst_ipynb.stat().st_size} bytes)")

    # Also flatten code cells into a single .py for grep/read
    blocks = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
            blocks.append(f"# ===== cell {i} (code) =====\n{src}\n")
        elif c["cell_type"] == "markdown":
            src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
            blocks.append(f"# ===== cell {i} (md) =====\n# " + src.replace("\n", "\n# ") + "\n")
    dst_py.write_text("\n".join(blocks), encoding="utf-8")
    print(f"Wrote {dst_py} ({dst_py.stat().st_size} bytes)")
