"""Add 2 cells to Stage 1 NB:
1. Kaggle Dataset upload (ckpt + history.json)
2. runtime.unassign() at the very end (auto-disconnect)
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp052\notebook\nb_stage1_xc_pretrain.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

def code_cell(cid, src):
    return {"cell_type": "code", "id": cid, "metadata": {},
            "source": src.splitlines(keepends=True), "execution_count": None, "outputs": []}

UPLOAD_CELL = code_cell("upload_kaggle", """# ============================================================
# Cell 15: Upload ckpt as Kaggle Dataset
# ============================================================
import json, shutil
from pathlib import Path

UPLOAD_DIR = Path("/content/exp052_upload")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# Copy ckpts + history
for src_file in [CFG.BEST_CKPT, CFG.LAST_CKPT, CFG.HIST_JSON]:
    if src_file.exists():
        shutil.copy(src_file, UPLOAD_DIR / src_file.name)
        print(f"  Copied: {src_file.name} ({src_file.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  WARN: {src_file.name} not found")

# Dataset metadata
USER = "maekeso"
SLUG = "birdclef2026-exp052-stage1-effv2s"
meta = {
    "title": "BirdCLEF2026 exp052 Stage 1 EfficientNetV2-S",
    "id": f"{USER}/{SLUG}",
    "licenses": [{"name": "other"}],
}
(UPLOAD_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

# Try version up first (if Dataset exists), fall back to create new
import subprocess

def run(cmd):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(f"stderr: {r.stderr[:500]}")
    return r.returncode

# Try version up
ret = run(f"kaggle datasets version -p {UPLOAD_DIR} -m 'Stage 1 epoch {CFG.EPOCHS} best={best_val:.4f}' -r tar")
if ret != 0:
    # Fall back: create new
    print("Version up failed, trying create new...")
    ret = run(f"kaggle datasets create -p {UPLOAD_DIR} -r tar")

if ret == 0:
    print(f"\\nOK Dataset uploaded: https://www.kaggle.com/datasets/{USER}/{SLUG}")
else:
    print(f"\\nERR upload failed. Manual upload:")
    print(f"  files: {[f.name for f in UPLOAD_DIR.iterdir()]}")
""")

DISCONNECT_CELL = code_cell("disconnect", """# ============================================================
# Cell 16: Disconnect runtime (auto-shutdown to save Colab credits)
# ============================================================
print("All done. Disconnecting runtime in 30 sec to save Colab credits...")
print("(if you want to keep runtime, interrupt this cell within 30 sec)")

import time
time.sleep(30)

from google.colab import runtime
runtime.unassign()  # auto-disconnect
""")

# Replace existing summary cell with upload + disconnect at end
# Insert UPLOAD before summary, DISCONNECT after summary
new_cells = []
for c in nb["cells"]:
    if c.get("id") == "summary":
        # Add upload BEFORE summary (so summary's print runs after upload status)
        new_cells.append(UPLOAD_CELL)
        new_cells.append(c)
        # Add disconnect AFTER summary
        new_cells.append(DISCONNECT_CELL)
    else:
        new_cells.append(c)

nb["cells"] = new_cells

# Also update summary cell to remove the disabled runtime.unassign() block
for c in nb["cells"]:
    if c.get("id") == "summary":
        src = "".join(c["source"])
        # Remove the disabled runtime.unassign() section
        old = """print()
# Disconnect runtime to save credits
print("Disconnecting runtime to save Colab credits...")
from google.colab import runtime
# runtime.unassign()  # uncomment to auto-disconnect
"""
        if old in src:
            src = src.replace(old, "")
        c["source"] = src.splitlines(keepends=True)
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB}")
print(f"  cells: {len(nb['cells'])}")
for c in nb["cells"]:
    n = len("".join(c["source"]).splitlines())
    print(f"    {c['cell_type']:8s} id={c.get('id'):15s} L={n}")
