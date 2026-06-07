"""Build + push exp118 = exp110 base + blend weight 0.25/0.40/0.35 (R3 boost).

Only blend weights change vs exp110 (LB 0.952):
  NB4: 0.30 -> 0.25 (further down the in-NB-CV-contaminated stream)
  Tucker: 0.40 (kept)
  R3 (exp029 R3 fold0): 0.30 -> 0.35 (boost NFNet test-domain stream)
R3 stream = exp106 fold0 single (= exp029 R3 fold0), SAME as exp110.
All else identical (Tucker ONNX 5-fold, exp029 R3 PyTorch, PP). ~80 min like exp110.
"""
import io, json, os, sys, tempfile, shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC_NB = Path("experiment/exp110/notebook/nb_exp110_blend_30_40_30.ipynb")
OUT_NB = Path("experiment/exp118/notebook/nb_exp118_blend_25_40_35.ipynb")
OUT_NB.parent.mkdir(parents=True, exist_ok=True)

nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

# Patch blend weights
done = False
for c in nb["cells"]:
    if c.get("id") != "blend":
        continue
    s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    s = s.replace(
        "BLEND_W_E10  = 0.30   # ★ exp110: NB4 weight 0.35 → 0.30 (in-NB CV contaminated stream を下げる)",
        "BLEND_W_E10  = 0.25   # ★ exp118: NB4 weight 0.30 → 0.25 (further down)")
    s = s.replace(
        "BLEND_W_E17  = 0.30   # ★ exp110: exp029 R3 weight 0.25 → 0.30 (NFNet test 域 generalization 強化)",
        "BLEND_W_E17  = 0.35   # ★ exp118: exp029 R3 weight 0.30 → 0.35 (R3 boost、exp110 方向延長)")
    c["source"] = s.splitlines(keepends=True)
    done = True
assert done, "blend cell not found"

# Patch header (cell 0)
for c in nb["cells"]:
    if c.get("id") == "hdr" or c["cell_type"] == "markdown":
        c["source"] = (
            "# exp118 — exp110 base + blend weight 0.25/0.40/0.35 (R3 boost)\n\n"
            "★ exp110 (LB 0.952) の weight tune 延長: NB4 0.30→0.25, R3 0.30→0.35\n\n"
            "## Stack (exp110 と同 ckpt、weight のみ変更)\n"
            "| stream | exp110 | exp118 |\n|---|---|---|\n"
            "| exp037 (NB4) | 0.30 | **0.25** |\n"
            "| Tucker SED 5-fold ONNX | 0.40 | 0.40 |\n"
            "| exp029 R3 fold0 (eca_nfnet_l1) | 0.30 | **0.35** |\n\n"
            "## 狙い\n"
            "- exp090→exp110 で NB4↓+R3↑ が +0.001。同方向を1歩進める\n"
            "- R3 slot = exp029 R3 fold0 single (exp116/117 の新 fold 混入は drag 確認済→使わない)\n"
            "- 期待 LB 0.952-0.953、slot 飽和なので +0.001 lottery + private hedge\n\n"
            "## time: ~80 min (exp110 と同構成、Tucker ONNX + R3 PyTorch)\n"
        ).splitlines(keepends=True)
        break

OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT_NB} ({OUT_NB.stat().st_size} bytes)")

# verify
nbc = json.loads(OUT_NB.read_text(encoding="utf-8"))
blob = "\n".join("".join(c.get("source", [])) for c in nbc["cells"])
assert "BLEND_W_E10  = 0.25" in blob
assert "BLEND_W_SED  = 0.40" in blob
assert "BLEND_W_E17  = 0.35" in blob
assert "exp090 —" not in blob and "exp078" not in blob
print("  [OK] weights 0.25/0.40/0.35, header clean")
# weight sum check
assert abs(0.25 + 0.40 + 0.35 - 1.0) < 1e-9
print("  [OK] weight sum = 1.0")

# === push ===
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)
if not os.environ.get("KAGGLE_API_TOKEN"):
    _kgat = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    os.environ["KAGGLE_API_TOKEN"] = _kgat

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

USER = "maekeso"
SLUG = "birdclef2026-exp118-blend-25-40-35"
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(OUT_NB, td / OUT_NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": "birdclef2026 exp118 blend 25 40 35",
        "code_file": OUT_NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": False,
        "competition_sources": ["birdclef-2026"],
        "dataset_sources": [
            "jaejohn/perch-meta",
            "rishikeshjani/perch-onnx-for-birdclef-2026",
            "tuckerarrants/bc2026-distilled-sed-public",
            f"{USER}/birdclef2026-exp029-l1-single",
        ],
        "kernel_sources": ["ashok205/tf-wheels"],
        "model_sources": ["google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1"],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", r.url, "Version:", r.version_number)
