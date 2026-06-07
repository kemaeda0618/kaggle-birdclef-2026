"""Build + push exp119-122: weight grid around exp110 (0.30/0.40/0.30).

R3 slot = exp029 R3 fold0 single (= exp110 ckpt), FIXED. Only blend weights vary.
Base = exp110 NB (Tucker ONNX 5-fold + exp029 R3 PyTorch, ~80 min, proven fit).

Grid:
  exp119: 0.30 / 0.35 / 0.35  (Tucker down, R3 up)
  exp120: 0.25 / 0.45 / 0.30  (Tucker up — recheck with R3=fold0)
  exp121: 0.20 / 0.40 / 0.40  (R3 = NB4 level, extreme)
  exp122: 0.35 / 0.35 / 0.30  (NB4 up, Tucker down — reverse direction)
"""
import io, json, os, sys, tempfile, shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)
if not os.environ.get("KAGGLE_API_TOKEN"):
    _kgat = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["key"]
    os.environ["KAGGLE_API_TOKEN"] = _kgat

SRC_NB = Path("experiment/exp110/notebook/nb_exp110_blend_30_40_30.ipynb")
src_nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

# (exp, nb4, sed, r3, slug)
GRID = [
    ("exp119", 0.30, 0.35, 0.35),
    ("exp120", 0.25, 0.45, 0.30),
    ("exp121", 0.20, 0.40, 0.40),
    ("exp122", 0.35, 0.35, 0.30),
]

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
USER = "maekeso"

# original exp110 weight lines (anchors to replace)
OLD_E10 = "BLEND_W_E10  = 0.30   # ★ exp110: NB4 weight 0.35 → 0.30 (in-NB CV contaminated stream を下げる)"
OLD_SED = "BLEND_W_SED  = 0.40   # Tucker public SED weight (維持、exp021 で 0.45 → -0.001 確認済)"
OLD_E17 = "BLEND_W_E17  = 0.30   # ★ exp110: exp029 R3 weight 0.25 → 0.30 (NFNet test 域 generalization 強化)"

for exp, w10, wsed, w17 in GRID:
    assert abs(w10 + wsed + w17 - 1.0) < 1e-9, f"{exp} weight sum != 1"
    nb = json.loads(json.dumps(src_nb))  # deep copy

    # patch blend
    done = False
    for c in nb["cells"]:
        if c.get("id") != "blend":
            continue
        s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        s = s.replace(OLD_E10, f"BLEND_W_E10  = {w10:.2f}   # {exp}: NB4 weight")
        s = s.replace(OLD_SED, f"BLEND_W_SED  = {wsed:.2f}   # {exp}: Tucker SED weight")
        s = s.replace(OLD_E17, f"BLEND_W_E17  = {w17:.2f}   # {exp}: exp029 R3 fold0 weight")
        c["source"] = s.splitlines(keepends=True)
        done = True
    assert done, f"{exp}: blend cell not found"

    # patch header
    for c in nb["cells"]:
        if c["cell_type"] == "markdown":
            c["source"] = (
                f"# {exp} — exp110 base, blend weight {w10:.2f}/{wsed:.2f}/{w17:.2f}\n\n"
                f"weight grid around exp110 (0.30/0.40/0.30, LB 0.952). R3 = exp029 R3 fold0 single (fixed).\n\n"
                f"| stream | weight |\n|---|---|\n"
                f"| exp037 (NB4) | {w10:.2f} |\n| Tucker SED 5-fold | {wsed:.2f} |\n| exp029 R3 fold0 | {w17:.2f} |\n\n"
                f"Tucker ONNX + R3 PyTorch, ~80 min (exp110 と同). Expected 0.951-0.953.\n"
            ).splitlines(keepends=True)
            break

    out_nb = Path(f"experiment/{exp}/notebook/nb_{exp}_weight.ipynb")
    out_nb.parent.mkdir(parents=True, exist_ok=True)
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    # verify
    blob = "\n".join("".join(c.get("source", [])) for c in json.loads(out_nb.read_text(encoding="utf-8"))["cells"])
    assert f"BLEND_W_E10  = {w10:.2f}" in blob and f"BLEND_W_SED  = {wsed:.2f}" in blob and f"BLEND_W_E17  = {w17:.2f}" in blob
    print(f"{exp}: {w10:.2f}/{wsed:.2f}/{w17:.2f} [OK build]")

    # push
    slug = f"birdclef2026-{exp}-w{int(w10*100)}-{int(wsed*100)}-{int(w17*100)}"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy(out_nb, td / out_nb.name)
        meta = {
            "id": f"{USER}/{slug}",
            "title": f"birdclef2026 {exp} w{int(w10*100)} {int(wsed*100)} {int(w17*100)}",
            "code_file": out_nb.name,
            "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_internet": False,
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
        print(f"  pushed: {r.url} V{r.version_number}")
