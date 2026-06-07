"""Build Kaggle NB to convert PyTorch ckpts → OpenVINO IR for 5 models.

Models converted:
1. exp015 R2 (5s convnext_pico, Tucker mel)
2. exp017 R2 (5s eca_nfnet_l0, Tucker mel)
3. exp029 R3 (5s eca_nfnet_l0, R3 distill)
4. exp081 R2 (10s eca_nfnet_l0, Tucker mel)
5. exp082 R2 (20s eca_nfnet_l0, Babych mel)

Each conversion uploads to corresponding Dataset's r2_ov/ subfolder
(or r2_ov/ for folder-structured datasets, ov/ for exp029 R3).

★ Internet=True required (openvino install + Kaggle Dataset upload)
"""
import json
from pathlib import Path

OUT_DIR = Path("experiment/convert_to_ov/notebook")
OUT_DIR.mkdir(parents=True, exist_ok=True)

cells = []

# ---- Cell 0: Markdown header ----
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Convert PyTorch ckpts → OpenVINO IR (5 models batch)\n",
        "\n",
        "## Models\n",
        "1. **exp015 R2** (5s convnext_pico, Tucker mel)\n",
        "2. **exp017 R2** (5s eca_nfnet_l0, Tucker mel)\n",
        "3. **exp029 R3** (5s eca_nfnet_l0, R3 distill)\n",
        "4. **exp081 R2** (10s eca_nfnet_l0, Tucker mel) — folder structure\n",
        "5. **exp082 R2** (20s eca_nfnet_l0, Babych mel) — folder structure\n",
        "\n",
        "## Output (per model, into existing Dataset's r2_ov/ folder)\n",
        "- `r2_ov/model.xml` (OpenVINO IR descriptor)\n",
        "- `r2_ov/model.bin` (OpenVINO IR weights, F32)\n",
        "\n",
        "## Constraints\n",
        "- Internet=True (openvino install + Kaggle upload)\n",
        "- CPU or GPU mode (CPU OK for conversion, no training)\n",
        "- Conversion time: ~3-5 min/model, total ~20-30 min\n",
        "\n",
        "## Inference NB の使い方 (変換後)\n",
        "```python\n",
        "import openvino as ov\n",
        "core = ov.Core()\n",
        "model = core.read_model('/kaggle/input/birdclef2026-expXXX-weights/r2_ov/model.xml')\n",
        "compiled = core.compile_model(model, 'CPU')\n",
        "result = compiled.create_infer_request().infer({input_name: mel_np})\n",
        "```\n",
    ]
})

# ---- Cell 1: Setup + install ----
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 1: Setup — install openvino + imports\n",
        "# ============================================================\n",
        "!pip install -q openvino==2024.4.0 onnx\n",
        "\n",
        "import os, json, shutil, tempfile, time\n",
        "from pathlib import Path\n",
        "import numpy as np\n",
        "\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import timm\n",
        "import openvino as ov\n",
        "\n",
        "print(f'torch={torch.__version__}, timm={timm.__version__}, ov={ov.__version__}')\n",
        "\n",
        "# Kaggle API auth (Kaggle env auto-loads /kaggle/secrets if set)\n",
        "from kaggle.api.kaggle_api_extended import KaggleApi\n",
        "api = KaggleApi(); api.authenticate()\n",
        "print('Kaggle authenticated OK')\n",
    ]
})

# ---- Cell 2: Model class ----
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 2: Model class (BirdSEDModel) + Wrapper for ONNX export\n",
        "# ============================================================\n",
        "NUM_CLASSES = 234\n",
        "PERCH_EMBED_DIM = 1536\n",
        "USE_PERCH_DISTILL = True\n",
        "\n",
        "class GeMFreqPool(nn.Module):\n",
        "    def __init__(self, p_init=3.0, eps=1e-6):\n",
        "        super().__init__()\n",
        "        self.p = nn.Parameter(torch.tensor(float(p_init)))\n",
        "        self.eps = eps\n",
        "    def forward(self, x):\n",
        "        p = self.p.clamp(min=1.0)\n",
        "        x = x.clamp(min=self.eps).pow(p)\n",
        "        x = x.mean(dim=2)\n",
        "        return x.pow(1.0 / p)\n",
        "\n",
        "\n",
        "class DistillHead(nn.Module):\n",
        "    def __init__(self, backbone_dim, embed_dim=1536):\n",
        "        super().__init__()\n",
        "        self.proj = nn.Linear(backbone_dim, embed_dim)\n",
        "    def forward(self, feature_map):\n",
        "        return self.proj(feature_map.mean(dim=[2, 3]))\n",
        "\n",
        "\n",
        "class BirdSEDModel(nn.Module):\n",
        "    def __init__(self, backbone_name, num_classes=NUM_CLASSES,\n",
        "                 drop_path_rate=0.0, hidden_dim=512, n_mels=256, time_frames=313):\n",
        "        super().__init__()\n",
        "        self.backbone = timm.create_model(\n",
        "            backbone_name, pretrained=False, in_chans=1,\n",
        "            num_classes=0, global_pool='', drop_path_rate=drop_path_rate,\n",
        "        )\n",
        "        with torch.no_grad():\n",
        "            dummy = torch.randn(1, 1, n_mels, time_frames)\n",
        "            feat = self.backbone(dummy)\n",
        "            self.backbone_dim = feat.shape[1]\n",
        "        self.gem_freq = GeMFreqPool(p_init=3.0)\n",
        "        self.dense = nn.Sequential(\n",
        "            nn.Dropout(0.25),\n",
        "            nn.Linear(self.backbone_dim, hidden_dim),\n",
        "            nn.ReLU(inplace=True),\n",
        "            nn.Dropout(0.5),\n",
        "        )\n",
        "        self.att = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)\n",
        "        self.cla = nn.Conv1d(hidden_dim, num_classes, kernel_size=1, bias=True)\n",
        "        if USE_PERCH_DISTILL:\n",
        "            self.distill_head = DistillHead(self.backbone_dim, PERCH_EMBED_DIM)\n",
        "\n",
        "    def forward(self, x, return_framewise=True):\n",
        "        h = self.backbone(x)\n",
        "        h_cls = self.gem_freq(h)\n",
        "        h_cls = h_cls.permute(0, 2, 1)\n",
        "        h_cls = self.dense(h_cls)\n",
        "        h_cls = h_cls.permute(0, 2, 1)\n",
        "        norm_att = torch.softmax(torch.tanh(self.att(h_cls)), dim=-1)\n",
        "        framewise = self.cla(h_cls)\n",
        "        clip_logits = torch.sum(norm_att * framewise, dim=2)\n",
        "        # Return both for OV (single output is harder to specify)\n",
        "        return clip_logits, framewise.permute(0, 2, 1)\n",
        "\n",
        "\n",
        "class OVWrapper(nn.Module):\n",
        "    '''Thin wrapper: always returns (clip_logits, framewise) for clean OV export.'''\n",
        "    def __init__(self, model):\n",
        "        super().__init__()\n",
        "        self.model = model\n",
        "    def forward(self, mel):\n",
        "        return self.model(mel)\n",
        "\n",
        "print('OK model class ready')\n",
    ]
})

# ---- Cell 3: Conversion + Upload helper ----
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell 3: Helpers — convert + upload\n",
        "# ============================================================\n",
        "ONNX_OUT_ROOT = Path('/kaggle/working/ov_exports')\n",
        "ONNX_OUT_ROOT.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "def convert_pytorch_to_ov(ckpt_path, backbone_name, n_mels, time_frames,\n",
        "                          model_name, drop_path_rate=0.0, hidden_dim=512,\n",
        "                          compress_fp16=False):\n",
        "    '''Convert PyTorch ckpt → ONNX → OpenVINO IR. Returns dir with model.xml + model.bin.'''\n",
        "    print(f'\\n--- {model_name} ---')\n",
        "    print(f'  ckpt: {ckpt_path}')\n",
        "    print(f'  backbone: {backbone_name}, mel ({n_mels}, {time_frames})')\n",
        "\n",
        "    # Load PyTorch state\n",
        "    state = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)\n",
        "    epoch = state.get('epoch', '?')\n",
        "    ns22 = state.get('best_ns22', float('nan'))\n",
        "    macro = state.get('best_macro', float('nan'))\n",
        "    print(f'  ckpt epoch={epoch}, best_ns22={ns22:.4f}, best_macro={macro:.4f}')\n",
        "\n",
        "    # Build model\n",
        "    model = BirdSEDModel(backbone_name=backbone_name, n_mels=n_mels,\n",
        "                         time_frames=time_frames, drop_path_rate=drop_path_rate,\n",
        "                         hidden_dim=hidden_dim).eval()\n",
        "    msg = model.load_state_dict(state['model_state'], strict=False)\n",
        "    print(f'  load: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}')\n",
        "    if len(msg.missing_keys) > 0:\n",
        "        print(f'    missing sample: {list(msg.missing_keys)[:3]}')\n",
        "    if len(msg.unexpected_keys) > 0:\n",
        "        print(f'    unexpected sample: {list(msg.unexpected_keys)[:3]}')\n",
        "    assert len(msg.missing_keys) < 5, f'Too many missing keys: {len(msg.missing_keys)}'\n",
        "\n",
        "    # Wrap for export\n",
        "    wrapper = OVWrapper(model).eval()\n",
        "\n",
        "    # ONNX export\n",
        "    out_dir = ONNX_OUT_ROOT / model_name\n",
        "    out_dir.mkdir(parents=True, exist_ok=True)\n",
        "    onnx_path = out_dir / 'model.onnx'\n",
        "    dummy = torch.randn(1, 1, n_mels, time_frames)\n",
        "    \n",
        "    print(f'  Exporting ONNX...')\n",
        "    t0 = time.time()\n",
        "    torch.onnx.export(\n",
        "        wrapper, dummy, str(onnx_path),\n",
        "        input_names=['mel'],\n",
        "        output_names=['clip_logits', 'framewise'],\n",
        "        dynamic_axes={\n",
        "            'mel': {0: 'batch'},\n",
        "            'clip_logits': {0: 'batch'},\n",
        "            'framewise': {0: 'batch'},\n",
        "        },\n",
        "        opset_version=17,\n",
        "        do_constant_folding=True,\n",
        "    )\n",
        "    onnx_mb = onnx_path.stat().st_size / 1e6\n",
        "    print(f'    ONNX saved ({onnx_mb:.1f} MB, {time.time()-t0:.1f}s)')\n",
        "\n",
        "    # ONNX → OpenVINO IR\n",
        "    print(f'  Converting ONNX → OV IR...')\n",
        "    t0 = time.time()\n",
        "    ov_model = ov.convert_model(str(onnx_path))\n",
        "    ov_xml = out_dir / 'model.xml'\n",
        "    ov.save_model(ov_model, str(ov_xml), compress_to_fp16=compress_fp16)\n",
        "    ov_bin = out_dir / 'model.bin'\n",
        "    print(f'    OV IR saved: {ov_xml.stat().st_size/1024:.1f}KB + {ov_bin.stat().st_size/1e6:.1f}MB ({time.time()-t0:.1f}s)')\n",
        "\n",
        "    # Cleanup ONNX (keep only OV IR)\n",
        "    onnx_path.unlink()\n",
        "    print(f'  Cleanup: ONNX removed, keeping OV IR only')\n",
        "\n",
        "    return out_dir\n",
        "\n",
        "\n",
        "def upload_ov_to_dataset(dataset_slug, ov_local_dir, ov_subfolder, version_notes):\n",
        "    '''Upload OV IR to existing Kaggle Dataset, preserving existing files.\n",
        "    Adds OV files in ov_subfolder (e.g., r2_ov/, ov/).'''\n",
        "    print(f'\\n  Uploading to {dataset_slug} → {ov_subfolder}/')\n",
        "    with tempfile.TemporaryDirectory() as td:\n",
        "        td = Path(td)\n",
        "        # 1. Copy existing dataset files (preserve)\n",
        "        existing = Path(f'/kaggle/input/{dataset_slug}')\n",
        "        if not existing.exists():\n",
        "            print(f'    [WARN] {existing} not attached, will create fresh dataset')\n",
        "            n_preserved = 0\n",
        "        else:\n",
        "            n_preserved = 0\n",
        "            for f in existing.rglob('*'):\n",
        "                if not f.is_file(): continue\n",
        "                rel = f.relative_to(existing)\n",
        "                # Skip __huggingface_repos__.json etc.\n",
        "                if f.name.startswith('__'): continue\n",
        "                dst = td / rel\n",
        "                dst.parent.mkdir(parents=True, exist_ok=True)\n",
        "                shutil.copy2(str(f), str(dst))\n",
        "                n_preserved += 1\n",
        "        print(f'    Preserved {n_preserved} existing files')\n",
        "\n",
        "        # 2. Add OV files in subfolder\n",
        "        ov_dst = td / ov_subfolder\n",
        "        ov_dst.mkdir(parents=True, exist_ok=True)\n",
        "        for f in ov_local_dir.iterdir():\n",
        "            if f.is_file():\n",
        "                shutil.copy2(str(f), str(ov_dst / f.name))\n",
        "                print(f'    Added: {ov_subfolder}/{f.name} ({f.stat().st_size/1e6:.1f}MB)')\n",
        "\n",
        "        # 3. Metadata\n",
        "        meta = {\n",
        "            'title': dataset_slug.replace('-', ' '),\n",
        "            'id': f'maekeso/{dataset_slug}',\n",
        "            'licenses': [{'name': 'CC0-1.0'}],\n",
        "        }\n",
        "        (td / 'dataset-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')\n",
        "\n",
        "        # 4. Upload (new version)\n",
        "        try:\n",
        "            t0 = time.time()\n",
        "            api.dataset_create_version(\n",
        "                folder=str(td), version_notes=version_notes,\n",
        "                dir_mode='zip', quiet=False,\n",
        "            )\n",
        "            print(f'    OK Uploaded ({time.time()-t0:.0f}s)')\n",
        "            return True\n",
        "        except Exception as e:\n",
        "            print(f'    FAILED: {type(e).__name__}: {str(e)[:300]}')\n",
        "            return False\n",
        "\n",
        "print('OK helpers ready')\n",
    ]
})

# ---- Cells 4-8: Per-model conversion + upload ----
models_config = [
    {
        "id": "exp015_r2",
        "name": "exp015 R2 (convnext_pico, 5s Tucker)",
        "dataset_slug": "birdclef2026-exp015-weights",
        "ckpt_paths": ["r2_ckpt_best_ns22.pth"],   # flat
        "backbone": "convnext_pico.d1_in1k",
        "n_mels": 256,
        "time_frames": 313,
        "drop_path_rate": 0.0,
        "hidden_dim": 512,
        "ov_subfolder": "r2_ov",
    },
    {
        "id": "exp017_r2",
        "name": "exp017 R2 (eca_nfnet_l0, 5s Tucker)",
        "dataset_slug": "birdclef2026-exp017-weights",
        "ckpt_paths": ["r2_ckpt_best_ns22.pth"],
        "backbone": "eca_nfnet_l0",
        "n_mels": 256,
        "time_frames": 313,
        "drop_path_rate": 0.0,
        "hidden_dim": 512,
        "ov_subfolder": "r2_ov",
    },
    {
        "id": "exp081_r2",
        "name": "exp081 R2 (eca_nfnet_l0, 10s Tucker)",
        "dataset_slug": "birdclef2026-exp081-weights",
        "ckpt_paths": ["r2/ckpt_best_ns22.pth"],  # folder
        "backbone": "eca_nfnet_l0",
        "n_mels": 256,
        "time_frames": 626,  # 10s × 32000 / 512 + 1
        "drop_path_rate": 0.0,
        "hidden_dim": 512,
        "ov_subfolder": "r2_ov",
    },
    {
        "id": "exp082_r2",
        "name": "exp082 R2 (eca_nfnet_l0, 20s Babych)",
        "dataset_slug": "birdclef2026-exp082-weights",
        "ckpt_paths": ["r2/ckpt_best_ns22.pth"],  # folder
        "backbone": "eca_nfnet_l0",
        "n_mels": 224,
        "time_frames": 512,  # 20s × 32000 / 1252 + 1 ≈ 511
        "drop_path_rate": 0.0,
        "hidden_dim": 512,
        "ov_subfolder": "r2_ov",
    },
]

for cfg in models_config:
    src_lines = [
        f"# ============================================================\n",
        f"# Cell: Convert {cfg['name']}\n",
        f"# ============================================================\n",
        f"print('='*60)\n",
        f"print('Converting {cfg['name']}')\n",
        f"print('='*60)\n",
        f"\n",
        f"dataset_slug = {repr(cfg['dataset_slug'])}\n",
        f"ckpt_candidates = {repr(cfg['ckpt_paths'])}\n",
        f"\n",
        f"# Find ckpt\n",
        f"ckpt_path = None\n",
        f"for cand in ckpt_candidates:\n",
        f"    p = Path('/kaggle/input') / dataset_slug / cand\n",
        f"    if p.exists():\n",
        f"        ckpt_path = p; break\n",
        f"assert ckpt_path is not None, f'ckpt not found in {{dataset_slug}}'\n",
        f"\n",
        f"# Convert\n",
        f"out_dir = convert_pytorch_to_ov(\n",
        f"    ckpt_path=ckpt_path,\n",
        f"    backbone_name={repr(cfg['backbone'])},\n",
        f"    n_mels={cfg['n_mels']},\n",
        f"    time_frames={cfg['time_frames']},\n",
        f"    model_name={repr(cfg['id'])},\n",
        f"    drop_path_rate={cfg['drop_path_rate']},\n",
        f"    hidden_dim={cfg['hidden_dim']},\n",
        f"    compress_fp16=False,\n",
        f")\n",
        f"\n",
        f"# Upload to Dataset (preserve existing + add r2_ov/)\n",
        f"ok = upload_ov_to_dataset(\n",
        f"    dataset_slug=dataset_slug,\n",
        f"    ov_local_dir=out_dir,\n",
        f"    ov_subfolder={repr(cfg['ov_subfolder'])},\n",
        f"    version_notes='Added OpenVINO IR ({cfg['ov_subfolder']}/)',\n",
        f")\n",
        f"print(f'\\n  Result: {{\"OK\" if ok else \"FAIL\"}}')\n",
    ]
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": src_lines,
    })

# ---- Final cell: Summary ----
cells.append({
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "# ============================================================\n",
        "# Cell: Summary\n",
        "# ============================================================\n",
        "print('\\n' + '='*60)\n",
        "print('Conversion summary')\n",
        "print('='*60)\n",
        "print('\\nConverted OV files:')\n",
        "for sub in sorted(ONNX_OUT_ROOT.iterdir()):\n",
        "    if sub.is_dir():\n",
        "        xml = sub / 'model.xml'\n",
        "        bin_f = sub / 'model.bin'\n",
        "        if xml.exists() and bin_f.exists():\n",
        "            print(f'  {sub.name}/model.xml ({xml.stat().st_size/1024:.1f}KB)')\n",
        "            print(f'  {sub.name}/model.bin ({bin_f.stat().st_size/1e6:.1f}MB)')\n",
        "\n",
        "print('\\nUploaded to:')\n",
        "for slug in ['birdclef2026-exp015-weights', 'birdclef2026-exp017-weights',\n",
        "             'birdclef2026-exp081-weights', 'birdclef2026-exp082-weights']:\n",
        "    print(f'  https://www.kaggle.com/datasets/maekeso/{slug}')\n",
    ]
})

# Build NB
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

nb_path = OUT_DIR / "convert_to_ov.ipynb"
nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"OK Built {nb_path} ({len(cells)} cells)")

# Build push script
push_script = '''"""Push convert-to-OV NB to Kaggle."""
import json, os, sys, io, tempfile, shutil
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if os.name == "nt" and not os.environ.get("PYTHONUTF8"):
    os.environ["PYTHONUTF8"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

if not os.environ.get("KAGGLE_API_TOKEN"):
    creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    if creds.get("key", "").startswith("KGAT_"):
        os.environ["KAGGLE_API_TOKEN"] = creds["key"]

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()

NB = Path(__file__).with_name("convert_to_ov.ipynb")
USER = "maekeso"
SLUG = "birdclef2026-convert-to-ov"
TITLE = "birdclef2026 convert to ov"

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    shutil.copy(NB, td / NB.name)
    meta = {
        "id": f"{USER}/{SLUG}",
        "title": TITLE,
        "code_file": NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": True,       # for openvino install + upload
        "competition_sources": [],
        "dataset_sources": [
            f"{USER}/birdclef2026-exp015-weights",
            f"{USER}/birdclef2026-exp017-weights",
            f"{USER}/birdclef2026-exp081-weights",
            f"{USER}/birdclef2026-exp082-weights",
        ],
        "kernel_sources": [],
    }
    (td / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    r = api.kernels_push(str(td))
    print("URL:", getattr(r, "url", None))
    print("Version:", getattr(r, "version_number", None))
'''

push_path = OUT_DIR / "_push_nb_convert_to_ov.py"
push_path.write_text(push_script, encoding="utf-8")
print(f"OK Built {push_path}")

# Syntax check
import ast
nb = json.loads(nb_path.read_text(encoding="utf-8"))
n_ok, n_err = 0, 0
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                      for ln in src.splitlines())
    try:
        ast.parse(clean); n_ok += 1
    except SyntaxError as e:
        n_err += 1
        print(f"  SyntaxError: {e}")
print(f"\nSyntax: {n_ok} OK, {n_err} ERRORS")
