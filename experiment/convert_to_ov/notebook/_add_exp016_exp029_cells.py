"""Add exp016 R2 + exp029 R3 OV conversion cells to convert_to_ov.ipynb.

Order after edit:
  0. Header (updated to 7 models)
  1. Setup
  2. Model class
  3. Helpers
  4. exp015 R2 convert
  5. exp017 R2 convert
  6. exp016 R2 convert  ★ NEW
  7. exp081 R2 convert
  8. exp082 R2 convert
  9. exp029 R3 convert  ★ NEW
  10. Summary (updated)
"""
import json
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).parent / "convert_to_ov.ipynb"
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

# ============================================================
# Cell content: exp016 R2 conversion
# ============================================================
exp016_cell_src = '''# ============================================================
# Cell: Convert exp016 R2 (regnety_008, 5s Tucker)
# ============================================================
print('='*60)
print('Converting exp016 R2 (regnety_008, 5s Tucker)')
print('='*60)

dataset_slug = 'birdclef2026-exp016-weights'
ckpt_candidates = ['r2_ckpt_best_ns22.pth']

DATASET_ROOTS = [
    Path('/kaggle/input/datasets/maekeso') / dataset_slug,
    Path('/kaggle/input') / dataset_slug,
]
dataset_root = None
for p in DATASET_ROOTS:
    if p.exists():
        dataset_root = p
        print(f'  dataset_root: {p}')
        break
assert dataset_root is not None, f'Dataset not mounted: tried {DATASET_ROOTS}'

ckpt_path = None
for cand in ckpt_candidates:
    p = dataset_root / cand
    if p.exists():
        ckpt_path = p; break

if ckpt_path is None:
    target_name = Path(ckpt_candidates[0]).name
    print(f'  Direct paths not found, rglob fallback for {target_name}...')
    for f in dataset_root.rglob(target_name):
        ckpt_path = f
        print(f'    Found via rglob: {f}')
        break

if ckpt_path is None:
    print(f'  [DEBUG] {dataset_root} contents:')
    for f in sorted(dataset_root.rglob('*'))[:30]:
        if f.is_file():
            print(f'    {f.relative_to(dataset_root)} ({f.stat().st_size/1e6:.1f}MB)')

assert ckpt_path is not None, f'ckpt not found in {dataset_slug}'

out_dir = convert_pytorch_to_ov(
    ckpt_path=ckpt_path,
    backbone_name='regnety_008',
    n_mels=256,
    time_frames=313,
    model_name='exp016_r2',
    drop_path_rate=0.0,
    hidden_dim=512,
    compress_fp16=False,
)

ok = upload_ov_to_dataset(
    dataset_slug=dataset_slug,
    ov_local_dir=out_dir,
    ov_subfolder='r2_ov',
    version_notes='Added OpenVINO IR (r2_ov/)',
)
print(f'\\n  Result: {"OK" if ok else "FAIL"}')
'''

# ============================================================
# Cell content: exp029 R3 conversion
# ============================================================
exp029_cell_src = '''# ============================================================
# Cell: Convert exp029 R3 (eca_nfnet_l1, 5s Tucker, fold 0)
# ============================================================
print('='*60)
print('Converting exp029 R3 (eca_nfnet_l1, 5s Tucker, fold 0)')
print('='*60)

dataset_slug = 'birdclef2026-exp029-l1-single'
ckpt_candidates = ['r3_fold0_ckpt_best_ns22.pth']

DATASET_ROOTS = [
    Path('/kaggle/input/datasets/maekeso') / dataset_slug,
    Path('/kaggle/input') / dataset_slug,
]
dataset_root = None
for p in DATASET_ROOTS:
    if p.exists():
        dataset_root = p
        print(f'  dataset_root: {p}')
        break
assert dataset_root is not None, f'Dataset not mounted: tried {DATASET_ROOTS}'

ckpt_path = None
for cand in ckpt_candidates:
    p = dataset_root / cand
    if p.exists():
        ckpt_path = p; break

if ckpt_path is None:
    target_name = Path(ckpt_candidates[0]).name
    print(f'  Direct paths not found, rglob fallback for {target_name}...')
    for f in dataset_root.rglob(target_name):
        ckpt_path = f
        print(f'    Found via rglob: {f}')
        break

if ckpt_path is None:
    print(f'  [DEBUG] {dataset_root} contents:')
    for f in sorted(dataset_root.rglob('*'))[:30]:
        if f.is_file():
            print(f'    {f.relative_to(dataset_root)} ({f.stat().st_size/1e6:.1f}MB)')

assert ckpt_path is not None, f'ckpt not found in {dataset_slug}'

# ★ exp029 R3: backbone eca_nfnet_l1 (l0 vs l1 capacity 異)
out_dir = convert_pytorch_to_ov(
    ckpt_path=ckpt_path,
    backbone_name='eca_nfnet_l1',
    n_mels=256,
    time_frames=313,
    model_name='exp029_r3',
    drop_path_rate=0.0,
    hidden_dim=512,
    compress_fp16=False,
)

# ★ exp029 dataset は r3_ov/ subfolder にする (r2 でなく r3)
ok = upload_ov_to_dataset(
    dataset_slug=dataset_slug,
    ov_local_dir=out_dir,
    ov_subfolder='r3_ov',
    version_notes='Added OpenVINO IR (r3_ov/)',
)
print(f'\\n  Result: {"OK" if ok else "FAIL"}')
'''

# ============================================================
# Updated header
# ============================================================
new_header = '''# Convert PyTorch ckpts → OpenVINO IR (7 models batch)

## Models
1. **exp015 R2** (5s convnext_pico, Tucker mel)
2. **exp017 R2** (5s eca_nfnet_l0, Tucker mel)
3. **exp016 R2** (5s regnety_008, Tucker mel) ★ NEW
4. **exp081 R2** (10s eca_nfnet_l0, Tucker mel)
5. **exp082 R2** (20s eca_nfnet_l0, Babych mel)
6. **exp029 R3** (5s eca_nfnet_l1, Tucker mel, fold 0) ★ NEW

## Output (per model, into existing Dataset's subfolder)
- `r2_ov/model.xml` + `r2_ov/model.bin` for R2 models
- **`r3_ov/model.xml` + `r3_ov/model.bin`** for exp029 R3 (note r3, not r2)

## Constraints
- Internet=True (openvino install + Kaggle upload)
- CPU or GPU mode (CPU OK for conversion)
- Conversion time: ~3-5 min/model, total ~25-40 min

## Inference NB usage (post-conversion)
```python
import openvino as ov
core = ov.Core()
model = core.read_model('/kaggle/input/datasets/maekeso/birdclef2026-exp029-l1-single/r3_ov/model.xml')
compiled = core.compile_model(model, 'CPU')
result = compiled.create_infer_request().infer({input_port: mel_np})
```
'''

# ============================================================
# Updated summary
# ============================================================
new_summary = '''# ============================================================
# Cell: Summary
# ============================================================
print('\\n' + '='*60)
print('Conversion summary')
print('='*60)
print('\\nConverted OV files:')
for sub in sorted(ONNX_OUT_ROOT.iterdir()):
    if sub.is_dir():
        xml = sub / 'model.xml'
        bin_f = sub / 'model.bin'
        if xml.exists() and bin_f.exists():
            print(f'  {sub.name}/model.xml ({xml.stat().st_size/1024:.1f}KB)')
            print(f'  {sub.name}/model.bin ({bin_f.stat().st_size/1e6:.1f}MB)')

print('\\nUploaded to:')
for slug, subfolder in [
    ('birdclef2026-exp015-weights', 'r2_ov'),
    ('birdclef2026-exp017-weights', 'r2_ov'),
    ('birdclef2026-exp016-weights', 'r2_ov'),
    ('birdclef2026-exp081-weights', 'r2_ov'),
    ('birdclef2026-exp082-weights', 'r2_ov'),
    ('birdclef2026-exp029-l1-single', 'r3_ov'),
]:
    print(f'  {subfolder}/ → https://www.kaggle.com/datasets/maekeso/{slug}')
'''


def make_code_cell(src_str):
    """Create a code cell dict matching existing structure."""
    # Match structure of existing cells (no 'id' field used in this NB)
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src_str.splitlines(keepends=True),
    }


def make_markdown_cell(src_str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src_str.splitlines(keepends=True),
    }


# ============================================================
# Locate cells by content prefix
# ============================================================
def find_cell_idx(cells, prefix):
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if src.lstrip().startswith(prefix):
            return i
    return -1

# Identify cell positions
idx_header = 0  # always first
idx_exp015 = find_cell_idx(nb["cells"], "# ============================================================\n# Cell: Convert exp015")
idx_exp017 = find_cell_idx(nb["cells"], "# ============================================================\n# Cell: Convert exp017")
idx_exp081 = find_cell_idx(nb["cells"], "# ============================================================\n# Cell: Convert exp081")
idx_exp082 = find_cell_idx(nb["cells"], "# ============================================================\n# Cell: Convert exp082")
idx_summary = find_cell_idx(nb["cells"], "# ============================================================\n# Cell: Summary")

print(f"Cell positions: header=0, exp015={idx_exp015}, exp017={idx_exp017}, "
      f"exp081={idx_exp081}, exp082={idx_exp082}, summary={idx_summary}")

assert idx_exp015 > 0 and idx_exp017 > 0 and idx_exp081 > 0 and idx_exp082 > 0 and idx_summary > 0

# ============================================================
# Apply edits
# ============================================================
cells = nb["cells"]

# Update header (idx 0)
cells[0] = make_markdown_cell(new_header)

# Update summary
cells[idx_summary] = make_code_cell(new_summary)

# Insert exp029 BEFORE summary (so order: ... exp082, exp029, summary)
cells.insert(idx_summary, make_code_cell(exp029_cell_src))

# Insert exp016 AFTER exp017 (before exp081)
# After inserting exp029, idx_exp081 may have shifted — but we insert before idx_exp081 from original
# Recompute idx_exp081 in new list
idx_exp081_new = find_cell_idx(cells, "# ============================================================\n# Cell: Convert exp081")
assert idx_exp081_new > 0
cells.insert(idx_exp081_new, make_code_cell(exp016_cell_src))

# ============================================================
# Verify final order
# ============================================================
print("\nFinal cell order:")
for i, c in enumerate(cells):
    src = "".join(c.get("source", []))
    header_line = src.split("\n")[0][:60] if c["cell_type"] == "code" else src.split("\n")[0][:60]
    print(f"  {i:2d} [{c['cell_type'][:4]}] {header_line}")

# Save
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nSaved: {NB_PATH}")
print(f"Cell count: {len(cells)}")
