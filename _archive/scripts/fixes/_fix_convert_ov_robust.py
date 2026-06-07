"""Fix OV conversion: try PyTorch direct convert first (skip ONNX),
fallback to ONNX opset 14, then opset 17.

Each model attempts:
1. ov.convert_model(pytorch_model, example_input=dummy) — direct, most robust
2. PyTorch → ONNX (opset 14) → OV
3. PyTorch → ONNX (opset 17) → OV

This handles:
- NFNet WS-Conv → ONNX BN issue
- onnxscript dependency (dynamo=False explicit)
- Different ONNX op-set compat
"""
import json
from pathlib import Path

P = Path("experiment/convert_to_ov/notebook/convert_to_ov.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

# Find helpers cell (cell with convert_pytorch_to_ov function)
for c in nb["cells"]:
    if c.get("cell_type") != "code": continue
    src = "".join(c.get("source", []))
    if "def convert_pytorch_to_ov" in src:
        # Replace entire helper function
        new_src = '''# ============================================================
# Cell 3: Helpers — convert + upload (robust multi-path)
# ============================================================
ONNX_OUT_ROOT = Path('/kaggle/working/ov_exports')
ONNX_OUT_ROOT.mkdir(parents=True, exist_ok=True)


def convert_pytorch_to_ov(ckpt_path, backbone_name, n_mels, time_frames,
                          model_name, drop_path_rate=0.0, hidden_dim=512,
                          compress_fp16=False):
    """Convert PyTorch ckpt → OpenVINO IR with robust multi-path fallback.

    Try order:
    1. ov.convert_model(pytorch_model, example_input=dummy) — direct
    2. PyTorch → ONNX opset 14 → OV
    3. PyTorch → ONNX opset 17 → OV
    """
    print(f'\\n--- {model_name} ---')
    print(f'  ckpt: {ckpt_path}')
    print(f'  backbone: {backbone_name}, mel ({n_mels}, {time_frames})')

    # Load PyTorch state
    state = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    epoch = state.get('epoch', '?')
    ns22 = state.get('best_ns22', float('nan'))
    macro = state.get('best_macro', float('nan'))
    print(f'  ckpt epoch={epoch}, best_ns22={ns22:.4f}, best_macro={macro:.4f}')

    # Build model
    model = BirdSEDModel(backbone_name=backbone_name, n_mels=n_mels,
                         time_frames=time_frames, drop_path_rate=drop_path_rate,
                         hidden_dim=hidden_dim).eval()
    msg = model.load_state_dict(state['model_state'], strict=False)
    print(f'  load: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}')
    if len(msg.missing_keys) > 0:
        print(f'    missing sample: {list(msg.missing_keys)[:3]}')
    if len(msg.unexpected_keys) > 0:
        print(f'    unexpected sample: {list(msg.unexpected_keys)[:3]}')
    assert len(msg.missing_keys) < 5, f'Too many missing keys: {len(msg.missing_keys)}'

    # Wrap for export
    wrapper = OVWrapper(model).eval()

    # Test forward to verify
    dummy = torch.randn(1, 1, n_mels, time_frames)
    with torch.no_grad():
        try:
            test_out = wrapper(dummy)
            print(f'  Test forward OK: clip {test_out[0].shape}, framewise {test_out[1].shape}')
        except Exception as e:
            raise RuntimeError(f'PyTorch forward failed: {type(e).__name__}: {e}')

    # Output dir
    out_dir = ONNX_OUT_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Path 1: ov.convert_model direct from PyTorch (most robust)
    # ============================================================
    print(f'\\n  [Path 1] Trying PyTorch direct → OV...')
    ov_model = None
    try:
        t0 = time.time()
        ov_model = ov.convert_model(wrapper, example_input=dummy)
        print(f'    [OK] Direct convert succeeded ({time.time()-t0:.1f}s)')
    except Exception as e:
        print(f'    [FAIL] {type(e).__name__}: {str(e)[:300]}')

    # ============================================================
    # Path 2: PyTorch → ONNX opset 14 → OV
    # ============================================================
    if ov_model is None:
        print(f'\\n  [Path 2] Trying PyTorch → ONNX opset 14 → OV...')
        onnx_path = out_dir / 'model.onnx'
        try:
            t0 = time.time()
            torch.onnx.export(
                wrapper, dummy, str(onnx_path),
                input_names=['mel'],
                output_names=['clip_logits', 'framewise'],
                dynamic_axes={
                    'mel': {0: 'batch'},
                    'clip_logits': {0: 'batch'},
                    'framewise': {0: 'batch'},
                },
                opset_version=14,
                do_constant_folding=True,
                dynamo=False,
            )
            print(f'    ONNX (opset 14) exported ({time.time()-t0:.1f}s, {onnx_path.stat().st_size/1e6:.1f}MB)')
            t0 = time.time()
            ov_model = ov.convert_model(str(onnx_path))
            print(f'    [OK] OV converted ({time.time()-t0:.1f}s)')
            onnx_path.unlink()
        except Exception as e:
            print(f'    [FAIL] {type(e).__name__}: {str(e)[:300]}')
            if onnx_path.exists():
                onnx_path.unlink()

    # ============================================================
    # Path 3: PyTorch → ONNX opset 17 → OV
    # ============================================================
    if ov_model is None:
        print(f'\\n  [Path 3] Trying PyTorch → ONNX opset 17 → OV...')
        onnx_path = out_dir / 'model.onnx'
        try:
            t0 = time.time()
            torch.onnx.export(
                wrapper, dummy, str(onnx_path),
                input_names=['mel'],
                output_names=['clip_logits', 'framewise'],
                dynamic_axes={
                    'mel': {0: 'batch'},
                    'clip_logits': {0: 'batch'},
                    'framewise': {0: 'batch'},
                },
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
            print(f'    ONNX (opset 17) exported ({time.time()-t0:.1f}s)')
            t0 = time.time()
            ov_model = ov.convert_model(str(onnx_path))
            print(f'    [OK] OV converted ({time.time()-t0:.1f}s)')
            onnx_path.unlink()
        except Exception as e:
            print(f'    [FAIL] {type(e).__name__}: {str(e)[:300]}')
            if onnx_path.exists():
                onnx_path.unlink()

    if ov_model is None:
        raise RuntimeError(f'All 3 conversion paths failed for {model_name}')

    # Save OV IR
    ov_xml = out_dir / 'model.xml'
    ov.save_model(ov_model, str(ov_xml), compress_to_fp16=compress_fp16)
    ov_bin = out_dir / 'model.bin'
    print(f'\\n  OV IR saved:')
    print(f'    {ov_xml.name}: {ov_xml.stat().st_size/1024:.1f}KB')
    print(f'    {ov_bin.name}: {ov_bin.stat().st_size/1e6:.1f}MB')

    return out_dir


def upload_ov_to_dataset(dataset_slug, ov_local_dir, ov_subfolder, version_notes):
    """Upload OV IR to existing Kaggle Dataset, preserving existing files."""
    print(f'\\n  Uploading to {dataset_slug} → {ov_subfolder}/')
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # ★ Kaggle Dataset mount: try datasets/maekeso/<slug> first
        existing = None
        for cand_existing in [
            Path(f'/kaggle/input/datasets/maekeso/{dataset_slug}'),
            Path(f'/kaggle/input/{dataset_slug}'),
        ]:
            if cand_existing.exists():
                existing = cand_existing
                break

        if existing is None:
            print(f'    [WARN] dataset not attached, will create fresh')
            n_preserved = 0
        else:
            n_preserved = 0
            for f in existing.rglob('*'):
                if not f.is_file(): continue
                if f.name.startswith('__'): continue
                rel = f.relative_to(existing)
                dst = td / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(f), str(dst))
                n_preserved += 1
            print(f'    Preserved {n_preserved} existing files')

        # Add OV files in subfolder
        ov_dst = td / ov_subfolder
        ov_dst.mkdir(parents=True, exist_ok=True)
        for f in ov_local_dir.iterdir():
            if f.is_file():
                shutil.copy2(str(f), str(ov_dst / f.name))
                print(f'    Added: {ov_subfolder}/{f.name} ({f.stat().st_size/1e6:.1f}MB)')

        # Metadata
        meta = {
            'title': dataset_slug.replace('-', ' '),
            'id': f'maekeso/{dataset_slug}',
            'licenses': [{'name': 'CC0-1.0'}],
        }
        (td / 'dataset-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

        # Upload
        try:
            t0 = time.time()
            api.dataset_create_version(
                folder=str(td), version_notes=version_notes,
                dir_mode='zip', quiet=False,
            )
            print(f'    OK Uploaded ({time.time()-t0:.0f}s)')
            return True
        except Exception as e:
            print(f'    FAILED: {type(e).__name__}: {str(e)[:300]}')
            return False


print('OK helpers ready (3-path robust converter + correct dataset mount)')
'''
        c["source"] = new_src.splitlines(keepends=True)
        print(f"OK Replaced convert_pytorch_to_ov with robust 3-path version")
        break

P.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

# Verify
nb = json.loads(P.read_text(encoding="utf-8"))
src_check = "".join(nb["cells"][3]["source"]) if len(nb["cells"]) > 3 else ""
has_path1 = "ov.convert_model(wrapper, example_input=dummy)" in src_check
has_path2 = "opset_version=14" in src_check
has_path3 = "opset_version=17" in src_check
print(f"\nPath 1 (direct PyTorch): {has_path1}")
print(f"Path 2 (ONNX opset 14): {has_path2}")
print(f"Path 3 (ONNX opset 17): {has_path3}")

# Syntax check
import ast
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
        print(f"  SyntaxError cell {c.get('id','?')}: {e}")
        err_line = e.lineno or 1
        for i, ln in enumerate(clean.splitlines()[max(0, err_line-2):err_line+2]):
            print(f"    L{max(0, err_line-2)+i+1}: {ln[:150]}")
print(f"\nSyntax: {n_ok} OK, {n_err} ERRORS")
