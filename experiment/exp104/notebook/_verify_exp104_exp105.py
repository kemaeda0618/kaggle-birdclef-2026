"""Strict verify of exp104/exp105 NBs.

Checks:
1. cell count == exp090 (no accidental cells added/removed)
2. sed-infer cell content == new OV version (literal match)
3. e29-infer cell content == new OV version per fold spec
4. All other cells == exp090 (byte-equal)
5. Both cells compile cleanly
6. probs_e17 + probs_sed variable names preserved (blend cell compatibility)
7. Critical constants present in new cells (mel params match exp090)
8. push metadata: enable_internet=False, kernel_sources includes both OV datasets
"""
import io, json, re, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

EXP090 = json.loads(Path("experiment/exp090/notebook/nb_exp090_tuned_tax.ipynb").read_text(encoding="utf-8"))
EXP104 = json.loads(Path("experiment/exp104/notebook/nb_exp104_e29_swap_3fold.ipynb").read_text(encoding="utf-8"))
EXP105 = json.loads(Path("experiment/exp105/notebook/nb_exp105_e29_swap_fold1.ipynb").read_text(encoding="utf-8"))


def cell_src(nb, cid):
    for c in nb["cells"]:
        if c.get("id") == cid:
            s = c["source"]
            return "".join(s) if isinstance(s, list) else s
    return None


def fmt_ok(name, ok, detail=""):
    tag = "[OK]" if ok else "[FAIL]"
    print(f"  {tag} {name}{(': ' + detail) if detail else ''}")
    return ok


def verify_nb(label, nb, folds_expected):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    fails = 0

    # 1. cell count
    n090 = len(EXP090["cells"]); n_target = len(nb["cells"])
    if not fmt_ok("cell count == exp090", n090 == n_target,
                   f"{n_target} vs {n090}"): fails += 1

    # 2. cell ids match (order preserved)
    ids090 = [c.get("id") for c in EXP090["cells"]]
    ids_t = [c.get("id") for c in nb["cells"]]
    if not fmt_ok("cell id order == exp090", ids090 == ids_t): fails += 1

    # 3. only sed-infer + e29-infer modified, all others byte-equal
    diffs = []
    for c090, ct in zip(EXP090["cells"], nb["cells"]):
        s090 = c090.get("source", "")
        st = ct.get("source", "")
        s090 = "".join(s090) if isinstance(s090, list) else s090
        st = "".join(st) if isinstance(st, list) else st
        if s090 != st:
            diffs.append(c090.get("id"))
    expected_diffs = {"sed-infer", "e29-infer"}
    if not fmt_ok("modified cells == {sed-infer, e29-infer}",
                   set(diffs) == expected_diffs,
                   f"diffs={diffs}"): fails += 1

    # 4. sed-infer: new content must use OV (not onnxruntime)
    sed_src = cell_src(nb, "sed-infer")
    if not fmt_ok("sed-infer mentions 'openvino' or 'ov.Core'",
                   "openvino" in sed_src and "ov.Core" in sed_src): fails += 1
    if not fmt_ok("sed-infer NOT using ort.InferenceSession",
                   "ort.InferenceSession" not in sed_src and "ort.SessionOptions" not in sed_src): fails += 1
    if not fmt_ok("sed-infer loads sed_fold*.xml (not .onnx)",
                   "sed_fold*.xml" in sed_src): fails += 1
    if not fmt_ok("sed-infer outputs probs_sed",
                   "probs_sed = np.stack(probs_sed)" in sed_src): fails += 1
    if not fmt_ok("sed-infer mel params (n_fft=2048, hop=512, n_mels=256, top_db=80)",
                   all(k in sed_src for k in ["N_FFT_SED  = 2048", "HOP_SED    = 512",
                                                "N_MELS_SED = 256", "TOP_DB_SED = 80"])): fails += 1
    if not fmt_ok("sed-infer uses librosa.power_to_db (matches exp090)",
                   "librosa.power_to_db" in sed_src): fails += 1
    if not fmt_ok("sed-infer per-instance standardize (matches exp090)",
                   "(s - s.mean()) / (s.std() + 1e-6)" in sed_src): fails += 1
    if not fmt_ok("sed-infer 5-fold expected",
                   "expected 5 folds" in sed_src): fails += 1
    if not fmt_ok("sed-infer ensemble = sum / len(sed_compiled)",
                   "p_sum / len(sed_compiled)" in sed_src): fails += 1
    if not fmt_ok("sed-infer gaussian_filter1d sigma=0.65",
                   "gaussian_filter1d(p_mean, sigma=0.65, axis=0, mode=\"nearest\")" in sed_src): fails += 1

    # 5. e29-infer: must use exp102 OV with correct folds
    e29_src = cell_src(nb, "e29-infer")
    expected_folds_repr = str(folds_expected)
    if not fmt_ok("e29-infer FOLDS_E102 matches",
                   f"FOLDS_E102 = {expected_folds_repr}" in e29_src,
                   f"expected FOLDS_E102 = {expected_folds_repr}"): fails += 1
    if not fmt_ok("e29-infer locates birdclef2026-exp102-3fold-ov",
                   "birdclef2026-exp102-3fold-ov" in e29_src): fails += 1
    if not fmt_ok("e29-infer outputs probs_e17 (blend compat)",
                   "probs_e17 = np.stack(probs_e17)" in e29_src): fails += 1
    if not fmt_ok("e29-infer mel: n_fft=2048, hop=512, n_mels=256 (exp029 match)",
                   "n_fft=2048" in e29_src and "hop_length=512" in e29_src and "n_mels=256" in e29_src): fails += 1
    if not fmt_ok("e29-infer per-instance standardize",
                   "(_mel[_i] - _mel[_i].mean()) / (_mel[_i].std() + 1e-6)" in e29_src): fails += 1
    if not fmt_ok("e29-infer 0.5*sigmoid(clip) + 0.5*sigmoid(frame_max)",
                   "0.5 * _p_clip + 0.5 * _p_frame" in e29_src): fails += 1
    if not fmt_ok("e29-infer gaussian sigma=0.65",
                   "gaussian_filter1d(_p_mean, sigma=0.65, axis=0, mode=\"nearest\")" in e29_src): fails += 1
    if not fmt_ok("e29-infer ensemble avg = sum / n_folds",
                   "_clip_sum / _n_folds" in e29_src and "_frame_sum / _n_folds" in e29_src): fails += 1
    if not fmt_ok("e29-infer outputs[0]=clip, outputs[1]=frame",
                   "_compiled.outputs[0]" in e29_src and "_compiled.outputs[1]" in e29_src): fails += 1
    if not fmt_ok("e29-infer reads audio_cache + N_WINDOWS",
                   "for _fi, _raw_60s in enumerate(audio_cache)" in e29_src
                   and "N_WINDOWS, WINDOW_SAMPLES" in e29_src): fails += 1

    # 6. compile both cells (with !pip line guard)
    for cid, src in [("sed-infer", sed_src), ("e29-infer", e29_src)]:
        clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln
                           for ln in src.splitlines())
        try:
            compile(clean, f"<{cid}>", "exec")
            fmt_ok(f"compile {cid}", True)
        except SyntaxError as e:
            fmt_ok(f"compile {cid}", False, f"{e}")
            fails += 1

    # 7. blend cell unchanged (uses probs_e17 + probs_sed)
    blend_src = cell_src(nb, "blend")
    blend090 = cell_src(EXP090, "blend")
    if not fmt_ok("blend cell byte-equal to exp090", blend_src == blend090): fails += 1
    if not fmt_ok("blend uses probs_e17 + probs_sed",
                   "probs_e17" in blend_src and "probs_sed" in blend_src): fails += 1

    # 8. bridge cell unchanged (audio_cache builder)
    bridge_src = cell_src(nb, "bridge")
    bridge090 = cell_src(EXP090, "bridge")
    if not fmt_ok("bridge cell byte-equal to exp090", bridge_src == bridge090): fails += 1

    print(f"\n  TOTAL: {'PASS' if fails == 0 else f'{fails} FAIL'}")
    return fails


def verify_push(label, push_path, expected_slug, expected_internet=False,
                 expected_ks_must_have=None):
    print(f"\n--- push: {push_path.name} ---")
    src = push_path.read_text(encoding="utf-8")
    fails = 0
    if not fmt_ok(f"SLUG={expected_slug}",
                   f'SLUG = "{expected_slug}"' in src): fails += 1
    if not fmt_ok(f"enable_internet={expected_internet}",
                   f'"enable_internet": {expected_internet}' in src): fails += 1
    for ks in (expected_ks_must_have or []):
        if not fmt_ok(f"kernel_sources has {ks}", f'"{ks}"' in src): fails += 1
    # title length check
    title_match = re.search(r'"title":\s*"([^"]+)"', src)
    if title_match:
        t = title_match.group(1)
        if not fmt_ok(f"title length {len(t)} ≤ 50",
                       len(t) <= 50, f'"{t}"'): fails += 1
    return fails


# Run all checks
total_fails = 0
total_fails += verify_nb("exp104 (3-fold ensemble)", EXP104, [0, 1, 2])
total_fails += verify_nb("exp105 (fold 1 single)", EXP105, [1])

print(f"\n{'='*70}\npush script verify\n{'='*70}")
total_fails += verify_push("exp104",
                            Path("experiment/exp104/notebook/_push_exp104.py"),
                            expected_slug="birdclef2026-exp104-e102-3fold",
                            expected_ks_must_have=[
                                "maekeso/birdclef2026-tucker-sed-ov",
                                "maekeso/birdclef2026-exp102-3fold-ov",
                                "ttahara/birdclef-2026-download-wheels",
                                "ashok205/tf-wheels",
                            ])
total_fails += verify_push("exp105",
                            Path("experiment/exp105/notebook/_push_exp105.py"),
                            expected_slug="birdclef2026-exp105-e102-fold1",
                            expected_ks_must_have=[
                                "maekeso/birdclef2026-tucker-sed-ov",
                                "maekeso/birdclef2026-exp102-3fold-ov",
                                "ttahara/birdclef-2026-download-wheels",
                                "ashok205/tf-wheels",
                            ])

print(f"\n{'='*70}")
print(f"GRAND TOTAL: {'ALL PASS ✓' if total_fails == 0 else f'{total_fails} FAILURES'}")
print(f"{'='*70}")
