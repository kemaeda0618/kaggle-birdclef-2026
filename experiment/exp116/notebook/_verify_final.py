"""Strict final verification of exp116/exp117 before tomorrow's sub.

Checks:
1. cell structure intact (same as exp090/exp112 base, only intended cells changed)
2. e29-infer: FOLDS_E106 correct, exp106 IR (FP32, NOT int8), OV outputs[0]=clip/[1]=frame
3. sed-infer: Tucker 3-fold ([:3]), librosa mel, probs_sed output
4. blend: weight 0.30/0.40/0.30, uses probs_exp010/probs_sed/probs_e17
5. variable namespace consistency (probs_e17 -> blend)
6. no stale tokens (exp102/exp111/e102/int8/5-fold full)
7. mel params (256/2048/512), per-instance standardize
8. all code cells compile
9. push metadata (separate check)
"""
import io, json, sys, re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

EXP090 = json.loads(Path("experiment/exp090/notebook/nb_exp090_tuned_tax.ipynb").read_text(encoding="utf-8"))

TARGETS = {
    "exp116": ("experiment/exp116/notebook/nb_exp116_fold01.ipynb", "[0, 1]"),
    "exp117": ("experiment/exp117/notebook/nb_exp117_fold02.ipynb", "[0, 2]"),
}


def cells(nb):
    return {c.get("id"): ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
            for c in nb["cells"]}


fc090 = cells(EXP090)

for exp_name, (path, folds) in TARGETS.items():
    print(f"\n{'='*60}\n{exp_name}\n{'='*60}")
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    fc = cells(nb)
    fails = 0

    def chk(name, cond):
        nonlocal_fail = not cond
        print(f"  {'[OK]' if cond else '[FAIL]'} {name}")
        return 0 if cond else 1

    # 1. cell count + ids vs exp090
    fails += chk("cell count == exp090", len(nb["cells"]) == len(EXP090["cells"]))
    fails += chk("cell ids == exp090", [c.get("id") for c in nb["cells"]] == [c.get("id") for c in EXP090["cells"]])

    # 2. which cells differ from exp090 (expect: hdr, sed-infer, e29-infer, blend)
    diffs = set()
    for cid in fc:
        if cid in fc090 and fc[cid] != fc090[cid]:
            diffs.add(cid)
        elif cid not in fc090:
            diffs.add(cid)
    expected = {"hdr", "sed-infer", "e29-infer", "blend"}
    fails += chk(f"modified cells == {expected}", diffs == expected, )
    if diffs != expected:
        print(f"      actual diffs: {diffs}")

    # 3. e29-infer checks
    e29 = fc["e29-infer"]
    fails += chk(f"FOLDS_E106 = {folds}", f"FOLDS_E106 = {folds}" in e29)
    fails += chk("uses exp106 IR (FP32, not int8)", "exp106_fold" in e29 and "int8" not in e29.lower().replace("abandoned -0.018","").replace("int8 abandoned",""))
    fails += chk("e106 OV dir = birdclef2026-e106-3fold-ov", "birdclef2026-e106-3fold-ov" in e29)
    fails += chk("outputs[0]=clip, outputs[1]=frame", "_compiled.outputs[0]" in e29 and "_compiled.outputs[1]" in e29)
    fails += chk("mel n_fft=2048/hop=512/n_mels=256", "n_fft=2048" in e29 and "hop_length=512" in e29 and "n_mels=256" in e29)
    fails += chk("per-instance standardize", "(_mel[_i] - _mel[_i].mean()) / (_mel[_i].std() + 1e-6)" in e29 or "mel_mean" in e29)
    fails += chk("0.5 sigmoid(clip)+0.5 sigmoid(frame_max)", "0.5 * _p_clip + 0.5 * _p_frame" in e29)
    fails += chk("gaussian smooth sigma=0.65", 'sigma=0.65' in e29)
    fails += chk("ensemble avg / n_folds", "_clip_sum / _n_folds" in e29 and "_frame_sum / _n_folds" in e29)
    fails += chk("output probs_e17", "probs_e17 = np.stack(probs_e17)" in e29)
    fails += chk("reads audio_cache + N_WINDOWS", "for _fi, _raw_60s in enumerate(audio_cache)" in e29 and "N_WINDOWS, WINDOW_SAMPLES" in e29)

    # 4. sed-infer checks
    sed = fc["sed-infer"]
    fails += chk("Tucker 3-fold ([:3])", "sed_xml_paths[:3]" in sed)
    fails += chk("Tucker OV (ov.Core)", "ov.Core" in sed and "openvino" in sed)
    fails += chk("librosa mel power_to_db", "librosa.power_to_db" in sed)
    fails += chk("sed mel params", "N_FFT_SED  = 2048" in sed and "HOP_SED    = 512" in sed and "N_MELS_SED = 256" in sed)
    fails += chk("output probs_sed", "probs_sed = np.stack(probs_sed)" in sed)
    fails += chk("ensemble sum/len(sed_compiled)", "p_sum_batch / len(sed_compiled)" in sed or "p_sum / len(sed_compiled)" in sed)

    # 5. blend checks
    blend = fc["blend"]
    fails += chk("BLEND_W_E10 = 0.30", "BLEND_W_E10  = 0.30" in blend)
    fails += chk("BLEND_W_SED = 0.40", "BLEND_W_SED  = 0.40" in blend)
    fails += chk("BLEND_W_E17 = 0.30", "BLEND_W_E17  = 0.30" in blend)
    fails += chk("blend uses probs_e17/probs_sed/probs_exp010", all(t in blend for t in ["probs_e17", "probs_sed", "probs_exp010"]))

    # 6. no stale tokens (anywhere)
    blob = "\n".join(fc.values())
    stale = [t for t in ["exp102", "exp111", "e102-ov", "5-fold full", "exp090 —", "exp078"] if t in blob]
    fails += chk(f"no stale tokens", len(stale) == 0)
    if stale:
        print(f"      stale: {stale}")

    # 7. compile all code cells
    cerr = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln for ln in src.splitlines())
        try:
            compile(clean, f"<{c.get('id')}>", "exec")
        except SyntaxError as e:
            cerr += 1
            print(f"      [COMPILE FAIL] {c.get('id')}: {e}")
    fails += chk("all code cells compile", cerr == 0)

    print(f"\n  >>> {exp_name}: {'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")

# push metadata check
print(f"\n{'='*60}\npush metadata\n{'='*60}")
push = Path("experiment/exp116/notebook/_push_exp116_117.py").read_text(encoding="utf-8")
for item in ['"enable_internet": False', "birdclef2026-e106-3fold-ov", "birdclef2026-tucker-sed-ov",
             "ttahara/birdclef-2026-download-wheels", "ashok205/tf-wheels",
             "rishikeshjani/perch-onnx-for-birdclef-2026", "jaejohn/perch-meta",
             "perch_v2_cpu/1"]:
    print(f"  {'[OK]' if item in push else '[FAIL]'} push has: {item}")
print(f"  [INFO] e106 IR source: {'birdclef2026-e106-3fold-ov (FP32)' if 'birdclef2026-e106-3fold-ov' in push else '??'}")
print(f"  [INFO] int8 IR in push? {'YES (WRONG!)' if 'int8' in push.lower() else 'no (correct, FP32)'}")
