"""Rewrite cell 0 (markdown header) of exp116/exp117 to match actual config."""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def header(exp_name, folds_label, fold_list):
    return f"""# {exp_name} — exp110 base + exp106 {folds_label} 2-fold (FP32, anchor restored)

★ **exp110 (LB 0.952) の R3 slot を exp106 {folds_label} 2-fold ensemble に置換**

## 背景
- exp110 (LB 0.952) = R3 slot に **exp029 R3 fold0** (= exp106 fold0) single
- exp112 (LB 0.951) = exp106 **fold1+2** に置換 → **anchor fold0 を落として -0.001**
- → {exp_name} = **fold0 (anchor) を復活** + 追加 fold で 2-fold ensemble

## Stack
| # | Model | Weight | Format |
|---|---|---|---|
| 1 | exp037 (LightProtoSSM + MLP + ResSSM + Perch) | 0.30 | FP32 |
| 2 | Tucker SED **3-fold** OV | 0.40 | FP32 (3-fold: ~82min budget margin) |
| 3 | **exp106 {folds_label} 2-fold** (eca_nfnet_l1, exp029 R3 recipe) | 0.30 | FP32 OV |

- exp106 fold{fold_list} = exp029 R3 recipe (mixup 0.4, drop_path 0.1)
- **fold0 = exp110 の anchor** (val 0.9351, standalone 0.923), fold{fold_list[1]} = 新 train fold

## INT8 不採用 (重要)
- INT8 量子化は exp113 で **-0.018 劣化** (real test) 確認 → **FP32 を使用**
- OV5c の SAFE 判定は contaminated/飽和 eval (AUC 0.999) の artifact だった

## Expected
- LB **0.951-0.953** (anchor 復活で exp112 0.951 から回復 + α)
- gold (0.957) は slot 飽和で届かない、shake-up 頼み
- 真の価値: **fold ensemble = private robustness (shake-down 防御)**

## Time budget
- 推定 sub time **~82 min** (Tucker 3-fold + exp106 2-fold FP32、timeout なし)

## Inputs
- `birdclef-2026` (competition)
- `jaejohn/perch-meta`, `rishikeshjani/perch-onnx-for-birdclef-2026`
- kernel_sources: `ashok205/tf-wheels`, `maekeso/birdclef2026-tucker-sed-ov`,
  `maekeso/birdclef2026-e106-3fold-ov` (FP32 e106 IR), `ttahara/birdclef-2026-download-wheels`
- model_sources: `google/bird-vocalization-classifier/.../perch_v2_cpu/1`
- enable_internet: **False**
"""


TARGETS = {
    "exp116": ("experiment/exp116/notebook/nb_exp116_fold01.ipynb", "fold0+1", [0, 1]),
    "exp117": ("experiment/exp117/notebook/nb_exp117_fold02.ipynb", "fold0+2", [0, 2]),
}

for exp_name, (path, folds_label, fold_list) in TARGETS.items():
    p = Path(path)
    nb = json.loads(p.read_text(encoding="utf-8"))
    # cell 0 should be the markdown hdr
    c0 = nb["cells"][0]
    assert c0["cell_type"] == "markdown", f"{exp_name}: cell0 not markdown"
    new = header(exp_name, folds_label, fold_list)
    c0["source"] = new.splitlines(keepends=True)
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    # verify
    blob0 = "".join(json.loads(p.read_text(encoding="utf-8"))["cells"][0]["source"])
    assert "exp090" not in blob0 and "exp078" not in blob0, f"{exp_name}: stale exp090/078 in header"
    assert exp_name in blob0 and folds_label in blob0
    print(f"{exp_name}: header rewritten, no exp090/078 [OK]")
