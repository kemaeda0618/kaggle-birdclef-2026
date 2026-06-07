"""Rewrite exp110 cell 0 header to match actual config (0.30/0.40/0.30, our LB 0.952 best)."""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

P = Path("experiment/exp110/notebook/nb_exp110_blend_30_40_30.ipynb")

HEADER = """# exp110 — exp090 base + blend weight tune (0.30 / 0.40 / 0.30)  ★ LB 0.952 (現 best)

★ **exp090 (LB 0.951) の blend weight を調整 → LB 0.952 (+0.001、我々の現 best)**

## 変更点 (vs exp090)
| stream | exp090 | **exp110** | 理由 |
|---|---|---|---|
| NB4 (exp037) | 0.35 | **0.30** | in-NB CV 汚染 stream を下げる |
| Tucker SED 5-fold | 0.40 | 0.40 (維持) | exp021 で 0.45 → -0.001 確認済 |
| exp029 R3 fold0 | 0.25 | **0.30** | NFNet test 域 generalization 強化 |

## Stack
| # | Model | Weight | Format |
|---|---|---|---|
| 1 | exp037 (LightProtoSSM + MLP + ResSSM + Perch) | 0.30 | FP32 |
| 2 | Tucker SED 5-fold ONNX | 0.40 | ONNX |
| 3 | exp029 R3 fold0 (eca_nfnet_l1, exp029 R3 recipe) | 0.30 | PyTorch (.pth) |

## 結果
- **LB 0.952** (exp090 0.951 から +0.001) = 我々の **現 best、final sub 第一候補 (anchor)**
- silver border (0.951) +0.001

## 位置づけ (2026-06-01 時点)
- public 天井 ~0.952 (exp112 で R3 slot 飽和確認)
- INT8 は dead (-0.018、exp113)、3-fold は FP32 timeout
- **exp110 = 確実 silver anchor**、robust 3-stream ensemble
- final 2: exp110 + (exp116/117 の best)

## Post-process (exp090 と同)
- lambda_prior 0.65, rank_aware power 0.6, TAX_SMOOTHING (α=0.15/0.05), Sonotype mirror

## Inputs
- `birdclef-2026` (competition)
- `jaejohn/perch-meta`, `rishikeshjani/perch-onnx-for-birdclef-2026`
- `tuckerarrants/bc2026-distilled-sed-public`, `maekeso/birdclef2026-exp029-l1-single`
- kernel_sources: `ashok205/tf-wheels`
- model_sources: `google/bird-vocalization-classifier/.../perch_v2_cpu/1`
- enable_internet: **False**
"""

nb = json.loads(P.read_text(encoding="utf-8"))
c0 = nb["cells"][0]
assert c0["cell_type"] == "markdown"
c0["source"] = HEADER.splitlines(keepends=True)
P.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

# verify
blob0 = "".join(json.loads(P.read_text(encoding="utf-8"))["cells"][0]["source"])
assert "exp078" not in blob0, "stale exp078"
assert "0.35" not in blob0.split("## Post-process")[0].replace("0.35 | **0.30**", ""), "stale 0.35 weight"
assert "exp110" in blob0 and "0.952" in blob0
print("exp110 header rewritten [OK]")
print("  - title/stack/weight reflect 0.30/0.40/0.30, LB 0.952")

# confirm blend weights still correct in code
blob = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
assert "BLEND_W_E10  = 0.30" in blob and "BLEND_W_SED  = 0.40" in blob and "BLEND_W_E17  = 0.30" in blob
print("  - blend code weights intact (0.30/0.40/0.30)")
