"""Modify nb_blend_topn1.ipynb (exp040 copy):
1. Change FCS_TOP_K=2, FCS_POWER=0.4 → TOP_K=1, POWER=1.0 (paper 256 spec)
2. Update hdr cell to reflect TopN N=1 ablation
3. (No SLUG change here, handled by push script)
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp048\notebook\nb_blend_topn1.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

# ---- 1. config cell: FCS values ----
for c in nb["cells"]:
    if c.get("id") == "config":
        src = "".join(c["source"])
        old_block = "FCS_TOP_K = 2\nFCS_POWER = 0.4"
        new_block = "# ★ exp048: paper 256 TopN N=1 spec (was FCS_TOP_K=2, FCS_POWER=0.4 in exp040)\nFCS_TOP_K = 1\nFCS_POWER = 1.0"
        if old_block in src:
            src = src.replace(old_block, new_block)
            c["source"] = src.splitlines(keepends=True)
            print("[config] OK FCS values updated to paper 256 spec")
        else:
            print(f"[config] WARN: pattern not found, searching FCS_TOP_K mentions...")
            for line in src.split("\n"):
                if "FCS_TOP_K" in line or "FCS_POWER" in line:
                    print(f"  {line.strip()}")
        break

# ---- 2. hdr cell ----
for c in nb["cells"]:
    if c.get("id") == "hdr":
        src = "".join(c["source"])
        new_hdr_prefix = """# exp048: TopN N=1 PP Ablation on exp040 blend

**Base**: exp040 (LB 0.949、現状 best blend) を完全コピー
**唯一の変更**: FCS spec を paper 256 流に変更
- 旧 (exp040): `FCS_TOP_K = 2, FCS_POWER = 0.4` (top-2 平均 × 0.4 乗)
- 新 (exp048): `FCS_TOP_K = 1, FCS_POWER = 1.0` (max prob × 直接乗算 = paper 256 TopN N=1)

**期待 LB**: 0.949 → 0.950-0.954 (+0.001-0.005、gold border 突破可能性)
**根拠**: paper 256 BC25 2位 stepwise ablation で TopN N=1 PP = +0.014 LB 寄与、未試行

---

"""
        # Prepend to existing hdr (preserve exp040 description below)
        c["source"] = (new_hdr_prefix + src).splitlines(keepends=True)
        print("[hdr] OK prefix added")
        break

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nSaved {NB}")

# Verify
print("\n=== Verify ===")
for c in nb["cells"]:
    if c.get("id") == "config":
        src = "".join(c["source"])
        for line in src.split("\n"):
            if "FCS_TOP_K" in line or "FCS_POWER" in line:
                print(f"  [config] {line.strip()}")
