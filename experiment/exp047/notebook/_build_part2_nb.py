"""Build Part 2 NB: clone existing xc-api-dl + add skip list + cap.

Goal: cover remaining 86 species (162 - 76 done) in a separate run.
"""
import json, shutil
from pathlib import Path

SRC = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_xc_api_dl.ipynb")
DST = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_xc_api_dl_part2.ipynb")
shutil.copy(SRC, DST)
nb = json.load(open(DST, encoding="utf-8"))

# Load completed list
completed_src = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_completed_list.py").read_text(encoding="utf-8")
# Extract set elements (just the entries inside {...})
completed_lines = []
for line in completed_src.split("\n"):
    s = line.strip()
    if s.startswith("'") and s.endswith("',"):
        completed_lines.append("    " + s)
completed_block = "\n".join(completed_lines)

# 1. hdr cell - update description
for c in nb["cells"]:
    if c.get("id") == "hdr":
        src = "".join(c["source"])
        new_prefix = """# BC2026 XC API Downloader **Part 2** (残 86 species)

Part 1 で 76 species 完了 (storage 18GB cap到達)。
Part 2 は **残り 86 species** を取得、別 Dataset 化。

## 変更点 (vs Part 1)
- Output dir: `/kaggle/working/xc_api_part2/` (Part 1 と分離)
- **COMPLETED_SCI_NAMES skip list** (Part 1 完了 76 species を query しない)
- Output Dataset: `birdclef2026-xc-api-aves-audio-part2`

---

"""
        c["source"] = (new_prefix + src).splitlines(keepends=True)
        print("[hdr] prefix added")
        break

# 2. setup cell - change output dir
for c in nb["cells"]:
    if c.get("id") == "setup":
        src = "".join(c["source"])
        src = src.replace(
            'OUT_DIR = Path("/kaggle/working/xc_api")',
            'OUT_DIR = Path("/kaggle/working/xc_api_part2")  # ★ Part 2: 別 dir'
        )
        c["source"] = src.splitlines(keepends=True)
        print("[setup] OUT_DIR -> xc_api_part2")
        break

# 3. aves_list cell - add COMPLETED_SCI_NAMES + filter Aves
for c in nb["cells"]:
    if c.get("id") == "aves_list":
        src = "".join(c["source"])
        # Add skip list at end
        skip_block = f"""

# ============================================================
# Part 2: Skip 76 species already completed in Part 1
# ============================================================
COMPLETED_SCI_NAMES = {{
{completed_block}
}}
print(f"\\n[Part 2] Skip list: {{len(COMPLETED_SCI_NAMES)}} completed species")

# Filter aves_df to only remaining species
n_before = len(aves_df)
aves_df = aves_df[~aves_df['scientific_name'].isin(COMPLETED_SCI_NAMES)].reset_index(drop=True)
print(f"[Part 2] aves_df filtered: {{n_before}} -> {{len(aves_df)}} (remaining species to query/DL)")
"""
        src += skip_block
        c["source"] = src.splitlines(keepends=True)
        print(f"[aves_list] COMPLETED skip list added ({completed_block.count(chr(10)) + 1} species)")
        break

# 4. save_dataset cell - new slug
for c in nb["cells"]:
    if c.get("id") == "save_dataset":
        src = "".join(c["source"])
        src = src.replace(
            'SLUG = "birdclef2026-xc-api-aves-audio"',
            'SLUG = "birdclef2026-xc-api-aves-audio-part2"'
        )
        src = src.replace(
            '"title": "BirdCLEF2026 XC API Aves Audio"',
            '"title": "BirdCLEF2026 XC API Aves Audio (Part 2)"'
        )
        # Also: change DRY_RUN default to False for Part 2 (we already trust the flow)
        # but actually keep it True for safety
        c["source"] = src.splitlines(keepends=True)
        print("[save_dataset] SLUG -> aves-audio-part2")
        break

json.dump(nb, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nSaved {DST}")
print(f"  cells: {len(nb['cells'])}")
