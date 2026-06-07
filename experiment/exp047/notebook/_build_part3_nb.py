"""Build Part 3 NB: XC API v3 for non-Aves species (72 species).

Based on Part 2 NB structure, replace species list with non-Aves.
Output: birdclef2026-xc-api-dl-part3 Kaggle Dataset
"""
import json, shutil
from pathlib import Path

SRC = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_xc_api_dl_part2.ipynb")
DST = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_xc_api_dl_part3.ipynb")
shutil.copy(SRC, DST)
nb = json.load(open(DST, encoding="utf-8"))

# Load non-Aves list
non_aves_src = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\_non_aves_list.py").read_text(encoding="utf-8")
non_aves_lines = []
for line in non_aves_src.split("\n"):
    s = line.strip()
    if s.startswith("(") and s.endswith("),"):
        non_aves_lines.append("    " + s)
non_aves_block = "\n".join(non_aves_lines)
print(f"Non-Aves species: {len(non_aves_lines)}")

# 1. hdr update
for c in nb["cells"]:
    if c.get("id") == "hdr":
        src = "".join(c["source"])
        new_prefix = """# BC2026 XC API Downloader **Part 3** (72 non-Aves species)

Part 1 (76 Aves) + Part 2 (83 Aves) で Aves 159 species 完了。
**Part 3 = 残り non-Aves 72 species (Amphibia 35 + Insecta 28 + Mammalia 8 + Reptilia 1)**

XC は鳥が main だが non-Aves 録音もあり、Amphibia/Mammalia 中心に取得期待。

## 変更点 (vs Part 2)
- species list: 72 non-Aves (Amphibia, Insecta, Mammalia, Reptilia)
- Output dir: `/kaggle/working/xc_api_part3/`
- Output Dataset: `birdclef2026-xc-api-aves-audio-part3`

## 期待

- 取得可能: Amphibia ~10-20、Mammalia 2-5、一部 Insecta、Reptilia 0-1
- 取得不能: XC に存在しない species 多数 (skip, no error)
- ~5-15 GB 想定

---

"""
        # Remove old Part 2 prefix, prepend new Part 3
        # Find original exp047 hdr (after Part 2 prefix)
        if "Part 2" in src:
            # Strip Part 2 prefix
            lines = src.split("\n")
            # Find "---" separator (end of Part 2 hdr block)
            sep_idx = None
            for i, line in enumerate(lines):
                if line.strip() == "---" and i < 30:
                    sep_idx = i
                    break
            if sep_idx is not None:
                # Keep everything after the first "---" (original content)
                original = "\n".join(lines[sep_idx+1:])
                src = new_prefix + original
            else:
                src = new_prefix + src
        c["source"] = src.splitlines(keepends=True)
        print("[hdr] OK Part 3 prefix")
        break

# 2. setup: change OUT_DIR
for c in nb["cells"]:
    if c.get("id") == "setup":
        src = "".join(c["source"])
        src = src.replace(
            'OUT_DIR = Path("/kaggle/working/xc_api_part2")  # ★ Part 2: 別 dir',
            'OUT_DIR = Path("/kaggle/working/xc_api_part3")  # ★ Part 3: non-Aves'
        )
        c["source"] = src.splitlines(keepends=True)
        print("[setup] OUT_DIR -> xc_api_part3")
        break

# 3. aves_list cell: replace Aves with non-Aves list
for c in nb["cells"]:
    if c.get("id") == "aves_list":
        # Replace entire content with non-Aves version
        new_src = f'''# ============================================================
# Cell 2: BC2026 non-Aves species list (embedded, 72 species)
# ============================================================
__BC2026_NON_AVES_RAW = [
{non_aves_block}
]

# Convert to df with columns matching Stage 1 pipeline
aves_df = pd.DataFrame(__BC2026_NON_AVES_RAW, columns=["scientific_name", "primary_label", "class_name"])
print(f"BC2026 non-Aves species: {{len(aves_df)}}")
print(f"  classes: {{aves_df['class_name'].value_counts().to_dict()}}")
print(aves_df.head().to_string(index=False))

def parse_sci_name(name):
    parts = str(name).strip().split()
    if len(parts) >= 2:
        return parts[0], " ".join(parts[1:])
    return parts[0], ""

aves_df[["genus", "species"]] = aves_df["scientific_name"].apply(
    lambda x: pd.Series(parse_sci_name(x))
)
print(f"\\nUnique genera: {{aves_df['genus'].nunique()}}")

# No skip list (Part 3 = fresh non-Aves, no prior runs)
'''
        c["source"] = new_src.splitlines(keepends=True)
        print(f"[aves_list] replaced with non-Aves list ({len(non_aves_lines)} species)")
        break

# 4. save_dataset: new slug
for c in nb["cells"]:
    if c.get("id") == "save_dataset":
        src = "".join(c["source"])
        src = src.replace(
            'SLUG = "birdclef2026-xc-api-aves-audio-part2"',
            'SLUG = "birdclef2026-xc-api-aves-audio-part3"'
        )
        src = src.replace(
            "BirdCLEF2026 XC API Aves Audio (Part 2)",
            "BirdCLEF2026 XC API non-Aves Audio (Part 3)"
        )
        c["source"] = src.splitlines(keepends=True)
        print("[save_dataset] SLUG -> aves-audio-part3")
        break

json.dump(nb, open(DST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nSaved {DST}")
print(f"  cells: {len(nb['cells'])}")
