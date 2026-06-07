"""Fix dataset paths in extract NB:
- iNat: /kaggle/input/datasets/shadowdude/train-recordings
- AnuraSet: /kaggle/input/datasets/bengtlueers/anuraset-v2-raw
- Competition data: /kaggle/input/competitions/birdclef-<year>
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp047\notebook\nb_extract_pretrain.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

# Patch extract_inat cell
for c in nb["cells"]:
    if c.get("id") == "extract_inat":
        src = "".join(c["source"])
        old = """INAT_ROOT_CANDIDATES = [
    Path("/kaggle/input/train-recordings"),
    Path("/kaggle/input/shadowdude-train-recordings"),
]"""
        new = """INAT_ROOT_CANDIDATES = [
    Path("/kaggle/input/datasets/shadowdude/train-recordings"),  # ★ 正規 path
    Path("/kaggle/input/train-recordings"),                       # legacy fallback
    Path("/kaggle/input/shadowdude-train-recordings"),
]"""
        if old in src:
            src = src.replace(old, new)
            c["source"] = src.splitlines(keepends=True)
            print("[extract_inat] OK iNat paths updated")
        else:
            print(f"[extract_inat] WARN: pattern not found")
        break

# Patch extract_anuraset cell
for c in nb["cells"]:
    if c.get("id") == "extract_anuraset":
        src = "".join(c["source"])
        old = """ANURA_CANDIDATES = [
    Path("/kaggle/input/anuraset-v2-raw"),
    Path("/kaggle/input/bengtlueers-anuraset-v2-raw"),
]"""
        new = """ANURA_CANDIDATES = [
    Path("/kaggle/input/datasets/bengtlueers/anuraset-v2-raw"),  # ★ 正規 path
    Path("/kaggle/input/anuraset-v2-raw"),                        # legacy fallback
    Path("/kaggle/input/bengtlueers-anuraset-v2-raw"),
]"""
        if old in src:
            src = src.replace(old, new)
            c["source"] = src.splitlines(keepends=True)
            print("[extract_anuraset] OK AnuraSet paths updated")
        else:
            print(f"[extract_anuraset] WARN: pattern not found")
        break

# Also: Competition data — already uses /kaggle/input/competitions/<slug>/ in v3, no change needed

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nSaved {NB}")

# Verify
print("\n=== Verify ===")
for c in nb["cells"]:
    if c.get("id") in ("extract_inat", "extract_anuraset"):
        src = "".join(c["source"])
        for line in src.split("\n"):
            if "input/datasets" in line or "INAT_ROOT_CANDIDATES" in line or "ANURA_CANDIDATES" in line:
                print(f"  [{c.get('id')}] {line.strip()[:120]}")
