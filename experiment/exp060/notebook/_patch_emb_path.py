"""Patch EMB_DIR in exp060/exp062 NBs to support private kernel_source path."""
import json, re
from pathlib import Path

# Old path (public):    /kaggle/input/notebooks/maekeso/<slug>/
# New path (private):   /kaggle/input/maekeso/<slug>/
# Need fallback to try both.

OLD_BLOCK = 'EMB_DIR = Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp010-nb1-embedding")'
NEW_BLOCK = '''_emb_candidates = [
    Path("/kaggle/input/notebooks/maekeso/birdclef2026-exp010-nb1-embedding"),
    Path("/kaggle/input/maekeso/birdclef2026-exp010-nb1-embedding"),  # private kernel_source
    Path("/kaggle/input/birdclef2026-exp010-nb1-embedding"),
]
EMB_DIR = next((p for p in _emb_candidates if p.exists()), _emb_candidates[0])
if not EMB_DIR.exists():
    for _hit in Path("/kaggle/input").rglob("soundscape_embeddings.npz"):
        EMB_DIR = _hit.parent; break'''

NBS = [
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp060\notebook\nb_blend_inline.ipynb"),
    Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp062\notebook\nb_blend_aves_exp032.ipynb"),
]

for NB_PATH in NBS:
    print(f"\n=== {NB_PATH.name} ===")
    nb = json.load(open(NB_PATH, encoding="utf-8"))
    patched = False
    for c in nb["cells"]:
        if c.get("id") == "config":
            src = "".join(c["source"])
            if OLD_BLOCK in src:
                src = src.replace(OLD_BLOCK, NEW_BLOCK)
                c["source"] = src.splitlines(keepends=True)
                print(f"  config cell patched")
                patched = True
            else:
                print(f"  WARN: OLD_BLOCK not found in config")
            break
    if patched:
        json.dump(nb, open(NB_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  Saved {NB_PATH.name}")
    else:
        print(f"  Skipped (no patch applied)")
