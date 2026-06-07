"""Build exp112 V5b: V4 + batch=12 SEQUENTIAL (no async).

V5 async caused kernel died. Revert to safer config:
  - Keep BATCH_FILES_E106 = 12 (big batch, simple)
  - Drop async API
  - Drop INFERENCE_NUM_THREADS hint (use OV default)
  - Tucker SED: BATCH_FILES_SED = 6
"""
import io, json, sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

V4_NB = Path("experiment/exp112/notebook/nb_exp112_v4_pathfix.ipynb")
V5B_NB = Path("experiment/exp112/notebook/nb_exp112_v5b_batch12.ipynb")

nb = json.loads(V4_NB.read_text(encoding="utf-8"))

# Patch sed-infer: change BATCH_FILES_SED 3 -> 6
sed_patched = False
for c in nb["cells"]:
    if c.get("id") != "sed-infer":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    assert "BATCH_FILES_SED = 3" in src, "BATCH_FILES_SED anchor not found"
    src = src.replace("BATCH_FILES_SED = 3", "BATCH_FILES_SED = 6")
    c["source"] = src.splitlines(keepends=True)
    sed_patched = True

# Patch e29-infer: change BATCH_FILES_E106 3 -> 12 (sequential, no async)
e29_patched = False
for c in nb["cells"]:
    if c.get("id") != "e29-infer":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    assert "BATCH_FILES_E106 = 3" in src, "BATCH_FILES_E106 anchor not found"
    src = src.replace("BATCH_FILES_E106 = 3", "BATCH_FILES_E106 = 12   # ★ V5b: big batch sequential")
    c["source"] = src.splitlines(keepends=True)
    e29_patched = True

assert sed_patched and e29_patched, f"patches missing: sed={sed_patched}, e29={e29_patched}"

V5B_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {V5B_NB} ({V5B_NB.stat().st_size} bytes)")

# verify
nb_check = json.loads(V5B_NB.read_text(encoding="utf-8"))
for c in nb_check["cells"]:
    cid = c.get("id")
    if cid == "sed-infer":
        s = "".join(c["source"])
        assert "BATCH_FILES_SED = 6" in s
        print("  [OK] sed-infer: BATCH=6")
    elif cid == "e29-infer":
        s = "".join(c["source"])
        assert "BATCH_FILES_E106 = 12" in s
        # confirm no async
        assert "start_async" not in s
        assert "INFERENCE_NUM_THREADS" not in s
        print("  [OK] e29-infer: BATCH=12 sequential (no async, no thread hint)")
