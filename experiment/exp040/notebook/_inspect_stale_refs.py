"""Find stale version references and outdated comments in exp040 NB."""
import json, re
nb = json.loads(open(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp040\notebook\nb_blend.ipynb", encoding="utf-8").read())

# Patterns to search for
stale_patterns = [
    r"v7\b", r"v6\b", r"v8\b", r"v9\b", r"v10\b",  # old version labels
    r"NB4 v\d",
    r"exp028", r"exp019", r"exp031", r"exp038",  # exp lineage references (might be stale in exp040)
    r"e17 R2", r"e17 \(", r"e14",  # old slot 3 names (before exp029 swap)
    r"ProtoSSM v\d",  # old ProtoSSM versions
]

print("=== Searching for stale references ===\n")
for ci, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    cid = c.get("id", "")
    hits = []
    for pat in stale_patterns:
        for m in re.finditer(pat, src):
            start = max(0, m.start() - 30)
            end = min(len(src), m.end() + 30)
            ctx = src[start:end].replace("\n", " ")
            hits.append((m.group(), ctx))
    if hits:
        print(f"[{ci:02d}] {cid}")
        for label, ctx in hits[:20]:
            print(f"    [{label}] ...{ctx}...")
