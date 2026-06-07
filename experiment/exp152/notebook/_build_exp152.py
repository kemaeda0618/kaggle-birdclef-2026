"""Build exp152: exp110 with seed 42 -> 618 (user's birthday). Last sub.

The single explicit seed in exp110 (cell v313_20) sets the ResidualSSM train/val file split:
    rng = torch.Generator(); rng.manual_seed(42)
Changing 42 -> 618 changes that split -> a different realization of the NB4 ResidualSSM
-> slightly different blend. Expected ~0.952 +/- seed noise. ONLY this one value changes.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path("experiment/exp110/notebook/nb_exp110_blend_30_40_30.ipynb")
DST = Path("experiment/exp152/notebook/nb_exp152_seed618.ipynb")
DST.parent.mkdir(parents=True, exist_ok=True)
nb = json.loads(SRC.read_text(encoding="utf-8"))

OLD = "rng.manual_seed(42)"
NEW = "rng.manual_seed(618)"

# whole-NB uniqueness check
total = sum("".join(c["source"]).count(OLD) for c in nb["cells"])
assert total == 1, f"expected exactly 1 seed occurrence, found {total}"

patched = False
for c in nb["cells"]:
    if c.get("id") != "v313_20":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    assert OLD in src, "seed line not in v313_20"
    c["source"] = src.replace(OLD, NEW).splitlines(keepends=True)
    patched = True
    break
assert patched, "v313_20 cell not found"

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {DST}")

# verify: exactly one 618, zero stale 42, blend cell untouched (still 30/40/30), compiles
chk = json.loads(DST.read_text(encoding="utf-8"))
tot618 = sum("".join(c["source"]).count(NEW) for c in chk["cells"])
tot42 = sum("".join(c["source"]).count(OLD) for c in chk["cells"])
assert tot618 == 1 and tot42 == 0, f"618={tot618}, 42={tot42}"
for c in chk["cells"]:
    if c.get("id") == "v313_20":
        s = "".join(c["source"])
        clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln for ln in s.splitlines())
        compile(clean, "<v313_20>", "exec")
    if c.get("id") == "blend":
        s = "".join(c["source"])
        assert "BLEND_W_E10  = 0.30" in s and "BLEND_W_SED  = 0.40" in s and "BLEND_W_E17  = 0.30" in s, "blend weights changed unexpectedly"
print("[OK] seed 42->618 (1 occurrence), blend 30/40/30 intact, v313_20 compiles")
