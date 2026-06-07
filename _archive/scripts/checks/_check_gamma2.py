"""Check POWER_GAMMA usage with absolute paths."""
import json
from pathlib import Path

BASE = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment")
files = {
    "exp017 R1": BASE / "exp017" / "notebook" / "nb_train.ipynb",
    "exp017 R2": BASE / "exp017" / "notebook" / "nb_train_r2.ipynb",
    "exp029 R3 (gen)": BASE / "exp029" / "notebook" / "_gen_nb_train_r3_l1_single.py",
    "exp046 R3 γ=1.2": BASE / "exp046" / "notebook" / "nb_train_r3_l1_gamma12.ipynb",
    "exp045 R3 filter": BASE / "exp045" / "notebook" / "nb_train_r3_l1_filtered.ipynb",
    "exp029 R4": BASE / "exp029" / "notebook" / "nb_train_r4_l1_single.ipynb",
}

for name, fp in files.items():
    if not fp.exists():
        print(f"\n=== {name}: NOT FOUND ({fp}) ===")
        continue
    print(f"\n=== {name} ===")
    if fp.suffix == ".ipynb":
        nb = json.loads(fp.read_text(encoding="utf-8"))
        all_src = "\n".join("".join(c.get("source", []))
                            for c in nb["cells"] if c["cell_type"] == "code")
    else:
        all_src = fp.read_text(encoding="utf-8")

    # Find lines mentioning POWER_GAMMA or pseudo gamma/power
    print("  --- POWER_GAMMA / pseudo gamma ---")
    found = 0
    for ln, line in enumerate(all_src.splitlines(), 1):
        line_l = line.lower()
        if ("power_gamma" in line_l or "powergamma" in line_l) and "#" not in line.strip()[:3]:
            print(f"  L{ln}: {line.strip()[:180]}")
            found += 1
        elif "pseudo" in line_l and ("gamma" in line_l or "power" in line_l):
            print(f"  L{ln}: {line.strip()[:180]}")
            found += 1
    if found == 0:
        print("  (no POWER_GAMMA pseudo usage)")

    # All POWER_GAMMA mentions including comments
    print("  --- All POWER_GAMMA mentions ---")
    for ln, line in enumerate(all_src.splitlines(), 1):
        if "POWER_GAMMA" in line or "power_gamma" in line:
            print(f"  L{ln}: {line.strip()[:180]}")

    # Look for np.power on pseudo / soft labels
    print("  --- np.power / .pow on pseudo or label ---")
    for ln, line in enumerate(all_src.splitlines(), 1):
        if ("np.power" in line or ".pow(" in line or "torch.pow" in line):
            # Check context
            print(f"  L{ln}: {line.strip()[:180]}")
