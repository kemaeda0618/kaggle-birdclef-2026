"""Fix slug: birdclef2026-exp102-l1-babych -> birdclef2026-exp102-l1-3fold-r3p"""
import io, sys, json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Fix gen script
SRC = Path(__file__).with_name("_gen_nb_train_r3.py")
content = SRC.read_text(encoding="utf-8")

patches = [
    ("birdclef2026-exp102-l1-babych", "birdclef2026-exp102-l1-3fold-r3p"),
    ("birdclef2026 exp102 l1 babych (R3 from e17, mixup 1.0 + dp 0.15 + WRS)",
     "birdclef2026 exp102 l1 3fold r3p (mixup 1.0 + dp 0.15 + WRS warm-up + EMA, 30 ep)"),
]
n = 0
for old, new in patches:
    if old in content:
        cnt = content.count(old)
        content = content.replace(old, new)
        n += cnt
        print(f"[{cnt}x OK] {old[:70]!r}")
SRC.write_text(content, encoding="utf-8")
print(f"\n[gen script] Total: {n}")

# Re-gen ipynb
import subprocess
r = subprocess.run([sys.executable, str(SRC)], capture_output=True, text=True, encoding="utf-8")
print(r.stdout)
if r.returncode != 0:
    print("[ERROR]", r.stderr)
