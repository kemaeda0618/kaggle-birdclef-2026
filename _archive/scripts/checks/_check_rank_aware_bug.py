"""Verify the duplicate kwarg bug in exp090."""
import json, io, sys, ast
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

nb90 = json.loads(Path("experiment/exp090/notebook/nb_exp090_tuned_tax.ipynb").read_text(encoding="utf-8"))

for c in nb90["cells"]:
    if c.get("id") == "exp037_full_pipeline":
        src = "".join(c["source"])
        lines = src.split("\n")
        # Find rank_aware_scaling block (multi-line call)
        for i, ln in enumerate(lines):
            if "rank_aware_scaling(" in ln and "def " not in ln:
                print(f"=== rank_aware_scaling call (around L{i}) ===")
                for j in range(max(0, i-1), min(len(lines), i+5)):
                    print(f"  L{j:3d}: {repr(lines[j])}")
                # Try to parse this specific call
                # Reconstruct multi-line expression
                # Find the closing paren
                expr_lines = []
                j = i
                paren_depth = 0
                while j < len(lines):
                    expr_lines.append(lines[j])
                    paren_depth += lines[j].count("(") - lines[j].count(")")
                    if paren_depth == 0 and "(" in "".join(expr_lines):
                        break
                    j += 1
                expr = "\n".join(expr_lines).lstrip()
                # Wrap to make parseable
                test_code = f"def f(probs, n_windows=12, power=0.5): pass\n" + expr
                try:
                    ast.parse(test_code)
                    print(f"\n  AST parse: OK (no error)")
                except SyntaxError as e:
                    print(f"\n  AST parse: SyntaxError at line {e.lineno}: {e.msg}")
                except Exception as e:
                    print(f"\n  AST parse: {type(e).__name__}: {e}")
        break
