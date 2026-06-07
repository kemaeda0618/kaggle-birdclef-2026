"""Test if duplicate kwarg in a multi-line function call is a Python error."""
import ast

code = '''def rank_aware_scaling(probs, n_windows=12, power=0.5): return probs
probs = "test"
N_WINDOWS = 12
probs = rank_aware_scaling(   probs, n_windows=N_WINDOWS, power=0.6,
                               power=0.4)
print("Reached end")
'''

print("--- ast.parse ---")
try:
    ast.parse(code)
    print("OK (no parse error)")
except SyntaxError as e:
    print(f"SyntaxError: {e.msg} at line {e.lineno}")

print("\n--- compile (compile-time check) ---")
try:
    compile(code, "<test>", "exec")
    print("OK (compile-time clear)")
except SyntaxError as e:
    print(f"SyntaxError: {e.msg} at line {e.lineno}")

print("\n--- exec (runtime) ---")
try:
    exec(compile(code, "<test>", "exec"))
    print("OK (executed)")
except SyntaxError as e:
    print(f"SyntaxError: {e.msg}")
except TypeError as e:
    print(f"TypeError: {e}")
