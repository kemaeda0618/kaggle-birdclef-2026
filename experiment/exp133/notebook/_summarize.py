import pandas as pd
D = pd.read_csv("experiment/exp133/notebook/_out/exp133_verify_v2.csv")
print(f"v2 OOF (n={len(D)} clips):")
print(f"{'alpha':>6} {'med':>5} {'mean':>6} {'top1':>6} {'top5':>6}")
for a in [0.0, 0.3, 0.5, 0.7, 1.0]:
    rk = D[f"rank_a{a}"]
    print(f"{a:>6} {rk.median():>5.1f} {rk.mean():>6.1f} {(rk==1).mean():>6.3f} {(rk<=5).mean():>6.3f}")
