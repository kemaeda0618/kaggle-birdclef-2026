"""Species type signature analysis."""
import pandas as pd
import ast

df = pd.read_csv("train.csv")

def parse_types(s):
    try:
        return [t.strip() for t in ast.literal_eval(s)]
    except:
        return []

df["type_list"] = df["type"].apply(parse_types)

# Normalize types: focus on canonical categories
CANONICAL = {
    "song", "call", "flight call", "alarm call", "dawn song",
    "subsong", "duet", "drumming", "begging call", "advertisement call",
    "territorial call", "social call", "canto", "chamado",
}

# For each species, compute % of samples with each canonical type
print("=" * 90)
print("=== Species with HIGHLY dominant canonical type (>80% of samples) ===")
print("=" * 90)

results = []
for sp in df["primary_label"].unique():
    sub = df[df["primary_label"] == sp]
    n_total = len(sub)
    if n_total < 10:
        continue
    # Count samples per canonical type
    for t in CANONICAL:
        n_with_t = sum(1 for tl in sub["type_list"] if t in tl)
        if n_with_t / n_total >= 0.80:
            sci = sub["scientific_name"].iloc[0]
            cls = sub["class_name"].iloc[0]
            results.append((sp, cls, sci, t, n_with_t, n_total))

# Sort by class then by sample count
results.sort(key=lambda x: (x[1], -x[5]))
for sp, cls, sci, t, n_with_t, n_total in results:
    print(f"  {sp:15} {cls:9} {sci[:30]:30} type={t:25} {n_with_t}/{n_total} ({n_with_t/n_total*100:.0f}%)")

print()
print(f"Total species with >80% dominant canonical type (n>=10): {len(results)}")
print()

# Class-level pattern
print("=" * 90)
print("=== Class-level type signature ===")
print("=" * 90)
for cls in ["Amphibia", "Insecta", "Mammalia", "Reptilia"]:
    sub = df[df["class_name"] == cls]
    print(f"\n{cls} ({sub['primary_label'].nunique()} species, {len(sub)} samples):")
    # All types appearing
    all_t = []
    for tl in sub["type_list"]:
        all_t.extend(tl)
    counts = pd.Series(all_t).value_counts()
    n_empty = (sub["type_list"].apply(len) == 0).sum()
    print(f"  Empty type tags: {n_empty}/{len(sub)} ({n_empty/len(sub)*100:.0f}%)")
    print("  Top 5 types:")
    for t, n in counts.head(5).items():
        print(f"    {t}: {n}")
