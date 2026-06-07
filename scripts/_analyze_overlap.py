"""
Analyze species overlap between BirdCLEF 2026 and past years.
Downloads 2025/2021 CSVs via kaggle CLI, then cross-refs with 2026 train sample counts.
Run: python _analyze_overlap.py
"""
import subprocess, zipfile, sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
PAST = ROOT / "_tmp_past_years"

# ── 1. Download past-year CSVs ────────────────────────────────────────────────
COMPS = {
    "birdclef-2025": ["taxonomy.csv", "train.csv"],
    "birdclef-2021": ["train_metadata.csv"],
}

for comp, files in COMPS.items():
    d = PAST / comp
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        out = d / f
        if out.exists():
            print(f"[skip] {comp}/{f}")
            continue
        r = subprocess.run(
            ["kaggle", "competitions", "download", "-c", comp, "-f", f, "-p", str(d)],
            capture_output=True, text=True,
        )
        zip_p = d / (f + ".zip")
        if zip_p.exists():
            with zipfile.ZipFile(zip_p) as z:
                z.extractall(d)
            zip_p.unlink()
        print(f"{'OK' if out.exists() else 'FAIL'}: {comp}/{f}  {r.stderr.strip()[:60]}")

# ── 2. Load 2026 taxonomy + sample counts ────────────────────────────────────
tax26 = pd.read_csv(ROOT / "taxonomy.csv")
train26 = pd.read_csv(ROOT / "train.csv")

cnt = train26["primary_label"].value_counts().rename("n_train")
tax26 = tax26.join(cnt, on="primary_label")
tax26["n_train"] = tax26["n_train"].fillna(0).astype(int)
sp26 = set(tax26["scientific_name"].str.lower())

# ── 3. Load past-year species ────────────────────────────────────────────────
def get_sci(comp_dir: Path):
    sci = set()
    for fname in ["taxonomy.csv", "train.csv", "train_metadata.csv"]:
        p = comp_dir / fname
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        if "scientific_name" in df.columns:
            sci |= set(df["scientific_name"].dropna().str.strip().str.lower())
    return sci

past_sci = {}
for comp in COMPS:
    year = int(comp.split("-")[1])
    past_sci[year] = get_sci(PAST / comp)
    print(f"\nBirdCLEF {year}: {len(past_sci[year])} sci-name species loaded")

# ── 4. Overlap × 2026 sample counts ─────────────────────────────────────────
for year in sorted(past_sci.keys(), reverse=True):
    overlap_mask = tax26["scientific_name"].str.lower().isin(past_sci[year])
    overlap_df = tax26[overlap_mask][
        ["scientific_name", "common_name", "class_name", "n_train"]
    ].sort_values("n_train")

    print(f"\n{'='*60}")
    print(f"BirdCLEF {year}  — {len(overlap_df)} overlapping species")
    print(f"{'='*60}")
    print(overlap_df.to_string(index=False))

    # low-sample subset
    low = overlap_df[overlap_df["n_train"] < 20]
    print(f"\n  → n_train < 20: {len(low)} species")
    print(f"  → n_train == 0: {(overlap_df['n_train']==0).sum()} species")

# ── 5. Save CSV ───────────────────────────────────────────────────────────────
rows = []
for year, sci in past_sci.items():
    sub = tax26[tax26["scientific_name"].str.lower().isin(sci)].copy()
    sub["past_year"] = year
    rows.append(sub)

out = pd.concat(rows).sort_values(["past_year", "n_train"])
out_path = ROOT / "_overlap_species_counts.csv"
out.to_csv(out_path, index=False, encoding="utf-8")
print(f"\nSaved: {out_path}")
