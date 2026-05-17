"""Inspect the structure of Tucker's audio_cache_meta.csv."""
import pandas as pd
from pathlib import Path

META_DIR = Path("./tucker_meta_test")

print("=== audio_cache_meta.csv ===")
df = pd.read_csv(META_DIR / "audio_cache_meta.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nDtypes:")
print(df.dtypes)
print(f"\nHead:")
print(df.head(5).to_string())
print(f"\nUnique primary_label count: {df['primary_label'].nunique() if 'primary_label' in df.columns else 'N/A'}")
print(f"\nUnique filename count: {df['filename'].nunique() if 'filename' in df.columns else 'N/A'}")

print("\n\n=== soundscape_cache_meta.csv ===")
df_sc = pd.read_csv(META_DIR / "soundscape_cache_meta.csv")
print(f"Shape: {df_sc.shape}")
print(f"Columns: {df_sc.columns.tolist()}")
print(f"\nDtypes:")
print(df_sc.dtypes)
print(f"\nHead:")
print(df_sc.head(5).to_string())
