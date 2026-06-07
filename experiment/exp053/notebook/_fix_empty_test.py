"""Fix Inference NB: handle empty test_soundscapes (Save Version 時の標準動作).

CLAUDE.md より: test_soundscapes ローカル/Save Version は空 (readme.txt のみ)
実 test data は Kaggle 自動 submit 時のみ配置。

Fix:
- Cell 7: 空でも DataFrame は proper columns で初期化
- Cell 8: 空 sub_df なら PP skip
- Cell 9: 空 sub_df なら sample_submission.csv ベースで empty submission 出力
"""
import json
from pathlib import Path

NB = Path(r"C:\Users\maeke\work\kaggle\birdclef-2026\experiment\exp053\notebook\nb_infer.ipynb")
nb = json.load(open(NB, encoding="utf-8"))

NEW_INFER_LOOP = '''# ============================================================
# Cell 7: Run inference on all test files (handle empty test_soundscapes)
# ============================================================
print(f"Starting inference on {len(test_files)} files...")

if len(test_files) == 0:
    print("\\n⚠️ test_soundscapes is empty (expected for Save Version)")
    print("   実 test data は Kaggle Submit 時に自動配置される")
    print("   この cell では empty sub_df 作成、submission.csv は dummy 出力")
    all_rows = []
else:
    start_t = time.time()
    all_rows = []
    for i, f in enumerate(tqdm(test_files, desc="infer")):
        rows = predict_file(f, model, mel_extractor)
        all_rows.extend(rows)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_t
            rate = (i + 1) / elapsed * 60
            eta = (len(test_files) - i - 1) / rate * 60 if rate > 0 else 0
            print(f"  [{i+1}/{len(test_files)}] {len(all_rows)} rows, {rate:.1f}/min, ETA {eta/60:.1f}min")
    elapsed = time.time() - start_t
    print(f"\\nInference DONE in {elapsed/60:.1f}min")

# Build sub_df with proper columns (even if empty)
cols = ["row_id"] + PRIMARY_LABELS
if all_rows:
    sub_df = pd.DataFrame(all_rows)
    # Ensure column order
    sub_df = sub_df[cols]
else:
    sub_df = pd.DataFrame(columns=cols)

print(f"\\nsub_df shape: {sub_df.shape}")
print(f"  columns count: {len(sub_df.columns)} (expected: {len(cols)})")
'''

NEW_PP = '''# ============================================================
# Cell 8: PP (TopN N=1, paper 256 spec) — skip if empty
# ============================================================
if len(sub_df) == 0:
    print("⚠️ sub_df is empty, skipping PP (will use sample_submission as template)")
else:
    print(f"Applying TopN N=1 PP (FCS_TOP_K={CFG.FCS_TOP_K}, FCS_POWER={CFG.FCS_POWER})...")
    # Extract file_id from row_id
    sub_df["_file_id"] = sub_df["row_id"].apply(lambda x: x.rsplit("_", 1)[0])

    prob_cols = [c for c in sub_df.columns if c in PRIMARY_LABELS]
    preds_array = sub_df[prob_cols].values.astype(np.float32)
    file_ids = sub_df["_file_id"].values

    unique_files = pd.unique(file_ids)
    file_to_idx = {f: i for i, f in enumerate(unique_files)}
    file_idx_arr = np.array([file_to_idx[f] for f in file_ids])

    for f_idx in range(len(unique_files)):
        mask = (file_idx_arr == f_idx)
        file_preds = preds_array[mask]
        if CFG.FCS_TOP_K == 1:
            file_max = file_preds.max(axis=0)
        else:
            sorted_preds = np.sort(file_preds, axis=0)
            file_max = sorted_preds[-CFG.FCS_TOP_K:].mean(axis=0)
        scale = np.power(file_max, CFG.FCS_POWER)
        preds_array[mask] = file_preds * scale[np.newaxis, :]

    for j, col in enumerate(prob_cols):
        sub_df[col] = preds_array[:, j]

    sub_df = sub_df.drop(columns=["_file_id"])
    print(f"PP done. preds range: [{preds_array.min():.4f}, {preds_array.max():.4f}]")
'''

NEW_SAVE = '''# ============================================================
# Cell 9: Save submission.csv (handle empty case)
# ============================================================
sample_sub_path = COMP_ROOT / "sample_submission.csv"
sample = pd.read_csv(sample_sub_path)
expected_cols = sample.columns.tolist()

if len(sub_df) == 0:
    print("⚠️ Empty sub_df, using sample_submission.csv as template (all zeros)")
    # Use sample_submission.csv structure (it has row_ids for test set)
    sub_df = sample.copy()
    print(f"  sample rows: {len(sub_df)}")
else:
    # Reorder columns to match sample
    sub_df = sub_df[expected_cols]
    print(f"Column order matched to sample_submission.csv ({len(expected_cols)} cols)")

OUT_PATH = Path("/kaggle/working/submission.csv")
sub_df.to_csv(OUT_PATH, index=False)
print(f"\\nSaved: {OUT_PATH}")
print(f"  shape: {sub_df.shape}")
print(f"  size: {OUT_PATH.stat().st_size/1e6:.1f} MB")
print(f"\\nFirst 3 rows:")
print(sub_df.head(3).iloc[:, :6])
'''

for c in nb["cells"]:
    if c.get("id") == "infer_loop":
        c["source"] = NEW_INFER_LOOP.splitlines(keepends=True)
    elif c.get("id") == "pp":
        c["source"] = NEW_PP.splitlines(keepends=True)
    elif c.get("id") == "save":
        c["source"] = NEW_SAVE.splitlines(keepends=True)

json.dump(nb, open(NB, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"Saved {NB}")
print("Fixed: empty test_soundscapes handling")
