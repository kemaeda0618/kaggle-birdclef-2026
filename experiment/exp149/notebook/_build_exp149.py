"""Build exp149: exp110 + site/hour prior applied LAST (after mirror/FCS/tax smoothing).

Mathematically correct order (vs exp148 which applied prior FIRST and contaminated the
magnitude-sensitive FCS / mirror / tax that were tuned on the natural blend distribution):
    rank-blend -> mirror -> FCS -> tax smoothing -> [prior LAST]
The value-shaping PP run in their tuned regime (mean~0.5); the metadata prior is the final
per-column re-ranking, undistorted.

CRITICAL alignment: after tax smoothing the submission is in sample_sub row order (reindexed),
NOT the original meta_te/blend order. So site/hour are aligned by ROW_ID (merge), not positionally.
Missing/NaN row_ids -> "__UNK__"/-1 -> apply_prior falls back to global_p (no crash).
lambda=0.30 (same as exp148) to isolate the ORDER effect.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path("experiment/exp110/notebook/nb_exp110_blend_30_40_30.ipynb")
DST = Path("experiment/exp149/notebook/nb_exp149_prior_last.ipynb")
DST.parent.mkdir(parents=True, exist_ok=True)
nb = json.loads(SRC.read_text(encoding="utf-8"))

TOCSV = 'submission.to_csv("submission.csv", index=False)'
TAX_LINE = "submission = f_tax_smoothing(submission, genus_alpha=0.15, class_alpha=0.05)"

PRIOR_BLOCK = (
    "# === exp149: site/hour prior applied LAST (after mirror/FCS/tax) — math-correct order ===\n"
    "# submission is in sample_sub order now -> align site/hour by ROW_ID (not positional).\n"
    "LAMBDA_POST_PRIOR = 0.30\n"
    "_pp_cols = PRIMARY_LABELS\n"
    "_pp_md = meta_te.drop_duplicates('row_id').copy()\n"
    "_pp_md['row_id'] = _pp_md['row_id'].astype(str)\n"
    "_pp_md = _pp_md.set_index('row_id')\n"
    "_pp_aligned = _pp_md.reindex(submission['row_id'].astype(str))\n"
    "_pp_sites = _pp_aligned['site'].astype(str).fillna('__UNK__').to_numpy()\n"
    "_pp_hours = pd.to_numeric(_pp_aligned['hour_utc'], errors='coerce').fillna(-1).astype(int).to_numpy()\n"
    "assert len(_pp_sites) == len(submission) and len(_pp_hours) == len(submission), 'prior row-align mismatch'\n"
    "_pp_P = submission[_pp_cols].values.astype(np.float32)\n"
    "_pp_eps = 1e-4\n"
    "_pp_Pc = np.clip(_pp_P, _pp_eps, 1.0 - _pp_eps)\n"
    "_pp_L = (np.log(_pp_Pc) - np.log1p(-_pp_Pc)).astype(np.float32)\n"
    "_pp_L = apply_prior(_pp_L, sites=_pp_sites, hours=_pp_hours, tables=prior_tables, lambda_prior=LAMBDA_POST_PRIOR)\n"
    "submission[_pp_cols] = (1.0 / (1.0 + np.exp(-np.clip(_pp_L, -50, 50)))).astype(np.float32)\n"
    "print(f\"  [exp149] prior applied LAST: lambda={LAMBDA_POST_PRIOR}, mean {submission[_pp_cols].values.mean():.4f}\")\n"
    + TOCSV
)

patched = False
for c in nb["cells"]:
    if c.get("id") != "blend":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    assert TAX_LINE in src, "tax smoothing line not found"
    assert TOCSV in src, "to_csv anchor not found"
    assert src.count(TOCSV) == 1, "to_csv not unique"
    # prior must land AFTER tax smoothing: tax line index < to_csv index
    assert src.index(TAX_LINE) < src.index(TOCSV), "tax smoothing must precede to_csv"
    src2 = src.replace(TOCSV, PRIOR_BLOCK)
    c["source"] = src2.splitlines(keepends=True)
    patched = True
    break
assert patched, "blend cell not found"

# global-scope sanity: needed symbols defined earlier
all_src = "\n".join("".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code")
assert "def apply_prior(" in all_src
assert "prior_tables   = build_prior_tables(" in all_src or "prior_tables = build_prior_tables(" in all_src
assert "meta_te, sc_te, emb_te = run_perch(" in all_src

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {DST}")

chk = json.loads(DST.read_text(encoding="utf-8"))
for c in chk["cells"]:
    if c.get("id") == "blend":
        s = "".join(c["source"])
        assert s.count("LAMBDA_POST_PRIOR = 0.30") == 1, "prior block not inserted once"
        assert s.count(TOCSV) == 1, "to_csv lost/duplicated"
        # prior block AFTER tax smoothing AND immediately before to_csv
        assert s.index(TAX_LINE) < s.index("LAMBDA_POST_PRIOR") < s.index(TOCSV), "prior not placed last (after tax, before to_csv)"
        # alignment is by row_id (not positional)
        assert "_pp_aligned = _pp_md.reindex(submission['row_id'].astype(str))" in s, "row_id alignment missing"
        clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln for ln in s.splitlines())
        compile(clean, "<blend>", "exec")
        print("[OK] prior placed LAST (after tax, before to_csv), row_id-aligned, compiles")
        break
