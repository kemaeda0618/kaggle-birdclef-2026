"""Build exp148: exp110 + POST-BLEND site/hour prior (applied once to the full 3-way blend).

Rationale: in exp110 the site/hour prior (apply_prior, lambda=0.65) is applied ONLY to the NB4 stream,
so its influence on the final blend is diluted to ~NB4's blend weight (0.30). exp148 applies the SAME
prior ONCE to the rank-blended output (logit space) so all 3 streams benefit equally.

Implementation safety (verified against exp110 source):
  - blend_flat rows align positionally with meta_te (all_row_ids = meta_te["row_id"]; same test_paths order).
  - Reuses the existing apply_prior() + prior_tables (no reimplementation).
  - apply_prior falls back to global_p for unknown site/hour -> no hidden-test crash.
  - lambda re-tuned smaller (rank-logit scale != raw Perch logit scale): LAMBDA_POST_PRIOR=0.30.
ONLY change: insert one block before the Sonotype mirror in the blend cell. Nothing else touched.
"""
import io, json, sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = Path("experiment/exp110/notebook/nb_exp110_blend_30_40_30.ipynb")
DST = Path("experiment/exp148/notebook/nb_exp148_postblend_prior.ipynb")
DST.parent.mkdir(parents=True, exist_ok=True)
nb = json.loads(SRC.read_text(encoding="utf-8"))

ANCHOR = "# === Sonotype mirror (公開 0.946 NB Cell 40) ==="
PRIOR_BLOCK = (
    "# === exp148: post-blend site/hour prior (apply ONCE to full blend, not just NB4) ===\n"
    "# blend_flat rows align with meta_te (all_row_ids = meta_te['row_id']); reuse apply_prior + prior_tables.\n"
    "LAMBDA_POST_PRIOR = 0.30\n"
    "assert len(blend_flat) == len(meta_te), f\"row mismatch blend={len(blend_flat)} meta_te={len(meta_te)}\"\n"
    "_pp_eps = 1e-4\n"
    "_pp_bf = np.clip(blend_flat, _pp_eps, 1.0 - _pp_eps)\n"
    "_pp_logit = (np.log(_pp_bf) - np.log1p(-_pp_bf)).astype(np.float32)\n"
    "_pp_logit = apply_prior(\n"
    "    _pp_logit,\n"
    "    sites=meta_te['site'].to_numpy(),\n"
    "    hours=meta_te['hour_utc'].to_numpy(),\n"
    "    tables=prior_tables,\n"
    "    lambda_prior=LAMBDA_POST_PRIOR,\n"
    ")\n"
    "blend_flat = (1.0 / (1.0 + np.exp(-np.clip(_pp_logit, -50, 50)))).astype(np.float32)\n"
    "print(f\"  [exp148] post-blend prior: lambda={LAMBDA_POST_PRIOR}, blend mean {blend_flat.mean():.4f}\")\n"
    + ANCHOR
)

patched = False
for c in nb["cells"]:
    if c.get("id") != "blend":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    assert ANCHOR in src, "Sonotype mirror anchor not found in blend cell"
    assert src.count(ANCHOR) == 1, "anchor not unique"
    # confirm the variables we rely on are referenced/used in this NB earlier
    assert "blend_flat = (BLEND_W_E10" in src, "blend_flat rank-blend line missing (unexpected blend cell)"
    src2 = src.replace(ANCHOR, PRIOR_BLOCK)
    c["source"] = src2.splitlines(keepends=True)
    patched = True
    break
assert patched, "blend cell not found"

# global-scope sanity: apply_prior / prior_tables / meta_te must be DEFINED earlier in the NB
all_src = "\n".join("".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code")
assert "def apply_prior(" in all_src, "apply_prior not defined in NB"
assert "prior_tables   = build_prior_tables(" in all_src or "prior_tables = build_prior_tables(" in all_src, "prior_tables not built"
assert "meta_te, sc_te, emb_te = run_perch(" in all_src, "meta_te not built"

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {DST}")

# verify the patched blend cell compiles and contains exactly one prior block
chk = json.loads(DST.read_text(encoding="utf-8"))
for c in chk["cells"]:
    if c.get("id") == "blend":
        s = "".join(c["source"])
        assert s.count("LAMBDA_POST_PRIOR = 0.30") == 1, "prior block not inserted exactly once"
        assert "apply_prior(" in s and "_pp_logit" in s
        assert s.count(ANCHOR) == 1, "mirror anchor lost/duplicated"
        # mirror must come AFTER the prior block
        assert s.index("LAMBDA_POST_PRIOR") < s.index(ANCHOR), "prior block not before mirror"
        clean = "\n".join("# " + ln if ln.lstrip().startswith(("!", "%")) else ln for ln in s.splitlines())
        compile(clean, "<blend>", "exec")
        print("[OK] post-blend prior inserted before mirror, blend cell compiles, anchors intact")
        break
