"""v5 patch for exp069a: Fix b0 weight filename typo.

V4 で b0 (Amphibia/Insecta sub model) の ckpt filename を `22_seed_15_epoch.pt` と書いたが、
実際の dataset listing は `22_seed_40_epoch.pt`。typo 修正。
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_babych.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

OLD = 'tf_efficientnet_b0.ns_jft_in1k_incest_amphibia_128_bs_0.0_drop_path_rate_20_duration_sed_type_0.5_mixup_p_(224, 512)_size_ce_4096_n_fft_full_data_22_seed_15_epoch.pt'
NEW = 'tf_efficientnet_b0.ns_jft_in1k_incest_amphibia_128_bs_0.0_drop_path_rate_20_duration_sed_type_0.5_mixup_p_(224, 512)_size_ce_4096_n_fft_full_data_22_seed_40_epoch.pt'

total = 0
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") != "code": continue
    src = "".join(c["source"])
    if OLD in src:
        new_src = src.replace(OLD, NEW)
        c["source"] = new_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None
        total += 1
        print(f"  Cell {i}: b0 filename fix {OLD[-30:]} → {NEW[-30:]}")

assert total >= 1, "b0 typo not found"
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"v5 patched: {NB_PATH}")
