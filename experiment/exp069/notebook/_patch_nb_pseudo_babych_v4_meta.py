"""v4 patch for exp069a: Restore MODELS_GROUP_META_1 list in Cell 14.

V2 で Cell 14 を rewrite した際に MODELS_GROUP_META_1 = [...] の定義が消えた。
prepare_models_pytorch() は META 引数を受け取るが、META リスト自体が NameError。

Fix: prepare_models_pytorch の前に MODELS_GROUP_META_1 の定義を復元 (リファレンス NB から)
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NB_PATH = Path(__file__).with_name("nb_pseudo_babych.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))


def set_code(i, text):
    nb["cells"][i]["cell_type"] = "code"
    nb["cells"][i]["source"] = text.splitlines(keepends=True)
    nb["cells"][i]["outputs"] = []
    nb["cells"][i]["execution_count"] = None


# Restore Cell 14 with MODELS_GROUP_META_1 definition + prepare_models_pytorch
set_code(14, '''# MODELS_GROUP_META_1: 7 Babych weight files + their ModelsGroupConfig refs
MODELS_GROUP_META_1 = [
    # iter 3 self-learning models (5 model)
    ("tf_efficientnet_b4.ns_jft_in1k_sampler_maxsum_iteration_3_v1_temp_0.55_64_bs_0.15_drop_path_rate_1_mixup_ratio_pseudo_data_20_duration_sed_type_0.5_mixup_p_(224, 512)_size_ce_4096_n_fft_0_fold_22_seed_25_epoch.pt", ModelsGroupConfig.model_config_6),
    ("tf_efficientnet_b3.ns_jft_in1k_sampler_maxsum_iteration_3_v1_temp_0.55_54_bs_0.15_drop_path_rate_1_mixup_ratio_pseudo_data_20_duration_sed_type_0.5_mixup_p_(224, 512)_size_ce_4096_n_fft_1_fold_25_epoch.pt", ModelsGroupConfig.model_config_1),
    ("regnety_016.tv2_in1k_sampler_maxsum_iteration_4_v1_temp_0.6_64_bs_0.15_drop_path_rate_1_mixup_ratio_pseudo_data_20_duration_sed_type_0.5_mixup_p_(224, 512)_size_ce_4096_n_fft_2_fold_25_epoch.pt", ModelsGroupConfig.model_config_2),
    ("regnety_016.tv2_in1k_sampler_maxsum_iteration_4_v1_framewise_temp_0.6_64_bs_0.15_drop_path_rate_1_mixup_ratio_pseudo_data_20_duration_sed_type_0.5_mixup_p_(224, 512)_size_ce_4096_n_fft_3_fold_25_epoch.pt", ModelsGroupConfig.model_config_2),
    ("eca_nfnet_l0.ra2_in1k_sampler_maxsum_iteration_3_v1_temp_0.55_128_bs_0.15_drop_path_rate_1_mixup_ratio_pseudo_data_20_duration_sed_type_0.5_mixup_p_(224, 512)_size_ce_4096_n_fft_additional_data_full_data_22_seed_15_epoch.pt", ModelsGroupConfig.model_config_5),
    # supervised + sub model (2 model)
    ("regnety_008.pycls_in1k_20_duration_sed_mixup_(224, 512)_size_ce_4096_n_fft_2_fold_15_epoch.pt", ModelsGroupConfig.model_config_3),
    ("tf_efficientnet_b0.ns_jft_in1k_incest_amphibia_128_bs_0.0_drop_path_rate_20_duration_sed_type_0.5_mixup_p_(224, 512)_size_ce_4096_n_fft_full_data_22_seed_15_epoch.pt", ModelsGroupConfig.model_config_4),
]
print(f"MODELS_GROUP_META_1: {len(MODELS_GROUP_META_1)} model entries")


def prepare_models_pytorch(model_meta_list, models_dir, device, dtype=torch.float16):
    """Load Babych weights as PyTorch models, push to CUDA + FP16. No OpenVINO."""
    models = []
    for weights_name, model_config in tqdm.tqdm(model_meta_list, desc="Loading Babych models"):
        # Build model architecture (uses cell 5 CLEFClassifierSED + cell 9 configs)
        model = CLEFClassifierSED(config=model_config)
        pt_path = os.path.join(models_dir, weights_name)
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Babych weight not found: {pt_path}")
        state = torch.load(pt_path, weights_only=True, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        model.to(device).to(dtype)
        models.append(model)
        print(f"  loaded {weights_name[:60]}... ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    return models


# Build & load all 7 Babych models
all_models = prepare_models_pytorch(MODELS_GROUP_META_1, BABYCH_DIR, DEVICE, dtype=torch.float16)
print(f"Loaded {len(all_models)} Babych models on {DEVICE} (FP16)")

# Reference mel spectrogram feature extractor from model 0 (all share same mel spec params)
mel_spec_generator = all_models[0].mel_spectr_generator
''')


NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"v4 META restore patched: {NB_PATH}")
