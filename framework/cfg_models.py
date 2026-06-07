"""M1-M7 model CFG dict 一元管理 (2026-05-24 framework 統一仕様)

Usage:
  from framework.cfg_models import M1, M2, M3, M4, M5, M6, M7, COMMON
  from framework.build_train_nb import build_nb
  build_nb(M3, output_path="experiment/exp072/notebook/nb_train.ipynb")

Reference:
  - memory feedback_train_logging_format.md: log format 仕様
  - memory feedback_framework_consistency_required.md: 統一性必須
  - memory project_babych_ensemble_plan.md: 7-model plan 確定版
  - memory project_m7_v2_result.md: M7 v2 完走、val_macro 0.93 確証
"""

# ─────────────────────────────────────
# COMMON: 全 model 共通 (M-R3 各 NB で同一値)
# ─────────────────────────────────────
COMMON = {
    # Sample rate / window
    "sr": 32_000,
    "n_windows": 12,                # 12 chunks per 60s test_soundscape
    "window_sec_inference": 5,      # inference 時 chunk

    # Loss / Optimizer 共通
    "loss_clip_weight": 0.5,
    "loss_frame_weight": 0.5,
    "warmup_epochs": 2,
    "lr_min": 1e-6,
    "weight_decay": 1e-4,

    # Augmentation 共通
    "spec_aug_freq": 10,
    "spec_aug_time": 10,

    # Validation
    "step_log_interval": 50,
    "ns22_k": 22,                   # lowest K species AUC for val_ns22

    # Seed
    "seed": 42,
}


# ─────────────────────────────────────
# M1: eca_nfnet_l1 + 5s + Perch ON + 5-fold (★ exp029 拡張)
# ─────────────────────────────────────
M1 = {
    **COMMON,
    "exp_id":           "exp070",
    "model_short":      "m1",
    "title":            "M1 nfnet_l1 5s Perch-ON R3 5-fold",
    # Architecture
    "backbone":         "eca_nfnet_l1",
    "backbone_pretrain": ".ra2_in1k",
    "in_chans":         1,
    "use_perch_distill": True,
    "perch_distill_lambda": 0.15,
    "perch_embed_dim":  1536,
    "drop_path_rate":   0.10,            # exp029 系継承
    # Mel
    "window_sec":       5,
    "n_mels":           256,
    "n_fft":            2048,
    "hop_length":       512,
    "fmin":             20,
    "fmax":             16000,
    "top_db":           80,
    "norm_mode":        "zscore",        # per-instance z-score (exp029 R3 流)
    # Init
    "init_source":      "imagenet",      # timm pretrained=True
    "babych_weight_pattern": None,
    # Pseudo source (★ M1: pseudo_exp048 直接使用)
    "pseudo_source":    "exp069c",        # exp069c output Kaggle Dataset
    "pseudo_file":      "pseudo_exp048_234.npz",
    "pseudo_power_k":   1.54,            # Babych iter2 spec
    # Training
    "n_folds":          5,
    "fold_ids_to_train": [0, 1, 2, 3, 4],
    "n_epochs":         20,
    "batch_size":       32,
    "lr":               5e-4,
    # Class config
    "n_classes":        234,
    "class_subset":     None,
    # MixUp
    "mixup_mode":       "beta",           # exp029 系: Beta(0.4, 0.4)
    "mixup_alpha":      0.4,
    "mixup_p":          0.5,
    "mixup_blend_fixed": None,
    # Output
    "kaggle_dataset_slug": "birdclef2026-exp070-m1-nfnet-l1-r3-5fold",
    # Platform
    "platform_preferred": "colab_g4",     # G4 (Pro+) で 5-fold 完走
    "compute_time_est_h": 10,
}


# ─────────────────────────────────────
# M2: eca_nfnet_l0 + 5s + Perch ON + 3-fold (Option I fresh restart)
# ─────────────────────────────────────
M2 = {
    **COMMON,
    "exp_id":           "exp071",
    "model_short":      "m2",
    "title":            "M2 nfnet_l0 5s Perch-ON R3 3-fold",
    "backbone":         "eca_nfnet_l0",
    "backbone_pretrain": ".ra2_in1k",
    "in_chans":         1,
    "use_perch_distill": True,
    "perch_distill_lambda": 0.15,
    "perch_embed_dim":  1536,
    "drop_path_rate":   0.10,
    "window_sec":       5,
    "n_mels":           256,
    "n_fft":            2048,
    "hop_length":       512,
    "fmin":             20,
    "fmax":             16000,
    "top_db":           80,
    "norm_mode":        "zscore",
    "init_source":      "imagenet",
    "babych_weight_pattern": None,
    "pseudo_source":    "exp069c",
    "pseudo_file":      "pseudo_exp048_234.npz",
    "pseudo_power_k":   1.54,
    "n_folds":          3,
    "fold_ids_to_train": [0, 1, 2],
    "n_epochs":         20,
    "batch_size":       32,
    "lr":               5e-4,
    "n_classes":        234,
    "class_subset":     None,
    "mixup_mode":       "beta",
    "mixup_alpha":      0.4,
    "mixup_p":          0.5,
    "mixup_blend_fixed": None,
    "kaggle_dataset_slug": "birdclef2026-exp071-m2-nfnet-l0-r3-3fold",
    "platform_preferred": "colab_l4",
    "compute_time_est_h": 12,
}


# ─────────────────────────────────────
# M3: eca_nfnet_l0 + 20s + Babych init + Perch OFF + 3-fold (★ mel軸 core)
# ─────────────────────────────────────
M3 = {
    **COMMON,
    "exp_id":           "exp072",
    "model_short":      "m3",
    "title":            "M3 nfnet_l0 20s Babych-init R3 3-fold",
    "backbone":         "eca_nfnet_l0",
    "backbone_pretrain": ".ra2_in1k",
    "in_chans":         3,                # Babych 流: 3-channel (mel × 3)
    "use_perch_distill": False,
    "perch_distill_lambda": 0.0,
    "perch_embed_dim":  None,
    "drop_path_rate":   0.15,            # ★ Babych R2+ noise spec
    "window_sec":       20,              # ★ Babych mel
    "n_mels":           224,             # ★ Babych spec
    "n_fft":            4096,            # ★
    "hop_length":       1252,            # ★
    "fmin":             0,               # ★ Babych
    "fmax":             16000,
    "top_db":           80,
    "norm_mode":        "min_max_0_1",    # ★ Babych default
    "init_source":      "babych",
    "babych_weight_pattern": "eca_nfnet_l0_*.pt",   # Babych iter3 ckpt
    "pseudo_source":    "exp069c",
    "pseudo_file":      "pseudo_hybrid_234.npz",   # ★ Hybrid (Babych overlap + exp048)
    "pseudo_power_k":   1.54,
    "n_folds":          3,
    "fold_ids_to_train": [0, 1, 2],
    "n_epochs":         20,
    "batch_size":       24,              # 20s mel で memory 上 batch 小
    "lr":               5e-4,
    "n_classes":        234,
    "class_subset":     None,
    "mixup_mode":       "fixed",          # ★ Babych: ratio 1.0 + blend 0.5
    "mixup_alpha":      None,
    "mixup_p":          1.0,             # 100% (Babych spec)
    "mixup_blend_fixed": 0.5,
    "kaggle_dataset_slug": "birdclef2026-exp072-m3-nfnet-l0-20s-babych-r3",
    "platform_preferred": "colab_l4",
    "compute_time_est_h": 24,
}


# ─────────────────────────────────────
# M4: tf_efficientnet_b3 + 20s + Babych init + Perch OFF + 1-fold
# ─────────────────────────────────────
M4 = {
    **COMMON,
    "exp_id":           "exp073",
    "model_short":      "m4",
    "title":            "M4 b3 20s Babych-init R3 1-fold",
    "backbone":         "tf_efficientnet_b3",
    "backbone_pretrain": ".ns_jft_in1k",
    "in_chans":         3,
    "use_perch_distill": False,
    "perch_distill_lambda": 0.0,
    "perch_embed_dim":  None,
    "drop_path_rate":   0.15,
    "window_sec":       20,
    "n_mels":           224,
    "n_fft":            4096,
    "hop_length":       1252,
    "fmin":             0,
    "fmax":             16000,
    "top_db":           80,
    "norm_mode":        "min_max_0_1",
    "init_source":      "babych",
    "babych_weight_pattern": "tf_efficientnet_b3*.pt",   # Babych iter3+ ckpt
    "pseudo_source":    "exp069c",
    "pseudo_file":      "pseudo_hybrid_234.npz",
    "pseudo_power_k":   1.54,
    "n_folds":          1,
    "fold_ids_to_train": [0],
    "n_epochs":         20,
    "batch_size":       32,
    "lr":               5e-4,
    "n_classes":        234,
    "class_subset":     None,
    "mixup_mode":       "fixed",
    "mixup_alpha":      None,
    "mixup_p":          1.0,
    "mixup_blend_fixed": 0.5,
    "kaggle_dataset_slug": "birdclef2026-exp073-m4-b3-20s-babych-r3",
    "platform_preferred": "kaggle_t4x2",
    "compute_time_est_h": 6,
}


# ─────────────────────────────────────
# M5' (ConvNeXt 20s ImageNet) - b4 から差し替え
# ─────────────────────────────────────
M5 = {
    **COMMON,
    "exp_id":           "exp074",
    "model_short":      "m5",
    "title":            "M5' ConvNeXt 20s R3 1-fold",
    "backbone":         "convnext_nano",                # convnext_pico 代替もあり
    "backbone_pretrain": ".in12k_ft_in1k",
    "in_chans":         3,
    "use_perch_distill": False,
    "perch_distill_lambda": 0.0,
    "perch_embed_dim":  None,
    "drop_path_rate":   0.15,
    "window_sec":       20,
    "n_mels":           224,
    "n_fft":            4096,
    "hop_length":       1252,
    "fmin":             0,
    "fmax":             16000,
    "top_db":           80,
    "norm_mode":        "min_max_0_1",
    "init_source":      "imagenet",       # ConvNeXt は Babych 不在
    "babych_weight_pattern": None,
    "pseudo_source":    "exp069c",
    "pseudo_file":      "pseudo_hybrid_234.npz",
    "pseudo_power_k":   1.54,
    "n_folds":          1,
    "fold_ids_to_train": [0],
    "n_epochs":         20,
    "batch_size":       32,
    "lr":               5e-4,
    "n_classes":        234,
    "class_subset":     None,
    "mixup_mode":       "fixed",
    "mixup_alpha":      None,
    "mixup_p":          1.0,
    "mixup_blend_fixed": 0.5,
    "kaggle_dataset_slug": "birdclef2026-exp074-m5-convnext-20s-r3",
    "platform_preferred": "kaggle_t4x2",
    "compute_time_est_h": 8,
}


# ─────────────────────────────────────
# M6: regnety_016 + 20s + Babych init + 1-fold
# ─────────────────────────────────────
M6 = {
    **COMMON,
    "exp_id":           "exp075",
    "model_short":      "m6",
    "title":            "M6 regnety_016 20s Babych-init R3 1-fold",
    "backbone":         "regnety_016",
    "backbone_pretrain": ".tv2_in1k",
    "in_chans":         3,
    "use_perch_distill": False,
    "perch_distill_lambda": 0.0,
    "perch_embed_dim":  None,
    "drop_path_rate":   0.15,
    "window_sec":       20,
    "n_mels":           224,
    "n_fft":            4096,
    "hop_length":       1252,
    "fmin":             0,
    "fmax":             16000,
    "top_db":           80,
    "norm_mode":        "min_max_0_1",
    "init_source":      "babych",
    "babych_weight_pattern": "regnety_016*.pt",   # Babych iter4 ckpt
    "pseudo_source":    "exp069c",
    "pseudo_file":      "pseudo_hybrid_234.npz",
    "pseudo_power_k":   1.54,
    "n_folds":          1,
    "fold_ids_to_train": [0],
    "n_epochs":         20,
    "batch_size":       32,
    "lr":               5e-4,
    "n_classes":        234,
    "class_subset":     None,
    "mixup_mode":       "fixed",
    "mixup_alpha":      None,
    "mixup_p":          1.0,
    "mixup_blend_fixed": 0.5,
    "kaggle_dataset_slug": "birdclef2026-exp075-m6-regnety-016-20s-babych-r3",
    "platform_preferred": "kaggle_t4x2",
    "compute_time_est_h": 6,
}


# ─────────────────────────────────────
# M7: tf_efficientnet_b0 sub + 20s + Babych init + 5-fold (★ DONE 2026-05-24)
# ─────────────────────────────────────
M7 = {
    **COMMON,
    "exp_id":           "exp076",
    "model_short":      "m7",
    "title":            "M7 b0 sub 20s Babych-init Stage1 5-fold",
    "backbone":         "tf_efficientnet_b0",
    "backbone_pretrain": ".ns_jft_in1k",
    "in_chans":         3,
    "use_perch_distill": False,
    "perch_distill_lambda": 0.0,
    "perch_embed_dim":  None,
    "drop_path_rate":   0.0,             # ★ Babych Stage 1: drop_path=0
    "window_sec":       20,
    "n_mels":           224,
    "n_fft":            4096,
    "hop_length":       1252,
    "fmin":             0,
    "fmax":             16000,
    "top_db":           80,
    "norm_mode":        "min_max_0_1",
    "init_source":      "babych",
    "babych_weight_pattern": "tf_efficientnet_b0_*incest_amphibia*.pt",
    "pseudo_source":    None,             # ★ M7: supervised only
    "pseudo_file":      None,
    "pseudo_power_k":   None,
    "n_folds":          5,                # ★ user 指定で 5
    "fold_ids_to_train": [0, 1, 2, 3, 4],
    "n_epochs":         40,                # ★ Babych b0 spec: 40 epoch
    "batch_size":       32,
    "lr":               5e-4,
    "n_classes":        63,                # ★ sub model: Amphibia 35 + Insecta 28
    "class_subset":     "amph_insec",
    "mixup_mode":       "beta",
    "mixup_alpha":      0.4,
    "mixup_p":          0.5,
    "mixup_blend_fixed": None,
    "kaggle_dataset_slug": "birdclef2026-exp076-m7-train",   # 既 push 済
    "platform_preferred": "kaggle_t4x2",
    "compute_time_est_h": 1,              # 実測 52 min
    # Status
    "status":           "completed",
    "result_val_macro_mean": 0.9310,
    "result_val_ns22_mean":  0.9048,
    "result_per_fold_ns22": {0: 0.8890, 1: 0.9032, 2: 0.9112, 3: 0.9283, 4: 0.8924},
}


# All models for iteration
ALL_MODELS = {
    "M1": M1, "M2": M2, "M3": M3, "M4": M4, "M5": M5, "M6": M6, "M7": M7
}


def print_summary():
    """全 model の CFG summary を表示。"""
    print(f"{'Model':<6} {'exp_id':<10} {'Backbone':<22} {'Mel':<6} {'Init':<10} {'Perch':<6} {'Fold':<6} {'Status'}")
    print("-" * 95)
    for name, cfg in ALL_MODELS.items():
        backbone = cfg['backbone']
        mel = f"{cfg['window_sec']}s"
        init = cfg['init_source']
        perch = "ON" if cfg['use_perch_distill'] else "OFF"
        fold = cfg['n_folds']
        status = cfg.get('status', 'pending')
        print(f"{name:<6} {cfg['exp_id']:<10} {backbone:<22} {mel:<6} {init:<10} {perch:<6} {fold:<6} {status}")


if __name__ == "__main__":
    print_summary()
