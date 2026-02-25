# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LADD Editing experiment config for FLUX.2-klein-4B.

This configuration sets up LADD distillation for image editing with the
FLUX.2-klein-4B model. It trains the model to perform edits in 4 steps
instead of the original 50 steps.

Usage:
    python train.py --config fastgen/configs/experiments/Flux2Klein/config_ladd_editing.py \
        --model.net.model_id="/path/to/merged-flux2-klein-4B" \
        --dataloader_train.datatags='["WDS:/path/to/editing_data"]'
"""

from fastgen.configs.discriminator import Discriminator_Flux2Klein_Config
import fastgen.configs.methods.config_ladd_editing as config_ladd_editing_default
from fastgen.configs.data import EditingLoaderConfig
from fastgen.configs.net import Flux2KleinBaseConfig


def create_config():
    """Create LADD Editing config for Flux2-Klein-4B."""
    config = config_ladd_editing_default.create_config()

    # ===== Network Configuration =====
    config.model.net = Flux2KleinBaseConfig
    # For custom merged LoRA model, override via CLI:
    # --model.net.model_id="/path/to/merged-flux2-klein-4B"

    # ===== Optimizer Configuration =====
    config.model.net_optimizer.lr = 1e-6
    config.model.discriminator_optimizer.lr = 2e-6

    # ===== Precision =====
    config.model.precision = "bfloat16"

    # ===== Input Shape =====
    # Flux2-Klein: 128 latent channels, no 2x2 packing
    # For 1024x1024 images: latent is [128, 128, 128]
    # For 512x512 images: latent is [128, 64, 64]
    config.model.input_shape = [128, 128, 128]  # [C, H, W] for 1024x1024

    # ===== Discriminator Configuration =====
    config.model.discriminator = Discriminator_Flux2Klein_Config
    # Feature extraction from blocks spread across the 25 total blocks
    # (5 joint + 20 single = 25 blocks)
    config.model.discriminator.feature_indices = [3, 10, 17, 24]

    # ===== GAN Configuration =====
    config.model.gan_r1_reg_weight = 0.1
    config.model.gan_r1_reg_alpha = 0.1
    config.model.gan_use_same_t_noise = True
    config.model.student_update_freq = 2

    # ===== Timestep Sampling =====
    config.model.sample_t_cfg.time_dist_type = "shifted"
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # ===== 4-Step Training =====
    config.model.student_sample_steps = 4
    config.model.sample_t_cfg.t_list = [0.999, 0.937, 0.833, 0.624, 0.0]

    # ===== Editing-Specific Configuration =====
    # Noise strength for source image
    # 0.7 = moderate denoising (good for anatomy corrections)
    # Increase for larger structural changes, decrease for subtle edits
    config.model.source_noise_strength = 0.7

    # ===== DataLoader Configuration =====
    config.dataloader_train = EditingLoaderConfig
    config.dataloader_train.batch_size = 2  # Adjust based on GPU memory
    config.dataloader_train.input_res = 1024  # Match input_shape

    # ===== Logging =====
    config.log_config.group = "flux2klein_ladd_editing"

    return config


# Convenience configs for different resolutions

def create_config_512():
    """Create config for 512x512 resolution (lower memory)."""
    config = create_config()
    config.model.input_shape = [128, 64, 64]  # [C, H, W] for 512x512
    config.dataloader_train.input_res = 512
    config.dataloader_train.batch_size = 4  # Can fit more at lower res
    config.log_config.group = "flux2klein_ladd_editing_512"
    return config


def create_config_768():
    """Create config for 768x768 resolution."""
    config = create_config()
    config.model.input_shape = [128, 96, 96]  # [C, H, W] for 768x768
    config.dataloader_train.input_res = 768
    config.dataloader_train.batch_size = 2
    config.log_config.group = "flux2klein_ladd_editing_768"
    return config
