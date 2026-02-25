# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration for LADD Editing model - image-to-image distillation.

This extends the base LADD config to support image editing distillation,
where the model learns to transform source images to edited targets
based on text instructions.
"""

import copy
import attrs
from omegaconf import DictConfig

from fastgen.utils import LazyCall as L
from fastgen.configs.config import (
    BaseModelConfig,
    BaseConfig,
)
from fastgen.configs.opt import BaseOptimizerConfig, BaseSchedulerConfig
from fastgen.methods.distribution_matching.ladd_editing import LADDEditingModel
from fastgen.configs.discriminator import Discriminator_Flux2Klein_Config
from fastgen.configs.callbacks import (
    WANDB_CALLBACK,
    GradClip_CALLBACK,
    GPUStats_CALLBACK,
    TrainProfiler_CALLBACK,
    ParamCount_CALLBACK,
)


@attrs.define(slots=False)
class ModelConfig(BaseModelConfig):
    """Model configuration for LADD Editing.

    Extends BaseModelConfig with editing-specific parameters.
    """

    # Discriminator configuration
    discriminator: DictConfig = attrs.field(factory=lambda: copy.deepcopy(Discriminator_Flux2Klein_Config))

    # Optimizer and scheduler for the discriminator
    discriminator_optimizer: DictConfig = attrs.field(factory=lambda: copy.deepcopy(BaseOptimizerConfig))
    discriminator_scheduler: DictConfig = attrs.field(factory=lambda: copy.deepcopy(BaseSchedulerConfig))

    # Student update frequency (1 student update per N discriminator updates)
    student_update_freq: int = 2

    # Use the same t and noise to perturb the real data and fake data
    gan_use_same_t_noise: bool = False

    # R1 regularization weight (0 means no R1 reg)
    gan_r1_reg_weight: float = 0.1
    # R1 regularization noise scale
    gan_r1_reg_alpha: float = 0.1

    # ===== Editing-specific parameters =====

    # Noise strength for source image (0.0-1.0)
    # Higher = more denoising required, better for large edits
    # Lower = preserves more source structure, better for subtle edits
    source_noise_strength: float = 0.7


@attrs.define(slots=False)
class Config(BaseConfig):
    """Top-level config for LADD Editing."""

    model: ModelConfig = attrs.field(factory=ModelConfig)
    model_class: DictConfig = L(LADDEditingModel)(
        config=None,
    )


def create_config():
    """Create default LADD Editing configuration."""
    config = Config()
    config.trainer.callbacks = DictConfig(
        {
            **GradClip_CALLBACK,
            **GPUStats_CALLBACK,
            **TrainProfiler_CALLBACK,
            **ParamCount_CALLBACK,
            **WANDB_CALLBACK,
        }
    )

    # Smaller batch size for editing (larger images, more memory)
    config.dataloader_train.batch_size = 2

    # Warm-up steps
    config.model.discriminator_scheduler.warm_up_steps = [0]
    config.model.net_scheduler.warm_up_steps = [0]

    return config
