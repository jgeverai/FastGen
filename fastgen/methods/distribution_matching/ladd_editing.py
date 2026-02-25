# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LADD (Latent Adversarial Diffusion Distillation) extended for image editing.

This module extends the base LADD model to support image-to-image editing distillation.
Instead of generating from pure noise (text-to-image), it generates from a source image
with added noise, learning to edit the source image based on text instructions.

Key differences from standard LADD:
- Training data: (source_image, target_image, instruction) triplets
- Student input: source_image + noise (not pure noise)
- Conditioning: text instruction + source image reference
"""

from __future__ import annotations

from functools import partial
from typing import Dict, Any, TYPE_CHECKING, Callable, Optional, Tuple
import torch

from fastgen.methods.distribution_matching.ladd import LADDModel
from fastgen.methods.common_loss import (
    gan_loss_generator,
    gan_loss_discriminator,
)
from fastgen.utils.basic_utils import convert_cfg_to_dict

if TYPE_CHECKING:
    from fastgen.configs.methods.config_ladd_editing import ModelConfig


class LADDEditingModel(LADDModel):
    """LADD extended for image-to-image editing distillation.

    This model distills an image editing model (like Flux2-Klein with editing LoRA)
    into a few-step model that can perform edits in 4 steps instead of 50.

    The key difference from text-to-image LADD is that:
    1. Input to student is source_image + noise (not pure noise)
    2. Training data consists of (source, target, instruction) triplets
    3. The discriminator compares edited outputs to ground truth edited images

    Args:
        config: Model configuration including:
            - source_noise_strength: Strength of noise added to source image (0.0-1.0)
            - All standard LADD config parameters
    """

    def __init__(self, config: "ModelConfig"):
        super().__init__(config)
        self.config = config

        # Editing-specific parameters
        self.source_noise_strength = getattr(config, "source_noise_strength", 0.7)

    def _prepare_training_data(
        self, data: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Any, torch.Tensor]:
        """Prepare training data from triplets.

        Args:
            data: Dictionary containing:
                - "source": Source images to edit [B, C, H, W]
                - "target": Ground truth edited images [B, C, H, W]
                - "instruction" or "text": Edit instruction text

        Returns:
            Tuple of (target_latents, condition, source_latents)
        """
        # Get source and target images
        source_images = data["source"].to(self.device, dtype=self.precision)
        target_images = data["target"].to(self.device, dtype=self.precision)

        # Get text instruction
        if "instruction" in data:
            text = data["instruction"]
        elif "text" in data:
            text = data["text"]
        else:
            raise KeyError("Data must contain 'instruction' or 'text' key")

        # Encode images to latent space
        with torch.no_grad():
            source_latents = self.net.vae.encode(source_images)
            target_latents = self.net.vae.encode(target_images)

        # Encode text condition
        with torch.no_grad():
            condition = self.net.text_encoder.encode(text, precision=self.precision)
            # Move condition tensors to device
            condition = tuple(c.to(self.device) for c in condition)

        return target_latents, condition, source_latents

    def _generate_noise_and_time(
        self,
        target_data: torch.Tensor,
        source_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate noise and timesteps for editing.

        For image editing, the student starts from a noisy version of the source image,
        not from pure noise. The noise level is controlled by source_noise_strength.

        Args:
            target_data: Target (edited) latents for dtype/device reference
            source_data: Source latents to add noise to

        Returns:
            input_student: Noisy source image as input to student
            t_student: Timestep for student input
            t: Timestep for discriminator
            eps: Noise used in forward process
        """
        batch_size = target_data.shape[0]
        eps_student = torch.randn_like(source_data)

        if self.config.student_sample_steps == 1:
            # Single-step distillation: use high noise level on source
            # The student learns to denoise from highly corrupted source → edited
            t_student = torch.full(
                (batch_size,),
                self.source_noise_strength * self.net.noise_scheduler.max_t,
                device=self.device,
                dtype=self.net.noise_scheduler.t_precision,
            )
            # Forward process: mix source with noise
            input_student = self.net.noise_scheduler.forward_process(
                source_data, eps_student, t_student
            )
        else:
            # Multi-step distillation: sample timesteps from a list
            t_student = self.net.noise_scheduler.sample_from_t_list(
                batch_size,
                sample_steps=self.config.student_sample_steps,
                t_list=self.config.sample_t_cfg.t_list,
                device=self.device,
            )
            # Apply noise to source image based on sampled timestep
            input_student = self.net.noise_scheduler.forward_process(
                source_data, eps_student, t_student
            )

        # Generate timestep and noise for discriminator (same as base LADD)
        t = self.net.noise_scheduler.sample_t(
            batch_size, **convert_cfg_to_dict(self.config.sample_t_cfg), device=self.device
        )
        eps = torch.randn_like(target_data)

        return input_student, t_student, t, eps

    def _student_update_step(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        data: Dict[str, Any],
        condition: Optional[Any] = None,
        source_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor | Callable]]:
        """Perform student model update step for editing.

        The student takes noisy_source → predicts edited image.

        Args:
            input_student: Noisy source image input
            t_student: Input timestep
            t: Discriminator timestep
            eps: Noise for forward process
            data: Original data batch
            condition: Text conditioning
            source_latents: Source image latents (for multi-step generation)

        Returns:
            Tuple of (loss_map, outputs)
        """
        # Verify student network is in training mode with gradients enabled
        assert self.net.training, "Student network should be in training mode"
        net_params = list(self.net.parameters())
        assert len(net_params) > 0, "Student network has no parameters"
        assert net_params[0].requires_grad, "Student network parameters should require grad"

        # Generate edited data from student
        gen_data = self.gen_data_from_net(input_student, t_student, condition=condition)

        # Verify generated data has gradients (through student network)
        assert gen_data.requires_grad, \
            f"gen_data requires_grad={gen_data.requires_grad}, grad_fn={gen_data.grad_fn}"

        # Perturb generated data for discriminator
        perturbed_data = self.net.noise_scheduler.forward_process(gen_data, eps, t)

        # Extract features from teacher network (no_grad for teacher, but we need grad through gen_data)
        fake_feat = self.teacher(
            perturbed_data,
            t,
            condition=condition,
            return_features_early=True,
            feature_indices=self.discriminator.feature_indices,
        )

        # Verify features are lists with expected number of elements
        assert isinstance(fake_feat, list), f"fake_feat should be list, got {type(fake_feat)}"
        assert len(fake_feat) == len(self.discriminator.feature_indices), \
            f"fake_feat length {len(fake_feat)} != feature_indices {len(self.discriminator.feature_indices)}"

        # Compute GAN loss for generator
        fake_feat_logit = self.discriminator(fake_feat)
        gan_loss_gen = gan_loss_generator(fake_feat_logit)

        # Verify loss has gradients
        assert gan_loss_gen.requires_grad, \
            f"gan_loss_gen requires_grad={gan_loss_gen.requires_grad}, grad_fn={gan_loss_gen.grad_fn}"

        # Build output dictionaries
        loss_map = {
            "total_loss": gan_loss_gen,
            "gan_loss_gen": gan_loss_gen,
        }
        outputs = self._get_outputs(
            gen_data, input_student, condition=condition, source_latents=source_latents
        )

        return loss_map, outputs

    def _discriminator_update_step(
        self,
        input_student: torch.Tensor,
        t_student: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        target_data: torch.Tensor,
        condition: Optional[Any] = None,
        source_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Perform discriminator update step for editing.

        The discriminator learns to distinguish between:
        - Real: perturbed target (ground truth edited) images
        - Fake: perturbed student-generated edits

        Args:
            input_student: Noisy source image input
            t_student: Input timestep
            t: Discriminator timestep
            eps: Noise for forward process
            target_data: Target (edited) latents
            condition: Text conditioning
            source_latents: Source image latents

        Returns:
            Tuple of (loss_map, outputs)
        """
        # Verify discriminator is in training mode with gradients enabled
        assert self.discriminator.training, "Discriminator should be in training mode"
        disc_params = list(self.discriminator.parameters())
        assert len(disc_params) > 0, "Discriminator has no parameters"
        assert disc_params[0].requires_grad, "Discriminator parameters should require grad"
        # Verify device consistency
        assert disc_params[0].device == target_data.device, \
            f"Discriminator device {disc_params[0].device} != data device {target_data.device}"

        with torch.no_grad():
            # Generate edited data from student
            gen_data = self.gen_data_from_net(input_student, t_student, condition=condition)
            x_t_sg = self.net.noise_scheduler.forward_process(gen_data, eps, t)

            # Extract fake features from teacher
            fake_feat = self.teacher(
                x_t_sg,
                t,
                condition=condition,
                return_features_early=True,
                feature_indices=self.discriminator.feature_indices,
            )

            # Compute real features from target (edited) images
            real_feat, t_real = self._compute_real_feat(
                real_data=target_data, t=t, eps=eps, condition=condition
            )

        # Verify features are lists with expected number of elements
        assert isinstance(real_feat, list), f"real_feat should be list, got {type(real_feat)}"
        assert isinstance(fake_feat, list), f"fake_feat should be list, got {type(fake_feat)}"
        assert len(real_feat) == len(self.discriminator.feature_indices), \
            f"real_feat length {len(real_feat)} != feature_indices {len(self.discriminator.feature_indices)}"

        # Compute discriminator loss
        real_feat_logit = self.discriminator(real_feat)
        fake_feat_logit = self.discriminator(fake_feat)

        # Verify discriminator outputs have gradients
        assert real_feat_logit.requires_grad, \
            f"real_feat_logit requires_grad={real_feat_logit.requires_grad}, grad_fn={real_feat_logit.grad_fn}"
        assert fake_feat_logit.requires_grad, \
            f"fake_feat_logit requires_grad={fake_feat_logit.requires_grad}, grad_fn={fake_feat_logit.grad_fn}"

        gan_loss_disc = gan_loss_discriminator(real_feat_logit, fake_feat_logit)

        # Verify GAN loss has gradients
        assert gan_loss_disc.requires_grad, \
            f"gan_loss_disc requires_grad={gan_loss_disc.requires_grad}, grad_fn={gan_loss_disc.grad_fn}"

        # R1 regularization (optional)
        gan_loss_ar1 = torch.zeros_like(gan_loss_disc)
        if self.config.gan_r1_reg_weight > 0:
            gan_loss_ar1 = self._compute_r1_regularization(
                real_feat_logit, target_data, t_real, condition=condition
            )
            # Verify R1 loss has gradients
            assert gan_loss_ar1.requires_grad, \
                f"gan_loss_ar1 requires_grad={gan_loss_ar1.requires_grad}, grad_fn={gan_loss_ar1.grad_fn}"

        total_loss = gan_loss_disc + self.config.gan_r1_reg_weight * gan_loss_ar1

        # Verify total loss has gradients
        assert total_loss.requires_grad, \
            f"total_loss requires_grad={total_loss.requires_grad}, grad_fn={total_loss.grad_fn}"

        loss_map = {
            "gan_loss_disc": gan_loss_disc,
            "total_loss": total_loss,
        }
        if self.config.gan_r1_reg_weight > 0:
            loss_map.update({"gan_loss_ar1": gan_loss_ar1})

        outputs = self._get_outputs(
            gen_data, input_student, condition=condition, source_latents=source_latents
        )

        return loss_map, outputs

    def _get_outputs(
        self,
        gen_data: torch.Tensor,
        input_student: torch.Tensor = None,
        condition: Any = None,
        source_latents: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | Callable]:
        """Get outputs for logging and visualization.

        For editing, we also include the source image for comparison.
        """
        if self.config.student_sample_steps == 1:
            assert input_student is not None, "input_student must not be None"
            # For single-step, input_student is the noisy source
            return {
                "gen_rand": gen_data,
                "input_rand": input_student,
                "source": source_latents,
            }
        else:
            noise = torch.randn_like(gen_data, dtype=self.precision)
            gen_rand_func = partial(
                self.generator_fn,
                net=self.net_inference,
                noise=noise,
                condition=condition,
                student_sample_steps=self.config.student_sample_steps,
                student_sample_type=self.config.student_sample_type,
                t_list=self.config.sample_t_cfg.t_list,
                precision_amp=self.precision_amp_infer,
            )
            return {
                "gen_rand": gen_rand_func,
                "input_rand": noise,
                "gen_rand_train": gen_data,
                "source": source_latents,
            }

    def single_train_step(
        self, data: Dict[str, Any], iteration: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor | Callable]]:
        """Single training step for LADD Editing.

        Orchestrates the alternating student/discriminator updates for
        image-to-image editing distillation.

        Args:
            data: Data dict containing source, target, and instruction.
            iteration: Current training iteration.

        Returns:
            loss_map: Dictionary containing loss values.
            outputs: Dictionary containing network outputs.
        """
        # Prepare training data (returns target_latents, condition, source_latents)
        target_latents, condition, source_latents = self._prepare_training_data(data)

        # Set up gradient requirements based on training phase
        self._setup_grad_requirements(iteration)

        # Generate noise and timesteps for editing
        input_student, t_student, t, eps = self._generate_noise_and_time(
            target_latents, source_latents
        )

        # Choose between student update or discriminator update
        if iteration % self.config.student_update_freq == 0:
            return self._student_update_step(
                input_student,
                t_student,
                t,
                eps,
                data,
                condition=condition,
                source_latents=source_latents,
            )
        else:
            return self._discriminator_update_step(
                input_student,
                t_student,
                t,
                eps,
                target_latents,
                condition=condition,
                source_latents=source_latents,
            )

    def gen_data_from_net(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Any] = None,
    ) -> torch.Tensor:
        """Generate edited data from the student network.

        For single-step: directly predict the edited image from noisy source.
        For multi-step: iteratively denoise to produce the edited image.

        Args:
            x_t: Input (noisy source image)
            t: Timestep
            condition: Text conditioning

        Returns:
            Generated edited image latents
        """
        if self.config.student_sample_steps == 1:
            # Single-step generation with autocast
            with self.autocast():
                pred = self.net(x_t, t, condition=condition, fwd_pred_type="x0")
            return pred
        else:
            # Multi-step generation
            return self.generator_fn(
                net=self.net,
                noise=x_t,  # Start from noisy source
                condition=condition,
                student_sample_steps=self.config.student_sample_steps,
                student_sample_type=self.config.student_sample_type,
                t_list=self.config.sample_t_cfg.t_list,
                precision_amp=self.precision_amp,
            )
