# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Flux2-Klein network implementation for FastGen.

FLUX.2-klein-4B is a 4B parameter rectified flow transformer for text-to-image
and image editing. Key differences from FLUX.1-dev:
- 5 joint blocks + 20 single blocks = 25 total (vs 57 in FLUX.1-dev)
- 128 latent channels (vs 16)
- patch_size=1 (no 2x2 packing)
- No embedded guidance (guidance_embeds=False)
- joint_attention_dim=7680

References:
- https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B
- https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
"""

import os
from typing import Any, Optional, List, Set, Union, Tuple
import types

import torch
import torch.utils.checkpoint
from torch import dtype
from torch.distributed.fsdp import fully_shard

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import Flux2Transformer2DModel, AutoencoderKLFlux2
from transformers import AutoTokenizer, AutoModelForCausalLM

from fastgen.networks.network import FastGenNetwork
from fastgen.networks.noise_schedule import NET_PRED_TYPES
from fastgen.utils.basic_utils import str2bool
from fastgen.utils.distributed.fsdp import apply_fsdp_checkpointing
import fastgen.utils.logging_utils as logger


class Flux2KleinTextEncoder:
    """Text encoder for Flux2-Klein using diffusers pipeline.

    Flux2-Klein uses Qwen3ForCausalLM with a specific encoding strategy:
    - Chat template formatting
    - Multi-layer hidden state extraction (layers 9, 18, 27)
    - Stacked embeddings

    We leverage diffusers' Flux2KleinPipeline for correct encoding.

    Note: Unlike CLIP+T5 based Flux.1, Flux2-Klein doesn't have native pooled
    embeddings. We create a mean-pooled representation for compatibility with
    the transformer's CombinedTimestepTextProjEmbeddings layer.
    """

    def __init__(self, model_id: str):
        from diffusers import Flux2KleinPipeline

        # Load the full pipeline to get correct encoding
        self._pipeline = Flux2KleinPipeline.from_pretrained(
            model_id,
            cache_dir=os.environ.get("HF_HOME"),
            torch_dtype=torch.bfloat16,
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )

        # Extract components we need
        self.tokenizer = self._pipeline.tokenizer
        self.text_encoder = self._pipeline.text_encoder
        self.text_encoder.eval().requires_grad_(False)

        # Store reference to pipeline's encoding method
        self._encode_prompt = self._pipeline._get_qwen3_prompt_embeds

        # Get the expected pooled projection dimension from the transformer config
        # Flux2-Klein uses 3072 for time_text_embed (24 heads * 128 dim)
        self._pooled_projection_dim = 3072

    def encode(
        self,
        conditioning: Optional[Any] = None,
        precision: dtype = torch.float32,
        max_sequence_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompts to embeddings.

        Args:
            conditioning: Text prompt(s) to encode.
            precision: Data type for the output embeddings.
            max_sequence_length: Maximum sequence length for tokenization.

        Returns:
            Tuple of (pooled_prompt_embeds, prompt_embeds) for API compatibility.
            pooled_prompt_embeds: Mean-pooled text embeddings projected to pooled_dim.
            prompt_embeds: Full sequence text embeddings.
        """
        if isinstance(conditioning, str):
            conditioning = [conditioning]

        with torch.no_grad():
            # Use diffusers' encoding method which handles:
            # - Chat template formatting
            # - Multi-layer extraction (layers 9, 18, 27)
            # - Proper stacking of hidden states
            prompt_embeds = self._encode_prompt(
                prompt=conditioning,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                max_sequence_length=max_sequence_length,
                device=self.text_encoder.device,
            )
            prompt_embeds = prompt_embeds.to(precision)

            # Create pooled representation via mean pooling
            # prompt_embeds shape: [B, seq_len, hidden_dim]
            # We mean-pool over sequence length to get [B, hidden_dim]
            pooled_prompt_embeds = prompt_embeds.mean(dim=1)

            # Project to expected pooled dimension if needed
            # The transformer's time_text_embed expects pooled_projection_dim
            if pooled_prompt_embeds.shape[-1] != self._pooled_projection_dim:
                # Create a simple linear projection (lazy initialization)
                if not hasattr(self, "_pooled_proj"):
                    self._pooled_proj = torch.nn.Linear(
                        pooled_prompt_embeds.shape[-1],
                        self._pooled_projection_dim,
                        bias=False,
                    ).to(pooled_prompt_embeds.device, pooled_prompt_embeds.dtype)
                    # Initialize with identity-like projection
                    torch.nn.init.eye_(self._pooled_proj.weight[:, : pooled_prompt_embeds.shape[-1]])
                pooled_prompt_embeds = self._pooled_proj(pooled_prompt_embeds)

        return pooled_prompt_embeds, prompt_embeds

    def to(self, *args, **kwargs):
        """Moves the model to the specified device."""
        self.text_encoder.to(*args, **kwargs)
        if hasattr(self, "_pooled_proj"):
            self._pooled_proj.to(*args, **kwargs)
        return self


class Flux2KleinImageEncoder:
    """VAE encoder/decoder for Flux2-Klein.

    Flux2-Klein uses AutoencoderKLFlux2 with 32 VAE latent channels.
    After encoding, latents are patchified (2x2) to 128 channels for the transformer:
    - VAE output: [B, 32, H, W]
    - After patchify: [B, 128, H/2, W/2]
    """

    def __init__(self, model_id: str):
        self.vae: AutoencoderKLFlux2 = AutoencoderKLFlux2.from_pretrained(
            model_id,
            cache_dir=os.environ.get("HF_HOME"),
            subfolder="vae",
            local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
        )
        self.vae.eval().requires_grad_(False)

        # Get VAE scaling factors from config
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.3611)
        self.shift_factor = getattr(self.vae.config, "shift_factor", 0.1159)

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """Patchify latents: pack 2x2 spatial patches into channels.

        Args:
            latents: [B, C, H, W] where C=32 (VAE channels)

        Returns:
            [B, C*4, H/2, W/2] where C*4=128 (transformer channels)
        """
        batch_size, num_channels, height, width = latents.shape
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """Unpatchify latents: unpack channels back to 2x2 spatial patches.

        Args:
            latents: [B, C, H, W] where C=128 (transformer channels)

        Returns:
            [B, C/4, H*2, W*2] where C/4=32 (VAE channels)
        """
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
        return latents

    def encode(self, real_images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space and patchify for transformer.

        Args:
            real_images: Input images in [-1, 1] range, shape [B, 3, H, W].

        Returns:
            torch.Tensor: Patchified latents [B, 128, H/16, W/16] for 1024x1024 input.
        """
        # VAE encode: [B, 3, H, W] -> [B, 32, H/8, W/8]
        latent_images = self.vae.encode(real_images, return_dict=False)[0].sample()
        # Apply shift and scale
        latent_images = (latent_images - self.shift_factor) * self.scaling_factor
        # Patchify: [B, 32, H/8, W/8] -> [B, 128, H/16, W/16]
        latent_images = self._patchify_latents(latent_images)
        return latent_images

    def decode(self, latent_images: torch.Tensor) -> torch.Tensor:
        """Decode patchified latents to images.

        Args:
            latent_images: Patchified latent representations [B, 128, H/16, W/16].

        Returns:
            torch.Tensor: Decoded images in [-1, 1] range.
        """
        # Unpatchify: [B, 128, H/16, W/16] -> [B, 32, H/8, W/8]
        latents = self._unpatchify_latents(latent_images)
        # Reverse shift and scale
        latents = (latents / self.scaling_factor) + self.shift_factor
        # VAE decode: [B, 32, H/8, W/8] -> [B, 3, H, W]
        images = self.vae.decode(latents, return_dict=False)[0].clip(-1.0, 1.0)
        return images

    def to(self, *args, **kwargs):
        """Moves the model to the specified device."""
        self.vae.to(*args, **kwargs)
        return self


def classify_forward_flux2(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[dict] = None,
    return_features_early: bool = False,
    feature_indices: Optional[Set[int]] = None,
    return_logvar: bool = False,
) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Modified forward pass for Flux2Transformer2DModel with feature extraction support.

    Note: Flux2-Klein has patch_size=1, so hidden_states shape is [B, H*W, C]
    where C=128 (latent channels).

    Args:
        hidden_states: Input latent states [B, H*W, C].
        encoder_hidden_states: T5 text encoder hidden states.
        pooled_projections: CLIP pooled text embeddings.
        timestep: Current timestep.
        img_ids: Image position IDs.
        txt_ids: Text position IDs.
        guidance: Guidance scale (unused for Flux2-Klein, kept for API compatibility).
        joint_attention_kwargs: Additional attention kwargs.
        return_features_early: If True, return features as soon as collected.
        feature_indices: Set of block indices to extract features from.
        return_logvar: If True, return log variance estimate.

    Returns:
        Model output, optionally with features or logvar.
    """
    if feature_indices is None:
        feature_indices = set()

    if return_features_early and len(feature_indices) == 0:
        return []

    idx, features = 0, []

    # Store original sequence length to compute spatial dims for feature reshaping
    # hidden_states: [B, H*W, C] where H*W is the sequence length
    seq_len = hidden_states.shape[1]
    spatial_size = int(seq_len**0.5)  # Assuming square spatial dimensions

    # 1. Patch embedding (no packing for Flux2-Klein)
    hidden_states = self.x_embedder(hidden_states)

    # 2. Time embedding (no guidance embedding for Flux2-Klein)
    timestep_scaled = timestep.to(hidden_states.dtype) * 1000
    temb = self.time_text_embed(timestep_scaled, pooled_projections)

    # 3. Text embedding
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # 4. Prepare positional embeddings
    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    # 5. Joint transformer blocks (5 blocks for Flux2-Klein)
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                joint_attention_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # Check if we should extract features at this index
        if idx in feature_indices:
            # Reshape from [B, seq_len, hidden_dim] to [B, hidden_dim, H, W] for discriminator
            feat = hidden_states.clone()
            B, S, C = feat.shape
            feat = feat.permute(0, 2, 1).reshape(B, C, spatial_size, spatial_size)
            features.append(feat)

        # Early return if we have all features
        if return_features_early and len(features) == len(feature_indices):
            return features

        idx += 1

    # 6. Single transformer blocks (20 blocks for Flux2-Klein)
    for block in self.single_transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                joint_attention_kwargs,
            )
        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # Check if we should extract features at this index
        if idx in feature_indices:
            # Reshape from [B, seq_len, hidden_dim] to [B, hidden_dim, H, W] for discriminator
            feat = hidden_states.clone()
            B, S, C = feat.shape
            feat = feat.permute(0, 2, 1).reshape(B, C, spatial_size, spatial_size)
            features.append(feat)

        # Early return if we have all features
        if return_features_early and len(features) == len(feature_indices):
            return features

        idx += 1

    # 7. Final projection
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    # If we have all the features, we can exit early
    if return_features_early:
        assert len(features) == len(feature_indices), f"{len(features)} != {len(feature_indices)}"
        return features

    # Prepare output
    if len(feature_indices) == 0:
        out = output
    else:
        out = [output, features]

    if return_logvar:
        logvar = self.logvar_linear(temb)
        return out, logvar

    return out


class Flux2Klein(FastGenNetwork):
    """Flux2-Klein network for text-to-image generation and image editing.

    FLUX.2-klein-4B is a 4B parameter model optimized for fast inference.
    Key characteristics:
    - 5 joint transformer blocks + 20 single transformer blocks
    - 128 latent channels (vs 16 in FLUX.1-dev)
    - No 2x2 patch packing (patch_size=1)
    - No embedded guidance
    - Hidden dimension: 3072

    References:
        - https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B (50-step base)
        - https://huggingface.co/black-forest-labs/FLUX.2-klein-4B (4-step distilled)
    """

    MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
    BASE_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        net_pred_type: str = "flow",
        schedule_type: str = "rf",
        disable_grad_ckpt: bool = False,
        guidance_scale: Optional[float] = None,  # Flux2-Klein doesn't use embedded guidance
        load_pretrained: bool = True,
        **model_kwargs,
    ):
        """Flux2-Klein constructor.

        Args:
            model_id: The HuggingFace model ID to load.
                Defaults to "black-forest-labs/FLUX.2-klein-4B" (distilled).
                Use "black-forest-labs/FLUX.2-klein-base-4B" for the base model.
            net_pred_type: Prediction type. Defaults to "flow" for flow matching.
            schedule_type: Schedule type. Defaults to "rf" (rectified flow).
            disable_grad_ckpt: Whether to disable gradient checkpointing during training.
                Defaults to False. Set to True when using FSDP to avoid memory access errors.
            guidance_scale: Guidance scale. Not used for Flux2-Klein (no embedded guidance).
            load_pretrained: Whether to load pretrained weights.
        """
        super().__init__(net_pred_type=net_pred_type, schedule_type=schedule_type, **model_kwargs)

        self.model_id = model_id
        self.guidance_scale = guidance_scale  # Kept for API compatibility
        self._disable_grad_ckpt = disable_grad_ckpt

        # Initialize the network (handles meta device and pretrained loading)
        self._initialize_network(model_id, load_pretrained)

        # Override forward with classify_forward for feature extraction
        self.transformer.forward = types.MethodType(classify_forward_flux2, self.transformer)

        # Disable cuDNN SDPA backend to avoid mha_graph->execute errors during backward
        if torch.backends.cuda.is_built():
            torch.backends.cuda.enable_cudnn_sdp(False)
            logger.info("Disabled cuDNN SDPA backend for Flux2-Klein compatibility")

        # Gradient checkpointing configuration
        if disable_grad_ckpt:
            self.transformer.disable_gradient_checkpointing()
        else:
            self.transformer.enable_gradient_checkpointing()

        torch.cuda.empty_cache()

    def _initialize_network(self, model_id: str, load_pretrained: bool) -> None:
        """Initialize the transformer network.

        Args:
            model_id: The HuggingFace model ID or local path.
            load_pretrained: Whether to load pretrained weights.
        """
        # Check if we're in a meta context (for FSDP memory-efficient loading)
        in_meta_context = self._is_in_meta_context()
        should_load_weights = load_pretrained and (not in_meta_context)

        if should_load_weights:
            logger.info("Loading Flux2-Klein transformer from pretrained")
            self.transformer: Flux2Transformer2DModel = Flux2Transformer2DModel.from_pretrained(
                model_id,
                cache_dir=os.environ["HF_HOME"],
                subfolder="transformer",
                local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            )
        else:
            # Load config and create model structure
            config = Flux2Transformer2DModel.load_config(
                model_id,
                cache_dir=os.environ["HF_HOME"],
                subfolder="transformer",
                local_files_only=str2bool(os.getenv("LOCAL_FILES_ONLY", "false")),
            )
            if in_meta_context:
                logger.info(
                    "Initializing Flux2-Klein transformer on meta device (zero memory, will receive weights via FSDP sync)"
                )
            else:
                logger.info("Initializing Flux2-Klein transformer from config (no pretrained weights)")
                logger.warning("Flux2-Klein transformer being initialized from config. No weights are loaded!")
            self.transformer: Flux2Transformer2DModel = Flux2Transformer2DModel.from_config(config)

        # Add logvar linear layer for variance estimation
        # Flux2-Klein uses 3072-dim time embeddings (24 heads * 128 dim)
        self.transformer.logvar_linear = torch.nn.Linear(3072, 1)

    def reset_parameters(self):
        """Reinitialize parameters for FSDP meta device initialization."""
        import torch.nn as nn

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        super().reset_parameters()
        logger.debug("Reinitialized Flux2-Klein parameters")

    def fully_shard(self, **kwargs):
        """Fully shard the Flux2-Klein network for FSDP.

        Flux2-Klein has:
        - transformer_blocks: 5 joint attention blocks
        - single_transformer_blocks: 20 single stream blocks
        """
        # Apply checkpointing first for proper casting during backward pass
        if self.transformer.gradient_checkpointing:
            self.transformer.disable_gradient_checkpointing()
            # Import block types - these may differ in Flux2
            from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock

            apply_fsdp_checkpointing(
                self.transformer,
                check_fn=lambda block: isinstance(block, (FluxTransformerBlock, FluxSingleTransformerBlock)),
            )
            logger.info("Applied FSDP activation checkpointing to Flux2-Klein transformer blocks")

        # Apply FSDP sharding to joint transformer blocks
        for block in self.transformer.transformer_blocks:
            fully_shard(block, **kwargs)

        # Apply FSDP sharding to single transformer blocks
        for block in self.transformer.single_transformer_blocks:
            fully_shard(block, **kwargs)

        fully_shard(self.transformer, **kwargs)

    def init_preprocessors(self):
        """Initialize text and image encoders."""
        if not hasattr(self, "text_encoder"):
            self.init_text_encoder()
        if not hasattr(self, "vae"):
            self.init_vae()

    def init_text_encoder(self):
        """Initialize the text encoder for Flux2-Klein."""
        self.text_encoder = Flux2KleinTextEncoder(model_id=self.model_id)

    def init_vae(self):
        """Initialize only the VAE for visualization."""
        self.vae = Flux2KleinImageEncoder(model_id=self.model_id)

    def to(self, *args, **kwargs):
        """Moves the model to the specified device."""
        super().to(*args, **kwargs)
        if hasattr(self, "text_encoder"):
            self.text_encoder.to(*args, **kwargs)
        if hasattr(self, "vae"):
            self.vae.to(*args, **kwargs)
        return self

    def _prepare_latent_image_ids(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare image position IDs for the transformer.

        Note: Flux2-Klein has patch_size=1, so no packing is applied.

        Args:
            height: Latent height.
            width: Latent width.
            device: Target device.
            dtype: Target dtype.

        Returns:
            torch.Tensor: Image position IDs [H*W, 3] (2D, no batch dim).
        """
        latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
        latent_image_ids[..., 1] = torch.arange(height, device=device, dtype=dtype)[:, None]
        latent_image_ids[..., 2] = torch.arange(width, device=device, dtype=dtype)[None, :]
        latent_image_ids = latent_image_ids.reshape(height * width, 3)
        return latent_image_ids

    def _prepare_text_ids(
        self,
        seq_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare text position IDs.

        Args:
            seq_length: Text sequence length.
            device: Target device.
            dtype: Target dtype.

        Returns:
            torch.Tensor: Text position IDs [seq_length, 3] (2D, no batch dim).
        """
        text_ids = torch.zeros(seq_length, 3, device=device, dtype=dtype)
        return text_ids

    def _flatten_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Flatten latents from [B, C, H, W] to [B, H*W, C] for Flux2-Klein transformer.

        Note: Flux2-Klein uses patch_size=1, so we just reshape without packing.

        Args:
            latents: Input latents [B, C, H, W].

        Returns:
            Flattened latents [B, H*W, C].
        """
        batch_size, channels, height, width = latents.shape
        # Permute to [B, H, W, C] then reshape to [B, H*W, C]
        latents = latents.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        return latents

    def _unflatten_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Unflatten latents from [B, H*W, C] to [B, C, H, W].

        Args:
            latents: Flattened latents [B, H*W, C].
            height: Target height.
            width: Target width.

        Returns:
            Unflattened latents [B, C, H, W].
        """
        batch_size = latents.shape[0]
        channels = latents.shape[2]
        # Reshape to [B, H, W, C] then permute to [B, C, H, W]
        latents = latents.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        return latents

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        r: Optional[torch.Tensor] = None,  # unused, kept for API compatibility
        guidance: Optional[torch.Tensor] = None,  # unused for Flux2-Klein
        return_features_early: bool = False,
        feature_indices: Optional[Set[int]] = None,
        return_logvar: bool = False,
        fwd_pred_type: Optional[str] = None,
        **fwd_kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of Flux2-Klein diffusion model.

        Args:
            x_t: The diffused data sample [B, C, H, W] where C=128 for Flux2-Klein.
            t: The current timestep.
            condition: Tuple of (pooled_prompt_embeds, prompt_embeds) from text encoder.
            r: Another timestep (for mean flow methods).
            return_features_early: If True, return features once collected.
            feature_indices: Set of block indices for feature extraction.
            return_logvar: If True, return the logvar.
            fwd_pred_type: Override network prediction type.
            guidance: Guidance (unused for Flux2-Klein, no embedded guidance).

        Returns:
            Model output tensor or tuple with logvar/features.
        """
        if feature_indices is None:
            feature_indices = set()
        if return_features_early and len(feature_indices) == 0:
            return []

        if fwd_pred_type is None:
            fwd_pred_type = self.net_pred_type
        else:
            assert fwd_pred_type in NET_PRED_TYPES, f"{fwd_pred_type} is not supported"

        height, width = x_t.shape[2], x_t.shape[3]

        # Unpack condition: (pooled_prompt_embeds, prompt_embeds)
        pooled_prompt_embeds, prompt_embeds = condition

        # Prepare position IDs (2D tensors, no batch dimension)
        img_ids = self._prepare_latent_image_ids(height, width, x_t.device, x_t.dtype)
        txt_ids = self._prepare_text_ids(prompt_embeds.shape[1], x_t.device, x_t.dtype)

        # Flatten latents for transformer: [B, C, H, W] -> [B, H*W, C]
        hidden_states = self._flatten_latents(x_t)

        model_outputs = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=t,  # Flux expects timestep in [0, 1]
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,  # Flux2-Klein doesn't use embedded guidance
            return_features_early=return_features_early,
            feature_indices=feature_indices,
            return_logvar=return_logvar,
        )

        if return_features_early:
            return model_outputs

        if return_logvar:
            out, logvar = model_outputs[0], model_outputs[1]
        else:
            out = model_outputs

        # Unflatten output: [B, H*W, C] -> [B, C, H, W]
        if isinstance(out, torch.Tensor):
            out = self._unflatten_latents(out, height, width)
            out = self.noise_scheduler.convert_model_output(
                x_t, out, t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )
        else:
            out[0] = self._unflatten_latents(out[0], height, width)
            out[0] = self.noise_scheduler.convert_model_output(
                x_t, out[0], t, src_pred_type=self.net_pred_type, target_pred_type=fwd_pred_type
            )

        if return_logvar:
            return out, logvar
        return out

    def _calculate_shift(
        self,
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
    ) -> float:
        """Calculate the shift value for the scheduler based on image resolution.

        This implements the resolution-dependent shift from the Flux paper.
        """
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        neg_condition: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        guidance_scale: Optional[float] = 4.0,
        num_steps: int = 4,  # Default to 4 steps for distilled model
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples using Euler flow matching.

        Args:
            noise: Initial noise tensor [B, C, H, W] where C=128 for Flux2-Klein.
            condition: Tuple of (pooled_prompt_embeds, prompt_embeds).
            neg_condition: Optional negative condition tuple for CFG.
            guidance_scale: Guidance scale for CFG. Set to 1.0 for distilled model.
            num_steps: Number of sampling steps (default 4 for distilled, 50 for base).
            **kwargs: Additional keyword arguments

        Returns:
            Generated latent samples.
        """
        batch_size, channels, height, width = noise.shape

        # Calculate image sequence length for shift calculation
        image_seq_len = height * width

        # Calculate resolution-dependent shift (mu)
        mu = self._calculate_shift(image_seq_len)

        # Initialize scheduler with proper shift
        scheduler = FlowMatchEulerDiscreteScheduler(shift=mu)
        scheduler.set_timesteps(num_steps, device=noise.device)
        timesteps = scheduler.timesteps

        # Initialize latents with proper scaling based on the initial timestep
        t_init = self.noise_scheduler.safe_clamp(
            timesteps[0] / 1000.0, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t
        )
        latents = self.noise_scheduler.latents(noise=noise, t_init=t_init)

        pooled_prompt_embeds, prompt_embeds = condition

        # Sampling loop
        for timestep in timesteps:
            # Scheduler timesteps are in [0, 1000], transformer expects [0, 1]
            t = (timestep / 1000.0).expand(batch_size)
            t = self.noise_scheduler.safe_clamp(t, min=self.noise_scheduler.min_t, max=self.noise_scheduler.max_t).to(
                latents.dtype
            )

            # CFG mode when neg_condition is provided
            if neg_condition is not None and guidance_scale != 1.0:
                neg_pooled, neg_prompt = neg_condition
                latent_model_input = torch.cat([latents, latents], dim=0)
                pooled_input = torch.cat([neg_pooled, pooled_prompt_embeds], dim=0)
                prompt_input = torch.cat([neg_prompt, prompt_embeds], dim=0)
                t_input = torch.cat([t, t], dim=0)

                noise_pred = self(
                    latent_model_input,
                    t_input,
                    (pooled_input, prompt_input),
                    fwd_pred_type="flow",
                )

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # No CFG (distilled model or guidance_scale=1.0)
                noise_pred = self(
                    latents,
                    t,
                    condition,
                    fwd_pred_type="flow",
                )

            # Euler step
            latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

        return latents
