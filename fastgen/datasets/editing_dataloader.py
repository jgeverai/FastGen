# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
WebDataset loader for image editing triplets.

This loader handles datasets in the format:
    {key}_source.jpg  - Source image to be edited
    {key}_target.jpg  - Ground truth edited result
    {key}_instruction.txt - Edit instruction text

Used for LADD editing distillation where the model learns to transform
source images into edited targets based on text instructions.
"""

from typing import Any, Dict, List, Optional

from fastgen.datasets.wds_dataloaders import WDSLoader, ImageWDSLoader
import fastgen.utils.logging_utils as logger


class EditingWDSLoader(ImageWDSLoader):
    """WebDataset loader for image editing triplets.

    Loads triplets of (source_image, target_image, instruction) for
    image-to-image editing distillation. Both source and target images
    are processed with the same transforms.

    WebDataset format expected:
        {sample_key}.source.jpg   - Source image
        {sample_key}.target.jpg   - Target (edited) image
        {sample_key}.instruction.txt - Edit instruction

    Args:
        datatags: List of WDS tags in format 'WDS:<PATH>'
        batch_size: Batch size for the dataloader
        input_res: Target image resolution (square). Default: 1024

        source_key: Item key for source images. Default: "source.jpg"
        target_key: Item key for target images. Default: "target.jpg"
        instruction_key: Item key for edit instructions. Default: "instruction.txt"

        **kwargs: Additional arguments passed to ImageWDSLoader.

    Output dict keys:
        - "source": Source image tensor [C, H, W] normalized to [-1, 1]
        - "target": Target image tensor [C, H, W] normalized to [-1, 1]
        - "instruction": Edit instruction string
        - "fname": Sample filename
        - "shard": Shard path

    Example config:
        ```python
        from fastgen.configs.data import EditingWebDatasetConfig

        dataloader = EditingWebDatasetConfig(
            datatags=["WDS:/data/editing_dataset/{00000..00099}.tar"],
            batch_size=2,
            input_res=1024,
            source_key="source.jpg",
            target_key="target.jpg",
            instruction_key="instruction.txt",
        )
        ```
    """

    def __init__(
        self,
        datatags: List[str],
        batch_size: int,
        input_res: int = 1024,
        source_key: str = "source.jpg",
        target_key: str = "target.jpg",
        instruction_key: str = "instruction.txt",
        **kwargs,
    ):
        # Store editing-specific keys
        self._source_key = source_key
        self._target_key = target_key
        self._instruction_key = instruction_key

        # Set up key_map for the parent class
        key_map = {
            "source": source_key,
            "target": target_key,
            "instruction": instruction_key,
        }

        super().__init__(
            datatags=datatags,
            batch_size=batch_size,
            input_res=input_res,
            key_map=key_map,
            txt_extensions=(instruction_key.split(".")[-1],),  # e.g., "txt"
            **kwargs,
        )
        logger.info(
            f"EditingWDSLoader initialized: source={source_key}, target={target_key}, "
            f"instruction={instruction_key}, res={input_res}"
        )

    def _preprocess(self, item: dict) -> Dict[str, Any]:
        """Preprocess editing triplet data.

        Ensures both source and target images are processed identically
        and the instruction text is properly decoded.

        Args:
            item: Decoded WebDataset item dict.

        Returns:
            Dict with 'source', 'target', 'instruction', 'fname', 'shard'.
        """
        output = super()._preprocess(item)

        # Validate that we have all required keys
        required_keys = ["source", "target", "instruction"]
        for key in required_keys:
            if key not in output:
                raise KeyError(f"Missing required key '{key}' in editing dataset item")

        # Ensure instruction is a string
        if isinstance(output["instruction"], bytes):
            output["instruction"] = output["instruction"].decode("utf-8").strip()

        return output

    def filter_items(self, item: dict) -> bool:
        """Filter function for editing triplet items.

        Ensures all three components (source, target, instruction) are present
        and contain valid data.

        Args:
            item: Raw WebDataset item dict (pre-decoding).

        Returns:
            True if item has all required components, False otherwise.
        """
        # Check that all required keys are present
        required_keys = [self._source_key, self._target_key, self._instruction_key]
        for key in required_keys:
            if key not in item:
                return False
            if not isinstance(item[key], (bytes, bytearray)):
                return False
            # Check that data is non-empty
            if len(item[key]) == 0:
                return False

        return super().filter_items(item)


class EditingLatentWDSLoader(WDSLoader):
    """WebDataset loader for pre-encoded editing triplets (latent space).

    For efficiency, you can pre-encode images to latent space and store them
    as .npy or .pth files. This loader handles such pre-encoded data.

    WebDataset format expected:
        {sample_key}.source_latent.pth - Source latent [C, H, W]
        {sample_key}.target_latent.pth - Target latent [C, H, W]
        {sample_key}.instruction.txt   - Edit instruction
        {sample_key}.condition.pth     - Optional: pre-computed text embedding

    Args:
        datatags: List of WDS tags in format 'WDS:<PATH>'
        batch_size: Batch size for the dataloader

        source_key: Item key for source latents. Default: "source_latent.pth"
        target_key: Item key for target latents. Default: "target_latent.pth"
        instruction_key: Item key for instructions. Default: "instruction.txt"
        condition_key: Optional item key for pre-computed text embeddings.

        **kwargs: Additional arguments passed to WDSLoader.

    Output dict keys:
        - "source": Source latent tensor [C, H, W]
        - "target": Target latent tensor [C, H, W]
        - "instruction": Edit instruction string (if condition_key not set)
        - "condition": Pre-computed text embedding (if condition_key set)
        - "fname": Sample filename
        - "shard": Shard path
    """

    def __init__(
        self,
        datatags: List[str],
        batch_size: int,
        source_key: str = "source_latent.pth",
        target_key: str = "target_latent.pth",
        instruction_key: str = "instruction.txt",
        condition_key: Optional[str] = None,
        **kwargs,
    ):
        # Set up key_map
        key_map = {
            "source": source_key,
            "target": target_key,
        }

        if condition_key is not None:
            key_map["condition"] = condition_key
        else:
            key_map["instruction"] = instruction_key

        super().__init__(
            datatags=datatags,
            batch_size=batch_size,
            key_map=key_map,
            txt_extensions=(instruction_key.split(".")[-1],) if condition_key is None else (),
            **kwargs,
        )
        logger.info(f"EditingLatentWDSLoader initialized with key_map={key_map}")
