#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Prepare an editing dataset for FastGen LADD training.

This script transforms a dataset structured as:
    input_dir/
    ├── before/
    │   ├── image001.jpg
    │   ├── image002.png
    │   └── ...
    └── after/
        ├── image001.jpg
        ├── image001.txt  (caption/instruction)
        ├── image002.png
        ├── image002.txt
        └── ...

Into WebDataset format for FastGen:
    output_dir/
    ├── shard-00000.tar
    │   ├── 000000.source.jpg
    │   ├── 000000.target.jpg
    │   ├── 000000.instruction.txt
    │   ├── 000001.source.jpg
    │   └── ...
    └── shard-00001.tar
        └── ...

Usage:
    python scripts/prepare_editing_dataset.py \\
        --input_dir /path/to/dataset \\
        --output_dir /path/to/webdataset \\
        --shard_size 500

    # Preview without creating files:
    python scripts/prepare_editing_dataset.py \\
        --input_dir /path/to/dataset \\
        --dry_run
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import io

# Check for webdataset
try:
    import webdataset as wds
    HAS_WEBDATASET = True
except ImportError:
    HAS_WEBDATASET = False
    print("Warning: webdataset not installed. Only --output_format=folder will work.")
    print("Install with: pip install webdataset")

from PIL import Image


# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def find_matching_pairs(
    input_dir: Path,
    before_subdir: str = "before",
    after_subdir: str = "after",
) -> List[Tuple[Path, Path, Path]]:
    """Find matching image pairs between before and after directories.

    Args:
        input_dir: Root directory containing before/ and after/ subdirs
        before_subdir: Name of the 'before' subdirectory
        after_subdir: Name of the 'after' subdirectory

    Returns:
        List of tuples: (before_image, after_image, caption_file)
    """
    before_dir = input_dir / before_subdir
    after_dir = input_dir / after_subdir

    if not before_dir.exists():
        raise FileNotFoundError(f"Before directory not found: {before_dir}")
    if not after_dir.exists():
        raise FileNotFoundError(f"After directory not found: {after_dir}")

    # Index before images by stem (filename without extension)
    before_images = {}
    for f in before_dir.iterdir():
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            before_images[f.stem] = f

    # Find matching pairs in after directory
    pairs = []
    missing_before = []
    missing_caption = []

    for f in after_dir.iterdir():
        if f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stem = f.stem

        # Check for matching before image
        if stem not in before_images:
            missing_before.append(stem)
            continue

        # Check for caption file
        caption_file = None
        for ext in ['.txt', '.caption', '.text']:
            candidate = after_dir / f"{stem}{ext}"
            if candidate.exists():
                caption_file = candidate
                break

        if caption_file is None:
            missing_caption.append(stem)
            continue

        pairs.append((before_images[stem], f, caption_file))

    # Report issues
    if missing_before:
        print(f"Warning: {len(missing_before)} images in 'after' have no matching 'before' image")
        if len(missing_before) <= 5:
            for name in missing_before:
                print(f"  - {name}")
        else:
            print(f"  First 5: {missing_before[:5]}")

    if missing_caption:
        print(f"Warning: {len(missing_caption)} image pairs have no caption file")
        if len(missing_caption) <= 5:
            for name in missing_caption:
                print(f"  - {name}")
        else:
            print(f"  First 5: {missing_caption[:5]}")

    # Sort by filename for reproducibility
    pairs.sort(key=lambda x: x[0].stem)

    return pairs


def convert_image_to_jpeg(image_path: Path, quality: int = 95) -> bytes:
    """Convert any image to JPEG bytes.

    Args:
        image_path: Path to the image file
        quality: JPEG quality (1-100)

    Returns:
        JPEG image as bytes
    """
    img = Image.open(image_path)

    # Convert to RGB if necessary (handles RGBA, P mode, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    return buffer.getvalue()


def create_webdataset(
    pairs: List[Tuple[Path, Path, Path]],
    output_dir: Path,
    shard_size: int = 500,
    jpeg_quality: int = 95,
) -> int:
    """Create WebDataset tar files from image pairs.

    Args:
        pairs: List of (before_image, after_image, caption) tuples
        output_dir: Directory to write tar shards
        shard_size: Number of samples per shard
        jpeg_quality: JPEG quality for converted images

    Returns:
        Number of samples written
    """
    if not HAS_WEBDATASET:
        raise ImportError("webdataset is required for tar output. Install with: pip install webdataset")

    output_dir.mkdir(parents=True, exist_ok=True)

    shard_pattern = str(output_dir / "shard-%05d.tar")

    count = 0
    with wds.ShardWriter(shard_pattern, maxcount=shard_size) as sink:
        for idx, (before_path, after_path, caption_path) in enumerate(pairs):
            # Read caption
            caption = caption_path.read_text(encoding='utf-8').strip()

            # Convert images to JPEG
            source_bytes = convert_image_to_jpeg(before_path, jpeg_quality)
            target_bytes = convert_image_to_jpeg(after_path, jpeg_quality)

            # Write sample
            sample = {
                "__key__": f"{idx:06d}",
                "source.jpg": source_bytes,
                "target.jpg": target_bytes,
                "instruction.txt": caption,
            }
            sink.write(sample)
            count += 1

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(pairs)} samples...")

    return count


def create_folder_structure(
    pairs: List[Tuple[Path, Path, Path]],
    output_dir: Path,
    jpeg_quality: int = 95,
) -> int:
    """Create a flat folder structure (for inspection or alternative loading).

    Creates:
        output_dir/
        ├── 000000_source.jpg
        ├── 000000_target.jpg
        ├── 000000_instruction.txt
        └── ...

    Args:
        pairs: List of (before_image, after_image, caption) tuples
        output_dir: Directory to write files
        jpeg_quality: JPEG quality for converted images

    Returns:
        Number of samples written
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx, (before_path, after_path, caption_path) in enumerate(pairs):
        prefix = f"{idx:06d}"

        # Convert and save source image
        source_bytes = convert_image_to_jpeg(before_path, jpeg_quality)
        (output_dir / f"{prefix}_source.jpg").write_bytes(source_bytes)

        # Convert and save target image
        target_bytes = convert_image_to_jpeg(after_path, jpeg_quality)
        (output_dir / f"{prefix}_target.jpg").write_bytes(target_bytes)

        # Copy caption
        caption = caption_path.read_text(encoding='utf-8').strip()
        (output_dir / f"{prefix}_instruction.txt").write_text(caption, encoding='utf-8')

        count += 1

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(pairs)} samples...")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Prepare editing dataset for FastGen LADD training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create WebDataset tar files (recommended):
  python scripts/prepare_editing_dataset.py \\
      --input_dir /data/my_editing_dataset \\
      --output_dir /data/fastgen_editing \\
      --shard_size 500

  # Create flat folder structure (for inspection):
  python scripts/prepare_editing_dataset.py \\
      --input_dir /data/my_editing_dataset \\
      --output_dir /data/fastgen_editing_flat \\
      --output_format folder

  # Preview without creating files:
  python scripts/prepare_editing_dataset.py \\
      --input_dir /data/my_editing_dataset \\
      --dry_run
        """
    )

    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input directory containing 'before/' and 'after/' subdirectories"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for WebDataset or folder structure"
    )
    parser.add_argument(
        "--output_format", type=str, default="webdataset", choices=["webdataset", "folder"],
        help="Output format: 'webdataset' (tar shards) or 'folder' (flat files)"
    )
    parser.add_argument(
        "--shard_size", type=int, default=500,
        help="Number of samples per WebDataset shard (default: 500)"
    )
    parser.add_argument(
        "--jpeg_quality", type=int, default=95,
        help="JPEG quality for converted images (default: 95)"
    )
    parser.add_argument(
        "--before_subdir", type=str, default="before",
        help="Name of the 'before' subdirectory (default: 'before')"
    )
    parser.add_argument(
        "--after_subdir", type=str, default="after",
        help="Name of the 'after' subdirectory (default: 'after')"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Preview what would be done without creating files"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    print(f"Scanning input directory: {input_dir}")
    print(f"  Before subdir: {args.before_subdir}/")
    print(f"  After subdir: {args.after_subdir}/")
    print()

    # Find matching pairs
    pairs = find_matching_pairs(
        input_dir,
        before_subdir=args.before_subdir,
        after_subdir=args.after_subdir,
    )

    print(f"\nFound {len(pairs)} valid image pairs")

    if len(pairs) == 0:
        print("No valid pairs found. Please check your directory structure.")
        sys.exit(1)

    # Preview first few pairs
    print("\nFirst 3 pairs:")
    for i, (before, after, caption) in enumerate(pairs[:3]):
        cap_text = caption.read_text(encoding='utf-8').strip()[:50]
        print(f"  {i+1}. {before.name} -> {after.name}")
        print(f"     Caption: {cap_text}...")

    if args.dry_run:
        print("\n[DRY RUN] No files created.")
        print("\nTo create the dataset, run without --dry_run:")
        print(f"  python {sys.argv[0]} --input_dir {args.input_dir} --output_dir <output_path>")
        return

    if args.output_dir is None:
        print("\nError: --output_dir is required (unless using --dry_run)")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    print(f"\nCreating {args.output_format} output at: {output_dir}")

    if args.output_format == "webdataset":
        count = create_webdataset(
            pairs,
            output_dir,
            shard_size=args.shard_size,
            jpeg_quality=args.jpeg_quality,
        )

        # Count shards created
        shards = list(output_dir.glob("shard-*.tar"))
        print(f"\nDone! Created {len(shards)} shards with {count} samples total")
        print("\nTo use in FastGen:")
        print(f'  --dataloader_train.datatags=\'["WDS:{output_dir}/shard-{{00000..{len(shards)-1:05d}}}.tar"]\'')

    else:  # folder
        count = create_folder_structure(
            pairs,
            output_dir,
            jpeg_quality=args.jpeg_quality,
        )
        print(f"\nDone! Created {count} sample triplets in {output_dir}")
        print("\nNote: To use with FastGen, convert to WebDataset format using --output_format=webdataset")


if __name__ == "__main__":
    main()
