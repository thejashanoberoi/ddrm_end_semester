#!/usr/bin/env python3
"""
DDRM PSNR Computation Script
=============================

Computes PSNR metrics from DDRM output folders containing original,
degraded (blurred/downsampled), and restored images.

Usage:
    python compute_psnr.py --input-folder exp/image_samples/demo_deblur

    python compute_psnr.py --input-folder exp/image_samples/demo_deblur \
        --output-json results.json --output-csv results.csv --verbose

Based on static analysis:
- Output pattern: runners/diffusion.py:293-321
- PSNR formula: runners/diffusion.py:319 (10 * log10(1 / MSE))
- File patterns: orig_{idx}.png, y0_{idx}.png, {idx}_-1.png

Author: Static Analysis System
Date: December 4, 2025
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

try:
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"Error: Required library not found: {e}")
    print("Please install: pip install numpy pillow")
    sys.exit(1)


class PSNRComputer:
    """
    Computes PSNR metrics from DDRM output folders.

    Based on DDRM code analysis (runners/diffusion.py:319):
        mse = torch.mean((restored - original) ** 2)
        psnr = 10 * torch.log10(1 / mse)

    Assumes images are in [0,1] range after normalization.
    """

    def __init__(self, eps: float = 1e-12, clip: bool = True, verbose: bool = True):
        """
        Initialize PSNR computer.

        Args:
            eps: Small epsilon to avoid division by zero (default: 1e-12)
            clip: Whether to clip images to [0,1] before computing MSE
            verbose: Print progress and results to console
        """
        self.eps = eps
        self.clip = clip
        self.verbose = verbose
        self.skipped_files = []

    def load_image(self, path: Path) -> Optional[np.ndarray]:
        """
        Load image and convert to float32 [0,1] range.

        Args:
            path: Path to image file

        Returns:
            numpy array of shape (H, W, C) with values in [0,1], or None if error
        """
        try:
            img = Image.open(path)

            # Convert to RGB if needed
            if img.mode != 'RGB':
                if self.verbose:
                    warnings.warn(f"Converting {path.name} from {img.mode} to RGB")
                img = img.convert('RGB')

            # Convert to numpy array [0, 255] uint8
            img_array = np.array(img, dtype=np.float32)

            # Normalize to [0, 1]
            img_array = img_array / 255.0

            # Clip if requested
            if self.clip:
                img_array = np.clip(img_array, 0.0, 1.0)

            return img_array

        except Exception as e:
            warnings.warn(f"Failed to load {path}: {e}")
            self.skipped_files.append(str(path))
            return None

    def resize_to_match(self, img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to match target shape.

        Args:
            img: Image array (H, W, C)
            target_shape: (target_H, target_W)

        Returns:
            Resized image array
        """
        if img.shape[:2] == target_shape:
            return img

        # Convert back to PIL for resizing
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil = img_pil.resize((target_shape[1], target_shape[0]), Image.BILINEAR)

        # Convert back to numpy [0,1]
        return np.array(img_pil, dtype=np.float32) / 255.0

    def compute_mse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Mean Squared Error between two images.

        Args:
            img1: First image (H, W, C) in [0,1]
            img2: Second image (H, W, C) in [0,1]

        Returns:
            MSE value (scalar)
        """
        # Ensure same shape
        if img1.shape != img2.shape:
            if self.verbose:
                warnings.warn(f"Shape mismatch: {img1.shape} vs {img2.shape}. Resizing...")
            img2 = self.resize_to_match(img2, img1.shape[:2])

        # Compute MSE: mean over all pixels and channels
        mse = np.mean((img1 - img2) ** 2)
        return float(mse)

    def compute_psnr(self, mse: float) -> float:
        """
        Compute PSNR from MSE.

        Formula (from runners/diffusion.py:319):
            PSNR = 10 * log10(1 / MSE)

        Args:
            mse: Mean Squared Error

        Returns:
            PSNR in dB, or inf if MSE < eps
        """
        if mse < self.eps:
            return float('inf')

        psnr = 10 * np.log10(1.0 / mse)
        return float(psnr)

    def compute_psnr_pair(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute PSNR between two images.

        Args:
            img1: First image (reference)
            img2: Second image (comparison)

        Returns:
            PSNR in dB
        """
        mse = self.compute_mse(img1, img2)
        return self.compute_psnr(mse)


def discover_images(input_folder: Path) -> Dict[int, Dict[str, Path]]:
    """
    Discover and group images by index.

    File patterns (from runners/diffusion.py:293-313):
        - orig_{idx}.png
        - y0_{idx}.png (degraded/blurred)
        - {idx}_-1.png (final restored)

    Args:
        input_folder: Path to folder containing images

    Returns:
        Dictionary mapping idx -> {'orig': Path, 'blurred': Path, 'restored': Path}
    """
    images = {}

    # Find all original images
    for orig_file in sorted(input_folder.glob('orig_*.png')):
        try:
            # Extract index from 'orig_{idx}.png'
            idx_str = orig_file.stem.replace('orig_', '')
            idx = int(idx_str)

            if idx not in images:
                images[idx] = {}

            images[idx]['orig'] = orig_file

        except ValueError:
            warnings.warn(f"Could not parse index from {orig_file.name}")
            continue

    # Find all degraded/blurred images
    for blurred_file in sorted(input_folder.glob('y0_*.png')):
        try:
            idx_str = blurred_file.stem.replace('y0_', '')
            idx = int(idx_str)

            if idx not in images:
                images[idx] = {}

            images[idx]['blurred'] = blurred_file

        except ValueError:
            warnings.warn(f"Could not parse index from {blurred_file.name}")
            continue

    # Find all restored images
    for restored_file in sorted(input_folder.glob('*_-1.png')):
        try:
            # Extract index from '{idx}_-1.png'
            idx_str = restored_file.stem.replace('_-1', '')
            idx = int(idx_str)

            if idx not in images:
                images[idx] = {}

            images[idx]['restored'] = restored_file

        except ValueError:
            warnings.warn(f"Could not parse index from {restored_file.name}")
            continue

    return images


def validate_image_sets(images: Dict[int, Dict[str, Path]], verbose: bool = True) -> Dict[int, Dict[str, Path]]:
    """
    Validate that we have matching orig and restored images.

    Args:
        images: Dictionary mapping idx -> image paths
        verbose: Print warnings

    Returns:
        Filtered dictionary with only valid sets (orig + restored)
    """
    valid_images = {}
    missing_counts = {'orig': 0, 'restored': 0, 'blurred': 0}

    for idx, paths in sorted(images.items()):
        has_orig = 'orig' in paths
        has_restored = 'restored' in paths
        has_blurred = 'blurred' in paths

        if not has_orig:
            missing_counts['orig'] += 1
            if verbose:
                warnings.warn(f"Index {idx}: Missing original image")

        if not has_restored:
            missing_counts['restored'] += 1
            if verbose:
                warnings.warn(f"Index {idx}: Missing restored image")

        if not has_blurred:
            missing_counts['blurred'] += 1
            # Blurred is optional for PSNR computation

        # Require at least orig and restored for PSNR computation
        if has_orig and has_restored:
            valid_images[idx] = paths

    if verbose and (missing_counts['orig'] > 0 or missing_counts['restored'] > 0):
        print(f"\n⚠️  Missing files:")
        print(f"   Original: {missing_counts['orig']}")
        print(f"   Restored: {missing_counts['restored']}")
        print(f"   Blurred: {missing_counts['blurred']} (optional)")
        print(f"   Processing {len(valid_images)}/{len(images)} image sets\n")

    return valid_images


def compute_all_psnr(
    input_folder: Path,
    eps: float = 1e-12,
    clip: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Compute PSNR for all images in folder.

    Args:
        input_folder: Path to DDRM output folder
        eps: Epsilon for avoiding division by zero
        clip: Clip images to [0,1]
        verbose: Print progress

    Returns:
        Dictionary with results
    """
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    # Initialize computer
    computer = PSNRComputer(eps=eps, clip=clip, verbose=verbose)

    if verbose:
        print(f"📁 Input folder: {input_folder}")
        print(f"🔍 Discovering images...")

    # Discover images
    images = discover_images(input_folder)

    if not images:
        raise ValueError(f"No images found in {input_folder}")

    if verbose:
        print(f"   Found {len(images)} image sets")

    # Validate
    valid_images = validate_image_sets(images, verbose=verbose)

    if not valid_images:
        raise ValueError(f"No valid image pairs (orig + restored) found")

    # Compute PSNR for each image
    results = {
        'input_folder': str(input_folder),
        'n_samples': len(valid_images),
        'per_image': [],
        'skipped': [],
    }

    psnr_orig_restored_list = []
    psnr_orig_blurred_list = []

    if verbose:
        print(f"\n📊 Computing PSNR...")
        print(f"{'Index':<8} {'PSNR (orig vs restored)':<25} {'PSNR (orig vs blurred)':<25}")
        print("=" * 65)

    for idx in sorted(valid_images.keys()):
        paths = valid_images[idx]

        # Load images
        orig = computer.load_image(paths['orig'])
        restored = computer.load_image(paths['restored'])

        if orig is None or restored is None:
            continue

        # Compute PSNR: original vs restored
        psnr_orig_restored = computer.compute_psnr_pair(orig, restored)
        psnr_orig_restored_list.append(psnr_orig_restored)

        # Optionally compute PSNR: original vs blurred
        psnr_orig_blurred = None
        if 'blurred' in paths:
            blurred = computer.load_image(paths['blurred'])
            if blurred is not None:
                psnr_orig_blurred = computer.compute_psnr_pair(orig, blurred)
                psnr_orig_blurred_list.append(psnr_orig_blurred)

        # Store result
        result_entry = {
            'idx': idx,
            'files': {
                'orig': paths['orig'].name,
                'restored': paths['restored'].name,
                'blurred': paths.get('blurred', {}).name if 'blurred' in paths else None,
            },
            'psnr_orig_restored': psnr_orig_restored,
            'psnr_orig_blurred': psnr_orig_blurred,
        }
        results['per_image'].append(result_entry)

        # Print if verbose
        if verbose:
            psnr_restored_str = f"{psnr_orig_restored:.4f} dB" if psnr_orig_restored != float('inf') else "∞ (perfect)"
            psnr_blurred_str = f"{psnr_orig_blurred:.4f} dB" if psnr_orig_blurred is not None and psnr_orig_blurred != float('inf') else "N/A"
            print(f"{idx:<8} {psnr_restored_str:<25} {psnr_blurred_str:<25}")

    # Compute aggregate statistics
    if psnr_orig_restored_list:
        # Filter out infinities for statistics
        finite_restored = [p for p in psnr_orig_restored_list if p != float('inf')]

        if finite_restored:
            results['aggregate'] = {
                'psnr_orig_restored': {
                    'mean': float(np.mean(finite_restored)),
                    'std': float(np.std(finite_restored, ddof=1)) if len(finite_restored) > 1 else 0.0,
                    'median': float(np.median(finite_restored)),
                    'min': float(np.min(finite_restored)),
                    'max': float(np.max(finite_restored)),
                    'n_finite': len(finite_restored),
                    'n_infinite': len(psnr_orig_restored_list) - len(finite_restored),
                }
            }

            # Add blurred statistics if available
            if psnr_orig_blurred_list:
                finite_blurred = [p for p in psnr_orig_blurred_list if p != float('inf')]
                if finite_blurred:
                    results['aggregate']['psnr_orig_blurred'] = {
                        'mean': float(np.mean(finite_blurred)),
                        'std': float(np.std(finite_blurred, ddof=1)) if len(finite_blurred) > 1 else 0.0,
                        'median': float(np.median(finite_blurred)),
                        'min': float(np.min(finite_blurred)),
                        'max': float(np.max(finite_blurred)),
                        'n_finite': len(finite_blurred),
                    }
        else:
            results['aggregate'] = {'note': 'All PSNR values are infinite (perfect reconstruction)'}

    # Add skipped files
    results['skipped'] = computer.skipped_files

    if verbose:
        print("=" * 65)
        if 'aggregate' in results and 'psnr_orig_restored' in results['aggregate']:
            agg = results['aggregate']['psnr_orig_restored']
            print(f"\n📈 Aggregate Statistics (Original vs Restored):")
            print(f"   Mean PSNR:   {agg['mean']:.4f} dB")
            print(f"   Std PSNR:    {agg['std']:.4f} dB")
            print(f"   Median PSNR: {agg['median']:.4f} dB")
            print(f"   Min PSNR:    {agg['min']:.4f} dB")
            print(f"   Max PSNR:    {agg['max']:.4f} dB")
            print(f"   Samples:     {agg['n_finite']} finite, {agg['n_infinite']} infinite")
        print()

    return results


def save_json(results: Dict, output_path: Path, verbose: bool = True):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"💾 JSON results saved to: {output_path}")


def save_csv(results: Dict, output_path: Path, verbose: bool = True):
    """Save results to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'idx',
            'filename_orig',
            'filename_restored',
            'filename_blurred',
            'psnr_orig_restored',
            'psnr_orig_blurred'
        ])

        # Write data rows
        for entry in results['per_image']:
            writer.writerow([
                entry['idx'],
                entry['files']['orig'],
                entry['files']['restored'],
                entry['files']['blurred'] or '',
                entry['psnr_orig_restored'] if entry['psnr_orig_restored'] != float('inf') else 'inf',
                entry['psnr_orig_blurred'] if entry['psnr_orig_blurred'] is not None and entry['psnr_orig_blurred'] != float('inf') else ''
            ])

    if verbose:
        print(f"💾 CSV results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute PSNR metrics from DDRM output folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python compute_psnr.py --input-folder exp/image_samples/demo_deblur
  
  # Custom output files
  python compute_psnr.py --input-folder exp/image_samples/demo_deblur \\
      --output-json results.json --output-csv results.csv
  
  # Silent mode
  python compute_psnr.py --input-folder exp/image_samples/demo_deblur --no-verbose
  
  # Disable clipping (not recommended)
  python compute_psnr.py --input-folder exp/image_samples/demo_deblur --no-clip

File Patterns (from runners/diffusion.py):
  orig_{idx}.png    - Original clean image
  y0_{idx}.png      - Degraded/blurred image (optional for PSNR)
  {idx}_-1.png      - Final restored image

PSNR Formula (from runners/diffusion.py:319):
  MSE = mean((original - restored)^2)
  PSNR = 10 * log10(1 / MSE)

Output:
  - JSON file with per-image PSNR and aggregate statistics
  - Optional CSV file for easy analysis
  - Console output with progress and results
        """
    )

    # Required arguments
    parser.add_argument(
        '--input-folder',
        type=str,
        required=True,
        help='Path to DDRM output folder (e.g., exp/image_samples/demo_deblur)'
    )

    # Output arguments
    parser.add_argument(
        '--output-json',
        type=str,
        default='psnr_results.json',
        help='Path for JSON output (default: psnr_results.json)'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default=None,
        help='Optional path for CSV output'
    )

    # Computation parameters
    parser.add_argument(
        '--clip',
        dest='clip',
        action='store_true',
        default=True,
        help='Clip images to [0,1] before computing MSE (default: True)'
    )
    parser.add_argument(
        '--no-clip',
        dest='clip',
        action='store_false',
        help='Disable clipping (not recommended)'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-12,
        help='Epsilon to avoid division by zero (default: 1e-12)'
    )

    # Verbosity
    parser.add_argument(
        '--verbose',
        dest='verbose',
        action='store_true',
        default=True,
        help='Print progress and results (default: True)'
    )
    parser.add_argument(
        '--no-verbose',
        dest='verbose',
        action='store_false',
        help='Suppress console output'
    )

    args = parser.parse_args()

    # Convert paths
    input_folder = Path(args.input_folder)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv) if args.output_csv else None

    try:
        # Print header
        if args.verbose:
            print("╔════════════════════════════════════════════════════════════════╗")
            print("║           DDRM PSNR Computation Tool                          ║")
            print("╚════════════════════════════════════════════════════════════════╝")
            print()

        # Compute PSNR
        results = compute_all_psnr(
            input_folder=input_folder,
            eps=args.eps,
            clip=args.clip,
            verbose=args.verbose
        )

        # Save results
        save_json(results, output_json, verbose=args.verbose)

        if output_csv:
            save_csv(results, output_csv, verbose=args.verbose)

        if args.verbose:
            print("\n✅ PSNR computation completed successfully!")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

