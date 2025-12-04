#!/usr/bin/env python3
"""
Configuration 1: Custom DDPM (CelebA-HQ)
=========================================

This script runs the Custom DDPM model for face image restoration.

Model Details:
- Type: Custom DDPM (type='simple')
- Architecture: Custom U-Net (~60M parameters)
- Base Channels: 128
- Attention: 1-head @ 1 level (16x16)
- Dataset: CelebA-HQ (faces)
- Classifier: None
- Independence: Fully standalone

Usage:
    python run_config1_custom_ddpm.py [options]

Example:
    # 4x super-resolution on faces
    python run_config1_custom_ddpm.py --deg sr4 --sigma_0 0.05
    
    # Deblurring faces
    python run_config1_custom_ddpm.py --deg deblur_gauss --sigma_0 0.0
    
    # Inpainting faces
    python run_config1_custom_ddpm.py --deg inp --sigma_0 0.0

Static Analysis Reference:
- Config: configs/celeba_hq.yml (type='simple')
- Model: models/diffusion.py::Model
- Loading: runners/diffusion.py:95-118
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_requirements():
    """Check if required files exist."""
    required_files = {
        'main.py': 'DDRM main script',
        'configs/celeba_hq.yml': 'CelebA-HQ configuration',
        'models/diffusion.py': 'Custom DDPM model',
    }

    missing = []
    for file, desc in required_files.items():
        if not Path(file).exists():
            missing.append(f"{file} ({desc})")

    if missing:
        print("❌ Missing required files:")
        for m in missing:
            print(f"  - {m}")
        print("\n⚠️  Make sure you're running from the DDRM repository root.")
        return False

    # Check if model checkpoint exists
    model_path = Path("exp/logs/celeba/celeba_hq.ckpt")
    if not model_path.exists():
        print("⚠️  CelebA-HQ model checkpoint not found at:")
        print(f"   {model_path}")
        print("\n📥 The model will be downloaded automatically on first run.")
        print("   Download source: AWS S3 (test bucket)")
        print("   Size: ~500 MB")
        print("   Note: This URL may be unreliable (see docs/CELEBA_DOWNLOAD_FIX.md)")
        print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run Configuration 1: Custom DDPM (CelebA-HQ)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 4x super-resolution with noise
  %(prog)s --deg sr4 --sigma_0 0.05 --timesteps 40
  
  # 8x super-resolution clean
  %(prog)s --deg sr8 --sigma_0 0.0 --timesteps 100
  
  # Gaussian deblurring
  %(prog)s --deg deblur_gauss --sigma_0 0.0
  
  # Inpainting (random 50%% mask)
  %(prog)s --deg inp --sigma_0 0.0

Supported Degradations:
  sr2, sr4, sr8, sr16       - Super-resolution (factor 2/4/8/16)
  sr_bicubic4, sr_bicubic8  - Bicubic super-resolution
  deblur_uni                - Uniform blur
  deblur_gauss              - Gaussian blur
  deblur_aniso              - Anisotropic blur
  inp, inp_lolcat, inp_lorem - Inpainting
  color                     - Colorization
  deno                      - Denoising

Output:
  exp/image_samples/{output_folder}/
    ├── orig_0.png    - Original images
    ├── y0_0.png      - Degraded images
    └── 0_-1.png      - Restored images
        """
    )

    # DDRM parameters
    parser.add_argument('--deg', type=str, default='sr4',
                        help='Degradation type (default: sr4)')
    parser.add_argument('--sigma_0', type=float, default=0.05,
                        help='Noise level in degraded observation (default: 0.05)')
    parser.add_argument('--timesteps', type=int, default=20,
                        help='Number of diffusion timesteps (default: 20, max: 1000)')
    parser.add_argument('--eta', type=float, default=0.85,
                        help='Eta parameter for sampling (default: 0.85)')
    parser.add_argument('--etaB', type=float, default=1.0,
                        help='EtaB parameter for sampling (default: 1.0)')

    # Data parameters
    parser.add_argument('--subset_start', type=int, default=0,
                        help='Start index for image subset (default: 0)')
    parser.add_argument('--subset_end', type=int, default=5,
                        help='End index for image subset (default: 5)')
    parser.add_argument('-i', '--output_folder', type=str, default='config1_custom_ddpm',
                        help='Output folder name (default: config1_custom_ddpm)')

    # Advanced options
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print command without executing')

    args = parser.parse_args()

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Build command
    cmd = [
        'python', 'main.py',
        '--ni',  # Non-interactive mode
        '--config', 'celeba_hq.yml',
        '--doc', 'celeba',
        '--timesteps', str(args.timesteps),
        '--eta', str(args.eta),
        '--etaB', str(args.etaB),
        '--deg', args.deg,
        '--sigma_0', str(args.sigma_0),
        '-i', args.output_folder,
        '--subset_start', str(args.subset_start),
        '--subset_end', str(args.subset_end),
    ]

    # Display configuration
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║       Configuration 1: Custom DDPM (CelebA-HQ)               ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    print("📋 Model Configuration:")
    print("  • Model Type: Custom DDPM (type='simple')")
    print("  • Config File: configs/celeba_hq.yml")
    print("  • Source: models/diffusion.py::Model")
    print("  • Parameters: ~60M")
    print("  • Base Channels: 128")
    print("  • Attention: 1-head @ 1 level (16x16)")
    print("  • Classifier: ❌ None")
    print("  • FP16: ❌ No")
    print("  • Dataset: CelebA-HQ (face images)")
    print()
    print("🎯 Task Configuration:")
    print(f"  • Degradation: {args.deg}")
    print(f"  • Noise Level (σ₀): {args.sigma_0}")
    print(f"  • Timesteps: {args.timesteps}")
    print(f"  • Eta (η): {args.eta}")
    print(f"  • EtaB (ηB): {args.etaB}")
    print(f"  • Images: {args.subset_end - args.subset_start} (idx {args.subset_start}-{args.subset_end})")
    print()
    print("📂 Output:")
    print(f"  • Folder: exp/image_samples/{args.output_folder}/")
    print(f"  • Files: orig_*.png, y0_*.png, *_-1.png")
    print()
    print("💻 Command:")
    print(f"  {' '.join(cmd)}")
    print()

    if args.dry_run:
        print("🔍 Dry run mode - command not executed")
        return 0

    # Confirm execution
    print("─" * 67)
    response = input("▶️  Run this configuration? [Y/n]: ").strip().lower()
    if response and response not in ('y', 'yes'):
        print("❌ Cancelled by user")
        return 1

    print()
    print("🚀 Starting Configuration 1 (Custom DDPM)...")
    print("=" * 67)
    print()

    # Execute
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("=" * 67)
        print("✅ Configuration 1 completed successfully!")
        print(f"📁 Results saved to: exp/image_samples/{args.output_folder}/")
        return result.returncode

    except subprocess.CalledProcessError as e:
        print()
        print("=" * 67)
        print(f"❌ Configuration 1 failed with exit code {e.returncode}")
        print()
        print("Common issues:")
        print("  1. CelebA-HQ model not downloaded (see docs/CELEBA_DOWNLOAD_FIX.md)")
        print("  2. Missing dataset images in exp/datasets/")
        print("  3. CUDA out of memory (try reducing --subset_end)")
        print("  4. Missing dependencies (check environment.yml)")
        return e.returncode

    except KeyboardInterrupt:
        print()
        print("⚠️  Interrupted by user")
        return 130


if __name__ == '__main__':
    sys.exit(main())

