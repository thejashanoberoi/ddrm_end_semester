#!/usr/bin/env python3
"""
Configuration 2: OpenAI Guided Diffusion (Unconditional)
=========================================================

This script runs the OpenAI UNet model WITHOUT classifier for
unconditional image restoration on ImageNet.

Model Details:
- Type: OpenAI UNet (type='openai', class_cond=false)
- Architecture: OpenAI U-Net (~280M parameters)
- Base Channels: 256
- Attention: 4-heads @ 3 levels [32,16,8]
- Dataset: ImageNet
- Classifier: None
- FP16: Yes
- Independence: Fully standalone

Usage:
    python run_config2_openai_uncond.py [options]

Example:
    # Uniform deblurring
    python run_config2_openai_uncond.py --deg deblur_uni --sigma_0 0.0
    
    # 4x super-resolution
    python run_config2_openai_uncond.py --deg sr4 --sigma_0 0.05
    
    # Fast inference (fewer timesteps)
    python run_config2_openai_uncond.py --deg sr4 --timesteps 10

Static Analysis Reference:
- Config: configs/imagenet_256.yml (class_cond=false)
- Model: guided_diffusion/unet.py::UNetModel
- Loading: runners/diffusion.py:120-141
- Sampling: functions/denoising.py:52 (cls_fn=None branch)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_requirements():
    """Check if required files exist."""
    required_files = {
        'main.py': 'DDRM main script',
        'configs/imagenet_256.yml': 'ImageNet unconditional config',
        'guided_diffusion/unet.py': 'OpenAI UNet model',
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

    # Check model checkpoint
    model_path = Path("exp/logs/imagenet/256x256_diffusion_uncond.pt")
    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024**3)
        print(f"\n✅ Model checkpoint found: {size_gb:.2f} GB")
    else:
        print(f"\n❌ Model checkpoint not found:")
        print(f"   {model_path}")
        print("\n📥 The model will be downloaded automatically:")
        print("   Source: OpenAI Azure Blob Storage")
        print("   Size: ~2.1 GB")
        print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run Configuration 2: OpenAI UNet (Unconditional)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Uniform deblurring
  %(prog)s --deg deblur_uni --sigma_0 0.0
  
  # 4x super-resolution with noise
  %(prog)s --deg sr4 --sigma_0 0.05 --timesteps 40
  
  # 8x super-resolution (high quality)
  %(prog)s --deg sr8 --sigma_0 0.0 --timesteps 100

Key Features:
  ✓ Large-scale OpenAI UNet (280M parameters)
  ✓ FP16 precision (2× faster than FP32)
  ✓ Multi-head attention (4 heads @ 3 levels)
  ✓ No classifier dependency (standalone)
  ✓ Fast inference

Supported Degradations:
  sr2, sr4, sr8, sr16       - Super-resolution
  sr_bicubic4, sr_bicubic8  - Bicubic SR
  deblur_uni, deblur_gauss  - Deblurring
  inp, inp_lolcat           - Inpainting
  color, deno, cs2, cs4     - Other tasks

Output:
  exp/image_samples/{output_folder}/
    ├── orig_0.png    - Original images
    ├── y0_0.png      - Degraded images
    └── 0_-1.png      - Restored images
        """
    )

    # DDRM parameters
    parser.add_argument('--deg', type=str, default='deblur_uni',
                        help='Degradation type (default: deblur_uni)')
    parser.add_argument('--sigma_0', type=float, default=0.0,
                        help='Noise level (default: 0.0)')
    parser.add_argument('--timesteps', type=int, default=20,
                        help='Number of timesteps (default: 20)')
    parser.add_argument('--eta', type=float, default=0.85,
                        help='Eta parameter (default: 0.85)')
    parser.add_argument('--etaB', type=float, default=1.0,
                        help='EtaB parameter (default: 1.0)')

    # Data parameters
    parser.add_argument('--subset_start', type=int, default=0,
                        help='Start index (default: 0)')
    parser.add_argument('--subset_end', type=int, default=5,
                        help='End index (default: 5)')
    parser.add_argument('-i', '--output_folder', type=str, default='config2_openai_uncond',
                        help='Output folder (default: config2_openai_uncond)')

    # Options
    parser.add_argument('--dry-run', action='store_true',
                        help='Print command without executing')

    args = parser.parse_args()

    # Check requirements
    if not check_requirements():
        sys.exit(1)

    # Build command
    cmd = [
        'python', 'main.py', '--ni',
        '--config', 'imagenet_256.yml',  # Unconditional config
        '--doc', 'imagenet',
        '--timesteps', str(args.timesteps),
        '--eta', str(args.eta),
        '--etaB', str(args.etaB),
        '--deg', args.deg,
        '--sigma_0', str(args.sigma_0),
        '-i', args.output_folder,
        '--subset_start', str(args.subset_start),
        '--subset_end', str(args.subset_end),
    ]

    # Display
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║     Configuration 2: OpenAI UNet (Unconditional)             ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    print("📋 Model: OpenAI U-Net (~280M params, 4-heads @ 3 levels)")
    print(f"🎯 Task: {args.deg} | Noise: {args.sigma_0} | Steps: {args.timesteps}")
    print(f"📂 Output: exp/image_samples/{args.output_folder}/")
    print(f"💻 Command: {' '.join(cmd)}")
    print()

    if args.dry_run:
        print("🔍 Dry run - not executed")
        return 0

    response = input("▶️  Run? [Y/n]: ").strip().lower()
    if response and response not in ('y', 'yes'):
        print("❌ Cancelled")
        return 1

    print("\n🚀 Starting Configuration 2...")
    print("=" * 67)

    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 67)
        print("✅ Completed!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        return 130


if __name__ == '__main__':
    sys.exit(main())

