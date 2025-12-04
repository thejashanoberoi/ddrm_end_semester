#!/usr/bin/env python3
"""
Example: Using compute_psnr.py with DDRM outputs
=================================================

This script demonstrates various ways to use compute_psnr.py for
analyzing DDRM restoration results.

Based on static analysis - shows expected usage patterns.
"""

import subprocess
import json
from pathlib import Path


def example_1_basic():
    """Example 1: Basic PSNR computation"""
    print("=" * 70)
    print("Example 1: Basic PSNR Computation")
    print("=" * 70)
    print()
    print("Scenario: Compute PSNR for a deblurring experiment")
    print()
    print("Command:")
    print("  python compute_psnr.py --input-folder exp/image_samples/demo_deblur")
    print()
    print("Expected output:")
    print("  - Console: Per-image PSNR values + statistics")
    print("  - File: psnr_results.json")
    print()
    print("Sample console output:")
    print("""
📁 Input folder: exp/image_samples/demo_deblur
🔍 Discovering images...
   Found 5 image sets

📊 Computing PSNR...
Index    PSNR (orig vs restored)   PSNR (orig vs blurred)
=================================================================
0        28.5432 dB                22.1234 dB
1        29.1234 dB                22.5678 dB
2        27.8765 dB                21.9876 dB
3        30.2345 dB                23.1234 dB
4        28.9876 dB                22.3456 dB
=================================================================

📈 Aggregate Statistics:
   Mean PSNR:   28.9530 dB
   Std PSNR:    0.8234 dB
   Median PSNR: 28.9876 dB
   Min PSNR:    27.8765 dB
   Max PSNR:    30.2345 dB

✅ PSNR computation completed successfully!
    """)


def example_2_custom_output():
    """Example 2: Custom output files"""
    print("=" * 70)
    print("Example 2: Custom Output Files")
    print("=" * 70)
    print()
    print("Scenario: Save results to specific JSON and CSV files")
    print()
    print("Command:")
    print("""
  python compute_psnr.py \\
      --input-folder exp/image_samples/demo_sr4 \\
      --output-json results/sr4_metrics.json \\
      --output-csv results/sr4_metrics.csv
    """)
    print()
    print("Expected output files:")
    print("  - results/sr4_metrics.json (detailed)")
    print("  - results/sr4_metrics.csv (tabular)")
    print()


def example_3_compare_configs():
    """Example 3: Compare three configurations"""
    print("=" * 70)
    print("Example 3: Compare Three Model Configurations")
    print("=" * 70)
    print()
    print("Scenario: Compare PSNR across Config 1, 2, and 3")
    print()
    print("Step 1: Run all three configurations")
    print("  python run_all_configs.py --timesteps 20 --subset_end 5")
    print()
    print("Step 2: Compute PSNR for each")
    print("""
  python compute_psnr.py --input-folder exp/image_samples/compare_config1 \\
      --output-json config1_psnr.json --no-verbose
  
  python compute_psnr.py --input-folder exp/image_samples/compare_config2 \\
      --output-json config2_psnr.json --no-verbose
  
  python compute_psnr.py --input-folder exp/image_samples/compare_config3 \\
      --output-json config3_psnr.json --no-verbose
    """)
    print()
    print("Step 3: Compare results with Python")
    print("""
  python << 'EOF'
import json

configs = {
    'Config 1 (Custom DDPM)': 'config1_psnr.json',
    'Config 2 (OpenAI UNet)': 'config2_psnr.json',
    'Config 3 (OpenAI + Classifier)': 'config3_psnr.json',
}

print("Configuration Comparison:\\n")
print(f"{'Configuration':<30} {'Mean PSNR':<12} {'Std PSNR':<12}")
print("=" * 60)

for name, file in configs.items():
    with open(file) as f:
        data = json.load(f)
        agg = data['aggregate']['psnr_orig_restored']
        print(f"{name:<30} {agg['mean']:>10.2f} dB  {agg['std']:>10.2f} dB")
EOF
    """)
    print()
    print("Expected output:")
    print("""
Configuration Comparison:

Configuration                  Mean PSNR    Std PSNR
============================================================
Config 1 (Custom DDPM)             27.85 dB        1.23 dB
Config 2 (OpenAI UNet)             29.12 dB        0.98 dB
Config 3 (OpenAI + Classifier)     29.87 dB        0.87 dB
    """)


def example_4_batch_processing():
    """Example 4: Batch processing multiple experiments"""
    print("=" * 70)
    print("Example 4: Batch Processing Multiple Experiments")
    print("=" * 70)
    print()
    print("Scenario: Process all output folders at once")
    print()
    print("Bash script:")
    print("""
#!/bin/bash
# File: batch_psnr.sh

mkdir -p psnr_results

for folder in exp/image_samples/*/; do
    basename=$(basename "$folder")
    
    # Skip if no image files
    if ! ls "$folder"/*.png &> /dev/null; then
        continue
    fi
    
    echo "Processing $basename..."
    
    python compute_psnr.py \\
        --input-folder "$folder" \\
        --output-json "psnr_results/${basename}.json" \\
        --output-csv "psnr_results/${basename}.csv" \\
        --no-verbose
done

echo "✅ Batch processing complete!"
echo "Results in: psnr_results/"
    """)
    print()
    print("Usage:")
    print("  chmod +x batch_psnr.sh")
    print("  ./batch_psnr.sh")
    print()


def example_5_python_api():
    """Example 5: Using as Python API"""
    print("=" * 70)
    print("Example 5: Using as Python API")
    print("=" * 70)
    print()
    print("Scenario: Import and use in custom analysis scripts")
    print()
    print("Python script:")
    print("""
from compute_psnr import compute_all_psnr, PSNRComputer
from pathlib import Path
import numpy as np

# Method 1: High-level API
results = compute_all_psnr(
    input_folder=Path('exp/image_samples/demo_deblur'),
    eps=1e-12,
    clip=True,
    verbose=True
)

# Access aggregate stats
mean_psnr = results['aggregate']['psnr_orig_restored']['mean']
print(f"Average PSNR: {mean_psnr:.2f} dB")

# Access per-image results
for entry in results['per_image']:
    idx = entry['idx']
    psnr = entry['psnr_orig_restored']
    print(f"Image {idx}: {psnr:.2f} dB")

# Find best and worst restorations
psnr_values = [e['psnr_orig_restored'] for e in results['per_image']]
best_idx = np.argmax(psnr_values)
worst_idx = np.argmin(psnr_values)

print(f"\\nBest restoration: Image {best_idx} ({psnr_values[best_idx]:.2f} dB)")
print(f"Worst restoration: Image {worst_idx} ({psnr_values[worst_idx]:.2f} dB)")

# Method 2: Low-level API (for custom workflows)
computer = PSNRComputer(eps=1e-12, clip=True, verbose=False)

img1 = computer.load_image(Path('exp/image_samples/demo_deblur/orig_0.png'))
img2 = computer.load_image(Path('exp/image_samples/demo_deblur/0_-1.png'))

if img1 is not None and img2 is not None:
    psnr = computer.compute_psnr_pair(img1, img2)
    print(f"\\nDirect PSNR computation: {psnr:.2f} dB")
    """)
    print()


def example_6_troubleshooting():
    """Example 6: Troubleshooting missing files"""
    print("=" * 70)
    print("Example 6: Troubleshooting Missing Files")
    print("=" * 70)
    print()
    print("Scenario: Some images missing, need diagnostics")
    print()
    print("Command:")
    print("  python compute_psnr.py --input-folder exp/image_samples/incomplete --verbose")
    print()
    print("Expected output with warnings:")
    print("""
📁 Input folder: exp/image_samples/incomplete
🔍 Discovering images...
   Found 5 image sets

UserWarning: Index 1: Missing restored image
UserWarning: Index 3: Missing original image

⚠️  Missing files:
   Original: 1
   Restored: 1
   Blurred: 0 (optional)
   Processing 3/5 image sets

📊 Computing PSNR...
Index    PSNR (orig vs restored)   PSNR (orig vs blurred)
=================================================================
0        28.5432 dB                22.1234 dB
2        27.8765 dB                21.9876 dB
4        28.9876 dB                22.3456 dB
=================================================================

📈 Aggregate Statistics (Original vs Restored):
   Mean PSNR:   28.4691 dB
   ...
   Samples:     3 finite, 0 infinite

✅ PSNR computation completed successfully!
    """)
    print()
    print("Check JSON output for details:")
    print("""
{
  "n_samples": 3,
  "per_image": [ ... ],  // Only 3 entries (0, 2, 4)
  "skipped": []
}
    """)


def example_7_integration():
    """Example 7: Full DDRM + PSNR workflow"""
    print("=" * 70)
    print("Example 7: Complete DDRM + PSNR Workflow")
    print("=" * 70)
    print()
    print("Scenario: End-to-end restoration and evaluation")
    print()
    print("Step 1: Run DDRM restoration")
    print("""
  python main.py --ni --config imagenet_256.yml --doc imagenet \\
      --timesteps 40 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05 \\
      -i experiment_sr4_t40 --subset_start 0 --subset_end 20
    """)
    print()
    print("Step 2: Compute PSNR")
    print("""
  python compute_psnr.py \\
      --input-folder exp/image_samples/experiment_sr4_t40 \\
      --output-json results/sr4_t40_metrics.json \\
      --output-csv results/sr4_t40_metrics.csv
    """)
    print()
    print("Step 3: Analyze and visualize")
    print("""
  python << 'EOF'
import json
import matplotlib.pyplot as plt

# Load results
with open('results/sr4_t40_metrics.json') as f:
    data = json.load(f)

# Extract PSNR values
indices = [e['idx'] for e in data['per_image']]
psnr_values = [e['psnr_orig_restored'] for e in data['per_image']]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(indices, psnr_values, 'o-', linewidth=2, markersize=8)
plt.axhline(data['aggregate']['psnr_orig_restored']['mean'], 
            color='r', linestyle='--', label='Mean')
plt.xlabel('Image Index')
plt.ylabel('PSNR (dB)')
plt.title('4x Super-Resolution Quality (40 timesteps)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('results/psnr_plot.png', dpi=150, bbox_inches='tight')
print("Plot saved to: results/psnr_plot.png")
EOF
    """)


def main():
    """Display all examples"""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║        compute_psnr.py - Usage Examples                          ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print("\n")

    examples = [
        example_1_basic,
        example_2_custom_output,
        example_3_compare_configs,
        example_4_batch_processing,
        example_5_python_api,
        example_6_troubleshooting,
        example_7_integration,
    ]

    for i, example_func in enumerate(examples, 1):
        print()
        example_func()
        print()
        if i < len(examples):
            input("Press Enter for next example...")
            print("\n" * 2)

    print("=" * 70)
    print("For more information, see:")
    print("  - docs/COMPUTE_PSNR_GUIDE.md (full documentation)")
    print("  - python compute_psnr.py --help (command-line help)")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()

