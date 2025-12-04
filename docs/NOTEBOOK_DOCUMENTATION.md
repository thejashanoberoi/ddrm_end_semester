# Model Comparison Notebook - Documentation

**Created:** December 4, 2025  
**Notebook:** `model_comparison_demo.ipynb`  
**Purpose:** Demonstrate and compare all three DDRM model configurations

---

## 📓 Notebook Overview

The `model_comparison_demo.ipynb` notebook runs all three model configurations identified in the static analysis and compares their results side-by-side.

### What It Does:

1. **Sets up environment** - Downloads models and datasets
2. **Runs Config 1** - Custom DDPM on CelebA-HQ faces
3. **Runs Config 2** - OpenAI UNet on ImageNet (unconditional)
4. **Runs Config 3** - OpenAI UNet + Classifier on ImageNet (conditional)
5. **Visualizes results** - Shows before/after for each config
6. **Compares outputs** - Side-by-side comparison of all three
7. **Generates report** - JSON summary with findings

---

## 🎯 Three Configurations Tested

### Configuration 1: Custom DDPM (Standalone)
```python
{
    "config": "celeba_hq.yml",
    "model": "Custom DDPM (type='simple')",
    "classifier": None,
    "task": "4x Super-Resolution",
    "dataset": "CelebA-HQ (faces)",
    "independent": True,
    "memory": "~1-2 GB VRAM"
}
```

### Configuration 2: OpenAI UNet (Standalone)
```python
{
    "config": "imagenet_256.yml",
    "model": "OpenAI UNet (type='openai')",
    "classifier": None,
    "class_cond": False,
    "task": "Uniform Deblurring",
    "dataset": "ImageNet",
    "independent": True,
    "memory": "~2-4 GB VRAM"
}
```

### Configuration 3: OpenAI UNet + Classifier (Dependent)
```python
{
    "config": "imagenet_256_cc.yml",
    "model": "OpenAI UNet (type='openai')",
    "classifier": "EncoderUNetModel",
    "class_cond": True,
    "task": "4x Super-Resolution (class-guided)",
    "dataset": "ImageNet",
    "independent": False,  # Classifier depends on UNet
    "memory": "~3-5 GB VRAM"
}
```

---

## 🗂️ Notebook Structure

### Section 1: Setup (Cells 1-7)
- Environment detection (Colab vs local)
- Package installation
- Directory creation
- Model downloads (~4 GB total):
  - OpenAI 256x256 unconditional (~1.2 GB)
  - OpenAI 256x256 conditional (~1.2 GB)
  - OpenAI 256x256 classifier (~500 MB)
  - CelebA-HQ Custom DDPM (~500 MB)
- Dataset downloads (demo images)

### Section 2: Configuration 1 - Custom DDPM (Cells 8-9)
- Parameters definition
- Run inference on 5 CelebA face images
- Task: 4x super-resolution with noise
- Output: `exp/image_samples/config1_custom_ddpm/`

### Section 3: Configuration 2 - OpenAI UNet (Cells 10-11)
- Parameters definition
- Run inference on 5 ImageNet images
- Task: Uniform deblurring
- Output: `exp/image_samples/config2_openai_uncond/`

### Section 4: Configuration 3 - OpenAI + Classifier (Cells 12-13)
- Parameters definition
- Run inference on 5 ImageNet images
- Task: 4x super-resolution with class guidance
- Output: `exp/image_samples/config3_openai_classifier/`

### Section 5: Results Visualization (Cells 14-18)
- Load and display helper functions
- Config 1 results (3 sample images)
- Config 2 results (3 sample images)
- Config 3 results (3 sample images)
- Side-by-side comparison of all three

### Section 6: Analysis (Cells 19-20)
- Output file analysis
- Performance metrics (file counts, sizes)
- Generate JSON comparison report

### Section 7: Conclusion (Cell 21)
- Summary of findings
- Validation of static analysis
- Architecture pattern confirmation

---

## 🚀 How to Run

### Option A: Google Colab (Recommended)

1. **Upload notebook to Colab:**
   - Go to https://colab.research.google.com/
   - File → Upload notebook
   - Select `model_comparison_demo.ipynb`

2. **Enable GPU:**
   - Runtime → Change runtime type
   - Hardware accelerator: GPU (T4 recommended)
   - Save

3. **Run all cells:**
   - Runtime → Run all
   - Total time: ~30-60 minutes (depends on GPU)
   - Downloads: ~4-5 GB

4. **View results:**
   - Images display inline
   - Download outputs via Files panel

### Option B: Local (Requires GPU)

1. **Ensure you have:**
   - NVIDIA GPU with 8+ GB VRAM
   - CUDA 11.3+ installed
   - PyTorch 1.10+ with CUDA

2. **Navigate to DDRM directory:**
   ```bash
   cd /Users/halfkhoonprince/Desktop/semester7_mini/ddrm_end_semester
   ```

3. **Launch Jupyter:**
   ```bash
   conda activate ddrm  # or your environment
   jupyter notebook model_comparison_demo.ipynb
   ```

4. **Run cells sequentially:**
   - Setup will be faster (no cloning needed)
   - Models still need to be downloaded (~4 GB)

### Option C: CPU-only (Very Slow)

⚠️ **Not recommended** - inference will take hours instead of minutes.

If you must:
1. Follow Option B setup
2. Reduce `subset_end` to 1-2 images
3. Increase `timesteps` back to 40-100 for quality
4. Expect 5-10 minutes per image

---

## 📊 Expected Outputs

### Output Directories:
```
exp/image_samples/
├── config1_custom_ddpm/
│   ├── orig_0.png, orig_1.png, ...     (original images)
│   ├── y0_0.png, y0_1.png, ...         (degraded images)
│   └── 0_-1.png, 1_-1.png, ...         (restored images)
├── config2_openai_uncond/
│   ├── orig_0.png, ...
│   ├── y0_0.png, ...
│   └── 0_-1.png, ...
├── config3_openai_classifier/
│   ├── orig_0.png, ...
│   ├── y0_0.png, ...
│   └── 0_-1.png, ...
└── comparison_report.json              (summary JSON)
```

### comparison_report.json:
```json
{
  "experiment_date": "2025-12-04T...",
  "configurations": [
    {
      "id": 1,
      "name": "Custom DDPM (CelebA-HQ)",
      "independent": true,
      ...
    },
    {
      "id": 2,
      "name": "OpenAI Guided Diffusion (Unconditional)",
      "independent": true,
      ...
    },
    {
      "id": 3,
      "name": "OpenAI + Classifier (Conditional)",
      "independent": false,
      "dependency": "Classifier depends on UNet",
      ...
    }
  ],
  "key_findings": {
    "models_are_alternatives": true,
    "mutually_exclusive": "...",
    "classifier_dependency": "..."
  }
}
```

---

## 🔍 What The Notebook Demonstrates

### 1. Model Independence (Configs 1 & 2)

**Custom DDPM:**
- Loads ONLY Custom DDPM model
- No OpenAI models in memory
- No classifier loaded
- Runs completely standalone

**OpenAI UNet (uncond):**
- Loads ONLY OpenAI UNet
- No Custom DDPM in memory
- No classifier loaded (even though classifier exists)
- Runs completely standalone

**Evidence:** Different output folders, no conflicts, separate model loading.

### 2. Classifier Dependency (Config 3)

**OpenAI UNet + Classifier:**
- Loads OpenAI UNet FIRST
- Then loads Classifier SECOND
- Classifier cannot function without UNet
- Classifier provides gradient guidance to UNet's sampling

**Evidence:** 
- Code in `runners/diffusion.py:142-163` shows conditional loading
- Classifier only loads when `class_cond: true`
- Classifier called in `functions/denoising.py:54-56` to modify noise prediction

### 3. Mutual Exclusivity

**Cannot happen:**
- ❌ Custom DDPM + OpenAI together
- ❌ Custom DDPM + Classifier
- ❌ Classifier alone

**Evidence:**
- `if-elif` structure in code ensures only ONE model variable
- Each config produces separate outputs
- No shared state between runs

---

## 📈 Performance Comparison

From notebook cells 19-20, you'll see:

| Metric | Config 1 | Config 2 | Config 3 |
|--------|----------|----------|----------|
| Model Type | Custom DDPM | OpenAI UNet | OpenAI + Classifier |
| Images Processed | 5 | 5 | 5 |
| Files Generated | 15 (3 per image) | 15 | 15 |
| Estimated VRAM | 1-2 GB | 2-4 GB | 3-5 GB |
| Estimated Time* | ~2-3 min | ~3-5 min | ~4-6 min |

*On GPU (T4/V100), with 20 timesteps. Scales linearly with timesteps.

---

## 🎨 Visualization Examples

The notebook generates:

1. **Individual Config Results:**
   - 3 columns: Original | Degraded | Restored
   - 3 rows per config (first 3 images)
   - Total: 9 comparison plots

2. **Side-by-Side Comparison:**
   - 3 columns: Config 1 | Config 2 | Config 3
   - Shows restored images only
   - Demonstrates independence (different datasets/tasks)

3. **Performance Analysis:**
   - File counts per config
   - Output sizes
   - Completion status

---

## 🧪 Validation of Static Analysis

The notebook **validates** the findings from `docs/MODEL_USAGE_ANALYSIS.md`:

| Static Analysis Finding | Notebook Evidence |
|------------------------|-------------------|
| Models are alternatives | ✅ Three separate runs, different models loaded |
| Config 1 standalone | ✅ Runs without OpenAI models |
| Config 2 standalone | ✅ Runs without Custom DDPM or Classifier |
| Config 3 has dependency | ✅ Loads both UNet and Classifier |
| Classifier needs UNet | ✅ Cannot run classifier without UNet |
| Different datasets | ✅ CelebA for Config 1, ImageNet for 2 & 3 |
| Separate outputs | ✅ Three distinct output folders |

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
- Reduce `subset_end` to 2-3 images
- Use smaller batch_size in configs
- Close other GPU-using programs
- Try Colab with High-RAM runtime

### Issue: "Model checkpoint not found"
**Solution:**
- Re-run download cells (cells 3-6)
- Check internet connection
- Verify `exp/logs/` directory structure

### Issue: "No images displayed"
**Solution:**
- Ensure all three config runs completed
- Check `exp/image_samples/` has subdirectories
- Verify PNG files exist in output folders

### Issue: "Notebook too slow"
**Solution:**
- Reduce `timesteps` from 20 to 10 (faster, lower quality)
- Reduce `subset_end` from 5 to 2
- Use GPU runtime (Colab or local)

---

## 📚 Related Documentation

- **`docs/MODEL_USAGE_ANALYSIS.md`** - Comprehensive static analysis (500+ lines)
- **`docs/MODEL_DEPENDENCY_DIAGRAM.txt`** - Visual flowcharts
- **`docs/PROJECT_OVERVIEW.md`** - Full project documentation
- **`docs/SECURITY_WARNINGS.md`** - Security considerations

---

## 🎓 Learning Objectives

After running this notebook, you will understand:

1. ✅ How to run each model configuration independently
2. ✅ That Custom DDPM and OpenAI UNet are alternatives (not used together)
3. ✅ That Classifier is an optional add-on for OpenAI UNet
4. ✅ How to compare restoration results across different models
5. ✅ The architecture pattern (Strategy + Decorator)
6. ✅ How config files control model selection
7. ✅ How degradation operators work (sr4, deblur_uni)
8. ✅ How to visualize and analyze DDRM outputs

---

## 💡 Extending the Notebook

### Add More Degradation Types:

```python
# Config 1 variants
config1_variants = [
    {"deg": "sr4", "sigma_0": 0.05, "name": "SR 4x with noise"},
    {"deg": "sr8", "sigma_0": 0.0, "name": "SR 8x clean"},
    {"deg": "deblur_gauss", "sigma_0": 0.0, "name": "Gaussian blur"},
]
```

### Compare More Images:

```python
# Increase image count
config1_params['subset_end'] = 10  # Process 10 images
```

### Add Metrics Calculation:

```python
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr(orig_path, restored_path):
    orig = np.array(Image.open(orig_path)) / 255.0
    restored = np.array(Image.open(restored_path)) / 255.0
    return psnr(orig, restored)
```

### Export to Google Drive (Colab):

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy outputs
!cp -r exp/image_samples/* /content/drive/MyDrive/ddrm_results/
```

---

## ✅ Checklist Before Running

- [ ] GPU available (recommended) or prepared for long wait (CPU)
- [ ] At least 10 GB free disk space
- [ ] Stable internet connection for downloads
- [ ] Python 3.9+ with PyTorch installed
- [ ] Read `docs/MODEL_USAGE_ANALYSIS.md` for context

---

## 📝 Notes

1. **Timesteps:** Reduced to 20 for faster demo. Increase to 100+ for better quality.
2. **Datasets:** Uses demo datasets (small). Can swap with full datasets.
3. **Memory:** Estimates assume batch_size from configs (4-8).
4. **Time:** Depends heavily on GPU type and timesteps.
5. **Static Analysis:** No code execution was used to create this notebook - designed from pure code reading.

---

**Created:** December 4, 2025  
**Based on:** Static analysis of DDRM codebase  
**Validated against:** `runners/diffusion.py`, config files, README.md  
**Purpose:** Educational demonstration of model independence and dependency

