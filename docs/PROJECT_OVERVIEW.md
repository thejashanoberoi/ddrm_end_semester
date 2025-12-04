# DDRM (Denoising Diffusion Restoration Models) - Project Overview

**Generated:** December 4, 2025  
**Analysis Type:** Static code analysis (no execution)  
**Entry Point:** Untitled5.ipynb

---

## 1. Title & Short Description

**Denoising Diffusion Restoration Models (DDRM)** is a research implementation that uses pre-trained Denoising Diffusion Probabilistic Models (DDPMs) to solve general linear inverse problems in image restoration. The method works efficiently without requiring problem-specific supervised training. It supports various degradation types including super-resolution, deblurring, inpainting, colorization, and compressive sensing.

**Paper:** [arXiv:2201.11793](https://arxiv.org/abs/2201.11793)  
**Authors:** Bahjat Kawar, Michael Elad (Technion), Stefano Ermon, Jiaming Song (Stanford)

---

## 2. Repository Entry Point & Execution Flow

### Primary Entry Point: `Untitled5.ipynb`

The Jupyter notebook demonstrates a Google Colab workflow for running DDRM experiments. Based on static analysis, the notebook performs the following sequence:

#### **Cell 1-2: Environment Setup (Repeated)**
```bash
# Clone the DDRM repository
!rm -rf ddrm
!git clone https://github.com/bahjat-kawar/ddrm.git
%cd ddrm

# Install minimal dependencies
!pip -q install pyyaml tqdm pillow numpy scipy einops matplotlib
```

#### **Cell 3: Download Pre-trained Models**
```bash
# Create directory structure
!mkdir -p exp/logs/imagenet

# Download OpenAI guided-diffusion models
BASE="https://openaipublic.blob.core.windows.net/diffusion/jul-2021"
!wget -nc "$BASE/256x256_diffusion_uncond.pt" -P exp/logs/imagenet
!wget -nc "$BASE/256x256_diffusion.pt" -P exp/logs/imagenet
!wget -nc "$BASE/256x256_classifier.pt" -P exp/logs/imagenet
!wget -nc "$BASE/512x512_diffusion.pt" -P exp/logs/imagenet
!wget -nc "$BASE/512x512_classifier.pt" -P exp/logs/imagenet
```

#### **Cell 4: Download Demo Datasets**
```bash
!mkdir -p exp/datasets
!git clone https://github.com/jiamings/ddrm-exp-datasets.git
!cp -r ddrm-exp-datasets/* exp/datasets/
```

#### **Cell 5: Download ImageNet Validation List**
```bash
wget -O exp/imagenet_val_1k.txt \
  https://raw.githubusercontent.com/XingangPan/deep-generative-prior/master/scripts/imagenet_val_1k.txt
```

#### **Cell 6: Verify Dataset Structure**
```bash
# Recursively find and count images
find exp/datasets -type f -iregex '.*\.\(jpg\|jpeg\|png\)$' | wc -l
```

#### **Cell 7: Prepare Custom ImageNet Dataset**
```python
# Python script that:
# 1. Collects images from ood folders recursively
# 2. Converts up to 1000 images to JPEG
# 3. Writes filenames to exp/imagenet_val_1k.txt
```

#### **Cell 8: Run DDRM Inference (Multiple Degradations)**
```bash
# Uniform deblurring
python main.py --ni --config imagenet_256.yml --doc imagenet \
  --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_uni --sigma_0 0.0 -i demo_deblur

# 4x super-resolution with noise
python main.py --ni --config imagenet_256.yml --doc imagenet \
  --timesteps 40 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05 -i demo_sr4
```

#### **Cell 9-10: Additional Experiments**
```bash
# 4x super-resolution with 40 timesteps
python main.py --ni --config imagenet_256.yml --doc imagenet \
  --timesteps 40 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05 -i demo_sr4_iter40
```

#### **Cell 11-12: Export Results to Google Drive**
```python
# Mount Google Drive and copy results
from google.colab import drive
drive.mount('/content/drive')

# Copy entire outputs or specific experiment folders
shutil.copytree(src, dst, dirs_exist_ok=True)
```

### **Execution Flow Summary:**

1. **Setup** → Clone repository and install dependencies
2. **Download** → Fetch pre-trained diffusion models (256x256, 512x512)
3. **Data Prep** → Download/prepare demo datasets
4. **Inference** → Run DDRM with various degradation types
5. **Export** → Save results to Google Drive

### **Secondary Entry Point: `main.py`**

Command-line interface for running DDRM sampling:

```bash
python main.py --ni --config {CONFIG}.yml --doc {DATASET} \
  --timesteps {STEPS} --eta {ETA} --etaB {ETA_B} \
  --deg {DEGRADATION} --sigma_0 {SIGMA_0} -i {IMAGE_FOLDER}
```

**Key Parameters:**
- `--config`: YAML configuration file (e.g., `imagenet_256.yml`)
- `--doc`: Dataset name (determines checkpoint location)
- `--timesteps`: Number of diffusion steps (default: 1000)
- `--eta`: η parameter for sampling (default: 0.85)
- `--etaB`: η_b parameter for sampling (default: 1.0)
- `--deg`: Degradation type (see section 9 for full list)
- `--sigma_0`: Noise level in degraded observation
- `-i, --image_folder`: Output directory name

---

## 3. File-by-File Summary

### **Core Files**

#### `main.py` (172 lines)
Entry point script that parses command-line arguments, loads configurations, sets up logging, initializes the Diffusion runner, and executes sampling. Uses `argparse` for CLI and `yaml` for config parsing. Handles device selection (CUDA/CPU), random seeds, and directory management. **(main.py:1-172)**

#### `environment.yml` (137 lines)
Conda environment specification with PyTorch 1.10.1, CUDA 11.3, and dependencies. **WARNING:** Contains Linux-specific packages (`ld_impl_linux-64`, `libgcc-ng`) and GPU-only packages (`cudatoolkit=11.3.1`). Includes pip package `torch-fidelity==0.3.0`. **(environment.yml:1-137)**

#### `README.md`
Comprehensive documentation covering project overview, installation, usage examples, and citation information. Describes supported degradation types and provides command examples for various tasks.

### **Configuration Files (`configs/`)**

#### `configs/imagenet_256.yml`
Configuration for ImageNet 256x256 model using OpenAI's pretrained diffusion model. Sets `subset_1k: True`, uses FP16 precision, unconditional generation (`class_cond: false`). Batch size: 8. **(configs/imagenet_256.yml:1-46)**

#### `configs/celeba_hq.yml`
Configuration for CelebA-HQ using "simple" model type (DDPM-style). Uses out-of-distribution mode (`out_of_dist: True`). Batch size: 4. Model channels: 128 with multipliers [1,1,2,2,4,4]. **(configs/celeba_hq.yml:1-37)**

#### `configs/bedroom.yml`
LSUN Bedroom configuration. Similar to celeba_hq but includes training hyperparameters: Adam optimizer with lr=0.00002, batch_size=64 for training. **(configs/bedroom.yml:1-54)**

#### Other configs
- `cat.yml`: LSUN Cat dataset
- `church.yml`: LSUN Church dataset
- `imagenet_256_cc.yml`: ImageNet with class-conditional generation
- `imagenet_512_cc.yml`: 512x512 class-conditional ImageNet

### **Runners (`runners/`)**

#### `runners/diffusion.py` (340+ lines)
Main orchestration class `Diffusion` that:
- Initializes diffusion schedules (beta, alpha)
- Loads pretrained models (simple DDPM or OpenAI UNet)
- Downloads checkpoints if missing
- Implements `sample()` and `sample_sequence()` methods
- Handles dataset loading and batching
- Instantiates degradation operators (H_funcs)
- Computes PSNR metrics
- Saves original, degraded, and restored images

**Key Methods:**
- `__init__(args, config, device)`: Initialize betas, alphas, variance schedules **(diffusion.py:58-91)**
- `sample()`: Load model, setup classifier (if class-conditional) **(diffusion.py:93-165)**
- `sample_sequence(model, cls_fn)`: Main sampling loop over dataset **(diffusion.py:167-332)**
- `sample_image(x, model, H_funcs, y_0, sigma_0, ...)`: Call denoising algorithm **(diffusion.py:334-340)**

**Model Loading:**
- "simple" type: Loads from local checkpoints or downloads from Heidelberg servers
- "openai" type: Loads from `exp/logs/imagenet/` or downloads from OpenAI Azure blob storage

### **Functions (`functions/`)**

#### `functions/denoising.py` (107 lines)
Implements the core DDRM algorithm:

- **`compute_alpha(beta, t)`**: Computes cumulative product of (1-beta) **(denoising.py:6-9)**
- **`efficient_generalized_steps(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn, classes)`**: Main reverse diffusion sampling with SVD-based degradation correction. Implements Algorithm 1 from the paper. Handles three cases: missing pixels (etaC), less noisy than observation (etaA), noisier than observation (etaB). **(denoising.py:11-107)**

**Side Effects:** Saves x0 predictions and intermediate xt states to CPU, logs progress with tqdm.

#### `functions/svd_replacement.py` (470+ lines)
Defines abstract base class `H_functions` and concrete degradation operators that avoid explicit SVD computation:

**Base Class:** `H_functions` (abstract) **(svd_replacement.py:3-66)**
- Methods: `V()`, `Vt()`, `U()`, `Ut()`, `singulars()`, `add_zeros()`, `H()`, `Ht()`, `H_pinv()`

**Concrete Implementations:**
1. **`GeneralH`**: General matrix with explicit SVD (memory inefficient) **(svd_replacement.py:69-105)**
2. **`Inpainting`**: Mask-based inpainting **(svd_replacement.py:108-145)**
3. **`Denoising`**: Identity operator (all singulars = 1) **(svd_replacement.py:148-166)**
4. **`SuperResolution`**: Block-averaging super-resolution **(svd_replacement.py:169-220)**
5. **`Colorization`**: RGB to grayscale projection **(svd_replacement.py:223-264)**
6. **`WalshHadamardCS`**: Compressive sensing via Fast Walsh-Hadamard Transform **(svd_replacement.py:267-303)**
7. **`SRConv`**: Convolution-based super-resolution (bicubic kernels) **(svd_replacement.py:306-380)**
8. **`Deblurring`**: 1D blur kernels **(svd_replacement.py:383-440)**
9. **`Deblurring2D`**: Anisotropic 2D blur kernels **(svd_replacement.py:443-470+)**

#### `functions/ckpt_util.py` (75 lines)
Checkpoint downloading and verification utilities:

- **`download(url, local_path, chunk_size)`**: Downloads files with progress bar **(ckpt_util.py:38-48)**
- **`md5_hash(path)`**: Computes MD5 checksum **(ckpt_util.py:51-54)**
- **`get_ckpt_path(name, root, check, prefix)`**: Retrieves checkpoint path, downloads if missing, verifies MD5 **(ckpt_util.py:57-75)**

**URLs:** Heidelberg University storage for LSUN/CIFAR10 models **(ckpt_util.py:5-13)**

### **Models (`models/`)**

#### `models/diffusion.py` (250+ lines)
Custom DDPM model implementation:

**Helper Functions:**
- `get_timestep_embedding(timesteps, embedding_dim)`: Sinusoidal positional encoding **(diffusion.py:6-22)**
- `nonlinearity(x)`: Swish activation **(diffusion.py:25-27)**
- `Normalize(in_channels)`: GroupNorm with 32 groups **(diffusion.py:30-31)**

**Modules:**
- `Upsample`: 2x nearest-neighbor upsampling with optional conv **(diffusion.py:34-50)**
- `Downsample`: 2x downsampling with conv or avgpool **(diffusion.py:53-71)**
- `ResnetBlock`: Residual block with time embedding **(diffusion.py:74-137)**
- `AttnBlock`: Self-attention block **(diffusion.py:140-189)**
- `Model`: Full U-Net architecture with time conditioning **(diffusion.py:192+)**

**`Model` Class:**
- Inherits from `nn.Module`
- U-Net with encoder-decoder structure
- Time embedding via dense layers
- Attention at specified resolutions
- Configurable channel multipliers

### **Guided Diffusion (`guided_diffusion/`)**

OpenAI's guided-diffusion implementation (adapted):

#### `guided_diffusion/unet.py` (700+ lines)
- `UNetModel`: Main diffusion U-Net with attention, class conditioning, FP16 support **(unet.py:396+)**
- `SuperResModel`: Extends UNetModel for super-resolution **(unet.py:667+)**
- `EncoderUNetModel`: Encoder-only variant **(unet.py:684+)**

#### `guided_diffusion/script_util.py` (200+ lines)
- `create_model()`: Factory function for UNetModel with appropriate defaults **(script_util.py:127-186)**
- `create_classifier()`: Factory for classifier guidance models
- Various defaults functions for hyperparameters

#### `guided_diffusion/nn.py`
Custom layers and utilities (conv_nd, linear, zero_module, etc.)

#### `guided_diffusion/fp16_util.py`
Mixed-precision training utilities

#### `guided_diffusion/logger.py`
Logging utilities

### **Datasets (`datasets/`)**

#### `datasets/__init__.py` (200+ lines)
Main dataset factory and transforms:

- **`get_dataset(args, config)`**: Returns train/test datasets based on config **(datasets/__init__.py:46-188)**
  - Supports: CELEBA, LSUN, CelebA_HQ, FFHQ, ImageNet
  - Handles out-of-distribution mode
  - Applies transforms (resize, crop, flip, to_tensor)
  
- **`data_transform(config, X)`**: Apply dequantization **(datasets/__init__.py:195-199)**
- **`inverse_data_transform(config, X)`**: Rescale from [-1,1] to [0,1] **(datasets/__init__.py:207-212)**

**Side Effects:** Downloads datasets if not present (CelebA), reads from disk

#### `datasets/celeba.py`
Custom CelebA dataset loader (likely adapted from torchvision)

#### `datasets/lsun.py`
LSUN dataset loader for bedroom, church, cat categories

#### `datasets/imagenet_subset.py` (100+ lines)
Custom ImageNet subset loader:
- **`ImageDataset`**: Loads images from file list **(imagenet_subset.py:51-100)**
- Reads stem names from text file, appends `.jpeg` suffix
- Optional normalization to [-1, 1]

### **Input Masks (`inp_masks/`)**

Binary `.npy` files for inpainting experiments:
- `lolcat_extra.npy`: Inpainting mask (lolcat image)
- `lorem3.npy`: Inpainting mask (text)

---

## 4. Function and Class Reference

### **High-Level Call Graph**

```
main.py::main()
  └─> parse_args_and_config()
  └─> Diffusion(args, config)
      └─> Diffusion.sample()
          ├─> get_dataset(args, config)
          ├─> create_model() [if type='openai']
          ├─> Model(config) [if type='simple']
          ├─> download() [if checkpoint missing]
          └─> Diffusion.sample_sequence(model, cls_fn)
              ├─> data_transform()
              ├─> H_funcs.H()  # Create degraded observation
              ├─> Diffusion.sample_image()
              │   └─> efficient_generalized_steps()
              │       └─> model(xt, t) [denoising network]
              └─> inverse_data_transform()
```

### **Key Functions**

| Function | Signature | File:Line | Description | Callers | Side Effects | Confidence |
|----------|-----------|-----------|-------------|---------|--------------|------------|
| `main` | `()` | main.py:165-172 | Entry point, orchestrates argument parsing and runner execution | CLI | Logging, directory creation | HIGH |
| `parse_args_and_config` | `()` | main.py:17-152 | Parse CLI args and YAML config, setup logging and dirs | main() | File I/O, mkdir, logging setup | HIGH |
| `dict2namespace` | `(config: dict)` | main.py:155-162 | Convert nested dict to argparse.Namespace | parse_args_and_config() | None | HIGH |
| `get_dataset` | `(args, config)` | datasets/__init__.py:46-188 | Factory for train/test datasets | Diffusion.sample() | Disk I/O, downloads | HIGH |
| `data_transform` | `(config, X)` | datasets/__init__.py:195-199 | Apply dequantization to images | Diffusion.sample_sequence() | None | HIGH |
| `inverse_data_transform` | `(config, X)` | datasets/__init__.py:207-212 | Rescale from [-1,1] to [0,1] | Diffusion.sample_sequence() | None | HIGH |
| `efficient_generalized_steps` | `(x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, cls_fn, classes)` | functions/denoising.py:11-107 | Core DDRM reverse diffusion algorithm | Diffusion.sample_image() | Model inference, tqdm progress | HIGH |
| `compute_alpha` | `(beta, t)` | functions/denoising.py:6-9 | Compute cumulative alpha_t | efficient_generalized_steps() | None | HIGH |
| `get_ckpt_path` | `(name, root, check, prefix)` | functions/ckpt_util.py:57-75 | Get checkpoint path, download if needed | Diffusion.sample() | Network I/O, file writes | HIGH |
| `download` | `(url, local_path, chunk_size)` | functions/ckpt_util.py:38-48 | Download file with progress bar | get_ckpt_path(), Diffusion.sample() | Network I/O, file writes | HIGH |
| `get_beta_schedule` | `(beta_schedule, beta_start, beta_end, num_diffusion_timesteps)` | runners/diffusion.py:22-52 | Generate beta schedule for diffusion | Diffusion.__init__() | None | HIGH |
| `create_model` | `(image_size, num_channels, ...)` | guided_diffusion/script_util.py:127-186 | Factory for OpenAI UNetModel | Diffusion.sample() | None | HIGH |
| `create_classifier` | `(...)` | guided_diffusion/script_util.py:189+ | Factory for classifier guidance model | Diffusion.sample() | None | MEDIUM |

### **Key Classes**

| Class | Base Class | File:Line | Key Methods | Attributes | Description |
|-------|------------|-----------|-------------|------------|-------------|
| `Diffusion` | object | runners/diffusion.py:55-340 | `__init__`, `sample`, `sample_sequence`, `sample_image` | `betas`, `alphas_cumprod`, `device`, `config` | Main runner for DDRM sampling |
| `Model` | nn.Module | models/diffusion.py:192+ | `__init__`, `forward` | `ch`, `temb_ch`, `down`, `up`, `mid` | Custom DDPM U-Net |
| `UNetModel` | nn.Module | guided_diffusion/unet.py:396+ | `__init__`, `forward`, `convert_to_fp16` | `image_size`, `model_channels`, `num_res_blocks` | OpenAI U-Net for diffusion |
| `H_functions` | ABC | functions/svd_replacement.py:3-66 | `V`, `Vt`, `U`, `Ut`, `singulars`, `H`, `H_pinv` | None | Abstract degradation operator |
| `Inpainting` | H_functions | functions/svd_replacement.py:108-145 | All from base | `missing_indices`, `kept_indices` | Inpainting degradation |
| `SuperResolution` | H_functions | functions/svd_replacement.py:169-220 | All from base | `ratio`, `y_dim`, `V_small`, `singulars_small` | Block-averaging SR |
| `Deblurring` | H_functions | functions/svd_replacement.py:383+ | All from base | `kernel`, `img_dim` | 1D blur degradation |
| `ImageDataset` | data.Dataset | datasets/imagenet_subset.py:51-100 | `__init__`, `__len__`, `__getitem__` | `metas`, `transform`, `root_dir` | Custom ImageNet loader |

---

## 5. Call Graph and Data Flow

### **Data Flow Diagram**

```
[Dataset Files on Disk]
         ↓
   get_dataset() → DataLoader
         ↓
   data_transform() [dequantization]
         ↓
   x_orig (clean images, [-1,1])
         ↓
   H_funcs.H(x_orig) → y_clean
         ↓
   y_0 = y_clean + noise
         ↓
   H_funcs.H_pinv(y_0) → pinv_y_0 (visualization)
         ↓
   efficient_generalized_steps(random_noise, y_0, model, H_funcs)
         ↓
   [Reverse Diffusion Loop]
   for t in [T, T-1, ..., 1]:
       xt → model(xt, t) → et (noise prediction)
       et → x0_t (predicted clean image)
       x0_t + degradation correction → xt_next
         ↓
   x_restored (final sample at t=0)
         ↓
   inverse_data_transform() [scale to [0,1]]
         ↓
   [Save PNG Files]
   - orig_{idx}.png
   - y0_{idx}.png
   - {idx}_-1.png (restored)
```

### **Key I/O Paths**

| Path | Type | Config Key | Purpose |
|------|------|------------|---------|
| `exp/logs/imagenet/*.pt` | Input | Hardcoded | Pre-trained diffusion model weights |
| `exp/logs/celeba/celeba_hq.ckpt` | Input | Hardcoded | CelebA-HQ model checkpoint |
| `exp/datasets/{dataset}/*` | Input | `args.exp` | Training/test images |
| `exp/imagenet_val_1k.txt` | Input | Hardcoded | ImageNet subset file list |
| `exp/image_samples/{args.image_folder}/*` | Output | `args.image_folder` | Restored images |
| `inp_masks/*.npy` | Input | Hardcoded | Inpainting masks |

**Configurable via CLI:**
- `--exp`: Base experiment directory (default: `exp`)
- `-i, --image_folder`: Output folder name (default: `images`)

**Hardcoded Paths:**
- Model checkpoints: `exp/logs/{imagenet|celeba}/...`
- Datasets: `exp/datasets/{imagenet|celeba|lsun|ood*}/...`
- Masks: `inp_masks/{lolcat_extra|lorem3}.npy`

---

## 6. Models Used

### **Model 1: OpenAI UNetModel (Guided Diffusion)**

- **Type:** Pre-trained library model (torch.nn.Module)
- **Defined In:** `guided_diffusion/unet.py:396-666`
- **Instantiated In:** `runners/diffusion.py:121-158` via `create_model()`
- **Architecture:**
  - U-Net with ResNet blocks and attention
  - Input: 3 channels (RGB), Output: 3 or 6 channels (with/without learned sigma)
  - Supports class-conditional generation (1000 ImageNet classes)
  - FP16 precision support
  - Configurable: num_channels=256, num_res_blocks=2, attention at [32,16,8] resolutions
- **Pretrained Weights:**
  - **256x256 unconditional:** `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt`
  - **256x256 conditional:** `.../256x256_diffusion.pt`
  - **512x512:** `.../512x512_diffusion.pt`
  - Loaded at `runners/diffusion.py:139`
- **Training:** No training in this codebase (inference only)
- **Usage:** Called in `efficient_generalized_steps()` as `model(xt, t)` → noise prediction

### **Model 2: Custom DDPM Model**

- **Type:** Custom torch.nn.Module
- **Defined In:** `models/diffusion.py:192-250+`
- **Instantiated In:** `runners/diffusion.py:98`
- **Architecture:**
  - Custom U-Net with time embedding
  - Encoder: [1,1,2,2,4,4] channel multipliers from base 128
  - Decoder: Mirror of encoder
  - Attention at resolution 16
  - ResNet blocks with GroupNorm and Swish
- **Pretrained Weights:**
  - **CelebA-HQ:** `https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt`
  - **LSUN (bedroom/cat/church):** Heidelberg University servers
  - Downloaded via `get_ckpt_path()` or direct `download()`
  - Loaded at `runners/diffusion.py:116`
- **Training:** Training configs present in bedroom.yml but not executed in notebook
- **Usage:** Called in `efficient_generalized_steps()` as `model(xt, t)` → noise prediction

### **Model 3: Classifier (Optional)**

- **Type:** EncoderUNetModel (from guided_diffusion)
- **Defined In:** `guided_diffusion/unet.py:684+`
- **Instantiated In:** `runners/diffusion.py:146-152` via `create_classifier()`
- **Architecture:**
  - Encoder-only U-Net for classification
  - Outputs class logits (1000 classes for ImageNet)
  - FP16 support
- **Pretrained Weights:**
  - **256x256:** `.../256x256_classifier.pt`
  - **512x512:** `.../512x512_classifier.pt`
- **Training:** No training (inference only)
- **Usage:** Classifier guidance via gradient of log p(y|x_t) → `cond_fn()` at `runners/diffusion.py:159-163`

### **Degradation Operators (Not Neural Networks)**

Nine analytical degradation operators (no learnable parameters):
1. **Inpainting** - Mask-based removal
2. **SuperResolution** - Block averaging (factor 2,4,8,16)
3. **SRConv** - Bicubic kernel SR
4. **Deblurring** - Uniform/Gaussian/Anisotropic blur
5. **Denoising** - Identity operator
6. **Colorization** - RGB to grayscale
7. **WalshHadamardCS** - Compressive sensing (factor 2,4)

---

## 7. Training & Evaluation

### **Training**

**Status:** No training is performed in this repository. It is **inference-only**.

**Evidence:**
- No `optimizer.step()`, `loss.backward()`, or training loops in main.py or notebook
- `Diffusion` class only has `sample()` method, no `train()` method
- Models are loaded from pre-trained checkpoints

**Training Configs Present (but unused):**
- `configs/bedroom.yml` includes training section:
  ```yaml
  training:
    batch_size: 64
    n_epochs: 10000
    n_iters: 5000000
    snapshot_freq: 5000
  
  optim:
    optimizer: "Adam"
    lr: 0.00002
    beta1: 0.9
  ```
- These are likely for reference or for training diffusion models from scratch (not covered in this repo)

### **Evaluation/Sampling**

**Hyperparameters (from CLI args and configs):**

| Parameter | Default | Config Source | Description |
|-----------|---------|---------------|-------------|
| `timesteps` | 1000 | CLI arg | Number of reverse diffusion steps |
| `eta` (ηA, ηC) | 0.85 | CLI arg | Sampling randomness parameter |
| `etaB` (ηB) | 1.0 | CLI arg | Noise parameter for noisier pixels |
| `sigma_0` | Required | CLI arg | Noise level in degraded observation |
| `batch_size` | 8 | Config YAML | Images processed per batch |
| `beta_start` | 0.0001 | Config YAML | Diffusion schedule start |
| `beta_end` | 0.02 | Config YAML | Diffusion schedule end |
| `num_diffusion_timesteps` | 1000 | Config YAML | Total timesteps in schedule |

**Metrics Computed:**

1. **PSNR (Peak Signal-to-Noise Ratio)**
   - Computed at `runners/diffusion.py:320-322`
   - Formula: `10 * log10(1 / MSE)`
   - Averaged over all samples
   - Printed to console and progress bar

**No other metrics** (no FID, SSIM, LPIPS) computed in the code.

**Logging:**
- Progress: `tqdm` progress bar with running PSNR
- Console: Final average PSNR and sample count
- No TensorBoard, Weights & Wandb, or CSV logging

**Saving/Loading:**

**Loading:**
- `torch.load(ckpt, map_location=device)` → `model.load_state_dict()`
- Checkpoints are `.pt` (OpenAI) or `.ckpt` (DDPM) format
- No optimizer states loaded (inference only)

**Saving:**
- Only output images saved: `tvu.save_image()` at `runners/diffusion.py:287-293, 315-318`
- Three files per sample:
  - `orig_{idx}.png`: Original clean image
  - `y0_{idx}.png`: Degraded observation (or pseudo-inverse)
  - `{idx}_-1.png`: Restored image
- No model checkpoints saved

---

## 8. Dependencies & Environment

### **Conda Environment Summary**

**Channels:**
- pytorch
- conda-forge  
- defaults

**Major Packages:**

| Package | Version | Purpose | Platform Issue |
|---------|---------|---------|----------------|
| python | 3.9.7 | Runtime | ✓ OK |
| pytorch | 1.10.1 | Deep learning framework | GPU: CUDA 11.3 build |
| torchvision | 0.11.2 | Image transforms/datasets | GPU: CUDA 11.3 build |
| torchaudio | 0.10.1 | (unused in code) | GPU: CUDA 11.3 build |
| cudatoolkit | 11.3.1 | CUDA runtime | ❌ **GPU-only, not on macOS** |
| numpy | 1.21.2 | Numerical computing | ✓ OK |
| scipy | 1.7.3 | Scientific computing | ✓ OK |
| pillow | 8.4.0 | Image I/O | ✓ OK |
| pyyaml | 6.0 | Config parsing | ✓ OK |
| tqdm | 4.62.3 | Progress bars | ✓ OK |
| tensorboard | 2.7.0 | (unused) | ✓ OK |
| requests | 2.26.0 | HTTP downloads | ✓ OK |
| lmdb | 0.9.29 | LSUN dataset format | ✓ OK |
| torch-fidelity | 0.3.0 | Evaluation metrics (unused) | ✓ OK (pip) |

**Linux-Specific Packages (❌ Not on macOS):**
- `_libgcc_mutex`, `_openmp_mutex`, `ld_impl_linux-64`, `libgcc-ng`, `libgfortran-ng`, `libgfortran5`, `libgomp`, `libstdcxx-ng`

### **macOS/CPU Adaptation Guide**

To run this project on **macOS with CPU-only**:

#### **Option 1: Modify environment.yml**

```yaml
# Remove or comment out these lines:
# - cudatoolkit=11.3.1=h2bc3f7f_2
# - pytorch-mutex=1.0=cuda

# Remove Linux-specific packages:
# - _libgcc_mutex=0.1=conda_forge
# - ld_impl_linux-64=2.36.1=hea4e1c9_2
# (... all libgcc, libgomp, etc.)

# Replace PyTorch with CPU build:
dependencies:
  # ... existing packages ...
  - pytorch=1.10.1=py3.9_0  # CPU-only build
  - torchvision=0.11.2=py39_cpu
  - torchaudio=0.10.1=py39  # or remove if unused
```

#### **Option 2: Create new environment from scratch**

```bash
# On macOS
conda create -n ddrm_cpu python=3.9
conda activate ddrm_cpu

# Install PyTorch CPU-only
conda install pytorch torchvision torchaudio -c pytorch

# Install other dependencies
pip install pyyaml tqdm pillow numpy scipy requests lmdb torch-fidelity
```

#### **Code Changes Required:**

**None required!** The code auto-detects device:

```python
# main.py:140
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
```

On macOS without CUDA, it will automatically use CPU.

**Performance Warning:** CPU inference will be **significantly slower** (10-100x) than GPU. For 40 timesteps on 256x256 images:
- **GPU (CUDA):** ~5-10 seconds per image
- **CPU:** ~5-10 minutes per image (estimated)

### **Missing Dependencies**

**Not in environment.yml but used in Untitled5.ipynb:**
- `einops` - Installed via pip in notebook cell 1
- `matplotlib` - Installed via pip in notebook cell 1

**Recommendation:** Add to environment.yml or requirements.txt

---

## 9. I/O & Data Sources

### **Input Data**

#### **Pre-trained Model Checkpoints**

1. **OpenAI Guided Diffusion Models**
   - **Source:** `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/`
   - **Files:**
     - `256x256_diffusion_uncond.pt` (~1.2 GB)
     - `256x256_diffusion.pt` (~1.2 GB)
     - `256x256_classifier.pt` (~500 MB)
     - `512x512_diffusion.pt` (~2.4 GB)
     - `512x512_classifier.pt` (~1 GB)
   - **Downloaded by:** Notebook cell 3, or `functions/ckpt_util.py:download()`
   - **Stored at:** `exp/logs/imagenet/`

2. **DDPM Models (Heidelberg)**
   - **Source:** `https://heibox.uni-heidelberg.de/f/...`
   - **Models:** CIFAR10, LSUN Bedroom/Cat/Church
   - **Downloaded by:** `functions/ckpt_util.py:get_ckpt_path()`
   - **Stored at:** `exp/logs/diffusion_models_converted/`

3. **CelebA-HQ Model**
   - **Source:** `https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt`
   - **Downloaded by:** `runners/diffusion.py:111-113`
   - **Stored at:** `exp/logs/celeba/celeba_hq.ckpt`

#### **Datasets**

1. **Demo Images (DDRM Exp Datasets)**
   - **Source:** `https://github.com/jiamings/ddrm-exp-datasets.git`
   - **Contents:** Sample images for OOD experiments
   - **Downloaded by:** Notebook cell 4
   - **Stored at:** `exp/datasets/ood*`, `exp/datasets/celeba_hq`

2. **ImageNet Validation Subset**
   - **Source:** `https://raw.githubusercontent.com/XingangPan/deep-generative-prior/master/scripts/imagenet_val_1k.txt`
   - **Contents:** List of 1000 image filenames
   - **Downloaded by:** Notebook cell 5
   - **Stored at:** `exp/imagenet_val_1k.txt`
   - **Format:** One stem per line (no extension), loader appends `.jpeg`

3. **Full Datasets (Not Downloaded by Notebook)**
   - **CelebA:** Downloaded by torchvision if `download=True`
   - **LSUN:** Must be manually downloaded and extracted
   - **ImageNet:** Must be manually downloaded (not provided by torchvision)

#### **Inpainting Masks**

- **Files:**
  - `inp_masks/lolcat_extra.npy`
  - `inp_masks/lorem3.npy`
- **Format:** NumPy binary (`.npy`)
- **Content:** Binary masks (0 = remove, 1 = keep)
- **Loaded at:** `runners/diffusion.py:205-211`

### **Output Data**

#### **Restored Images**

- **Location:** `exp/image_samples/{args.image_folder}/`
- **Format:** PNG (8-bit RGB)
- **Files per sample:**
  - `orig_{idx}.png` - Original clean image
  - `y0_{idx}.png` - Degraded observation (or pseudo-inverse)
  - `{idx}_-1.png` - Restored image
- **Naming:** `idx` ranges from `args.subset_start` to `args.subset_end`

### **Degradation Types**

Supported degradation operators (from `runners/diffusion.py:199-262`):

| Code | Type | Description | Parameters |
|------|------|-------------|------------|
| `cs2`, `cs4` | Compressive Sensing | Walsh-Hadamard, compression by 2 or 4 | Random permutation |
| `inp`, `inp_lolcat`, `inp_lorem` | Inpainting | Mask-based pixel removal | Mask file or random 50% |
| `deno` | Denoising | Identity (pure noise removal) | sigma_0 |
| `deblur_uni` | Deblurring | Uniform 9-tap blur | 1/9 kernel |
| `deblur_gauss` | Deblurring | Gaussian blur (σ=10) | 5-tap Gaussian |
| `deblur_aniso` | Deblurring | Anisotropic 2D blur | σ1=1, σ2=20 |
| `sr2`, `sr4`, `sr8`, `sr16` | Super-Resolution | Block averaging | Downscale factor |
| `sr_bicubic4`, `sr_bicubic8`, `sr_bicubic16` | Super-Resolution | Bicubic downsampling | Downscale factor |
| `color` | Colorization | RGB to grayscale | - |

---

## 10. Security & Privacy Notes

### **External Data Downloads**

The code downloads files from external URLs without verification (except MD5 for DDPM checkpoints):

1. **OpenAI Models (Azure Blob Storage)**
   - URL: `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/*`
   - Files: `256x256_diffusion*.pt`, `512x512_*.pt`, classifiers
   - **Risk:** No checksum verification (could be tampered if Azure compromised)
   - **Location:** `runners/diffusion.py:130-145`

2. **CelebA-HQ Model (AWS S3)**
   - URL: `https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt`
   - **Risk:** No checksum verification, S3 bucket name suggests test account
   - **Location:** `runners/diffusion.py:112`

3. **DDPM Models (Heidelberg University)**
   - URLs: `https://heibox.uni-heidelberg.de/f/...`
   - **Protection:** MD5 verification in `functions/ckpt_util.py:73`
   - **Risk:** Low (academic institution, verified)

4. **ImageNet File List (GitHub)**
   - URL: `https://raw.githubusercontent.com/XingangPan/deep-generative-prior/master/scripts/imagenet_val_1k.txt`
   - **Risk:** Low (plain text file)

5. **Demo Datasets (GitHub)**
   - URL: `https://github.com/jiamings/ddrm-exp-datasets.git`
   - **Risk:** Low (public repo, images only)

### **Hard-Coded Secrets**

**None found.** No API keys, tokens, or credentials in the code.

### **Privacy Concerns**

1. **Dataset Licensing:**
   - ImageNet: Requires agreement with organizers
   - CelebA: Academic use only
   - LSUN: Check license before commercial use

2. **Model Checkpoints:**
   - OpenAI models: MIT license (likely)
   - CelebA-HQ model: Unknown license (S3 test bucket)

### **Recommendations**

1. **Add Checksum Verification:**
   ```python
   # For OpenAI and CelebA-HQ downloads
   EXPECTED_SHA256 = {...}
   assert sha256(downloaded_file) == EXPECTED_SHA256[model_name]
   ```

2. **Use HTTPS Verification:**
   - `requests.get(..., verify=True)` (default, but make explicit)

3. **Document Data Provenance:**
   - Add LICENSE file noting model/data sources and licenses

4. **Pin Checkpoint URLs:**
   - Consider mirroring checkpoints to avoid external dependency failures

---

## 11. Unknowns (Cannot Be Determined Without Execution)

### **Runtime-Dependent Values**

1. **Actual Dataset Sizes**
   - Number of images in `exp/datasets/{ood*,celeba_hq,imagenet}`
   - **Why Unknown:** Depends on what user downloads/prepares
   - **Impact:** Determines total inference time and output count

2. **Model Output Shapes**
   - Whether models output 3 or 6 channels (learned sigma)
   - **Why Unknown:** Depends on checkpoint used (config says `learn_sigma: True` but implementation slices `[:, :3]`)
   - **Impact:** Memory usage, potential errors if mismatched

3. **Classifier Usage**
   - Whether classifier guidance is actually applied
   - **Why Unknown:** `class_cond: false` in imagenet_256.yml, so classifier code likely skipped
   - **Impact:** Sampling quality and speed

4. **GPU Memory Requirements**
   - Exact VRAM needed for different batch sizes
   - **Why Unknown:** Depends on image_size, timesteps, model size
   - **Estimate:** 256x256, batch_size=8, likely ~6-10 GB VRAM

5. **Numerical Stability**
   - Whether singular value thresholding (ZERO=3e-2) causes issues
   - **Why Unknown:** Depends on specific degradation matrices
   - **Impact:** Could cause NaNs or poor restoration

6. **Actual Execution Order in Notebook**
   - Cells 1-2 are identical (repeated setup)
   - **Why Unknown:** Execution order metadata not preserved in static read
   - **Impact:** May cause confusion, but both do same thing

7. **Random Seeds**
   - While `args.seed=1234` is set, actual reproducibility unknown
   - **Why Unknown:** PyTorch/CUDA non-determinism not disabled
   - **Impact:** Results may vary slightly between runs

8. **Performance Benchmarks**
   - Actual inference time per image
   - **Why Unknown:** Depends on hardware (GPU model, CPU, memory)
   - **Estimate:** ~5-10 seconds per image on V100 GPU

### **Configuration Ambiguities**

1. **`out_of_dist` Flag Semantics**
   - celeba_hq.yml sets `out_of_dist: True`
   - bedroom.yml also sets `out_of_dist: true`
   - **Why Ambiguous:** Seems to mean "use ood folder" but name suggests different distribution
   - **Impact:** Dataset loading path selection

2. **Training Config Usage**
   - bedroom.yml has complete training config (optimizer, lr, epochs)
   - **Why Ambiguous:** No training code present, unclear if for documentation or future use
   - **Impact:** None (not executed)

3. **Tensor Scaling**
   - Comment says "to account for scaling to [-1,1]" at `runners/diffusion.py:261`
   - `sigma_0 = 2 * args.sigma_0`
   - **Why Ambiguous:** Not clear if this applies to all configs or only some
   - **Impact:** Noise level tuning may be non-intuitive

### **Missing Information**

1. **Library Versions in Colab**
   - Notebook says "use Colab's default PyTorch"
   - **Why Unknown:** Colab PyTorch version changes over time
   - **Impact:** Could cause compatibility issues

2. **Intermediate Outputs**
   - `efficient_generalized_steps()` returns `xs` and `x0_preds` lists
   - Only final `xs[-1]` is saved
   - **Why Unknown:** Whether intermediate visualizations would be useful
   - **Impact:** Debugging difficulty

3. **Error Handling**
   - Very limited try/except blocks
   - **Why Unknown:** What happens on OOM, missing files, corrupt checkpoints
   - **Impact:** Unclear failure modes

---

## 12. Actionable Recommendations

### **Priority 1: Critical for Running**

1. **Create requirements.txt**
   ```txt
   torch>=1.10.0
   torchvision>=0.11.0
   numpy>=1.21.0
   scipy>=1.7.0
   Pillow>=8.4.0
   PyYAML>=6.0
   tqdm>=4.62.0
   requests>=2.26.0
   lmdb>=0.9.29
   torch-fidelity>=0.3.0
   einops>=0.4.0
   matplotlib>=3.4.0
   ```

2. **Add macOS/CPU Installation Guide to README**
   ```markdown
   ## Installation (macOS / CPU-only)
   
   ```bash
   conda create -n ddrm python=3.9
   conda activate ddrm
   conda install pytorch torchvision -c pytorch
   pip install -r requirements.txt
   ```
   ```

3. **Add Checksum Verification**
   - Extend `functions/ckpt_util.py` to verify SHA256 for OpenAI models
   - Add checksums to config or constants file

4. **Document Data Preparation**
   - Create `docs/DATA_SETUP.md` with step-by-step instructions
   - Include expected directory structure diagram
   - Add download links and mirror options

### **Priority 2: Improve Code Quality**

5. **Add Docstrings**
   - All public functions in `functions/`, `runners/`, `models/`
   - Example:
     ```python
     def efficient_generalized_steps(...):
         """
         DDRM reverse diffusion sampling with degradation correction.
         
         Args:
             x: Initial noise tensor [B, C, H, W]
             seq: Timestep sequence (reversed)
             model: Denoising network
             ...
         
         Returns:
             xs: List of xt states
             x0_preds: List of predicted clean images
         """
     ```

6. **Type Hints**
   - Add type annotations to function signatures
   - Use `from typing import Optional, Tuple, List, Union`

7. **Config Validation**
   - Add schema validation for YAML configs (e.g., using `pydantic`)
   - Check required keys, valid value ranges

8. **Error Handling**
   - Wrap downloads in try/except with retry logic
   - Check file existence before loading
   - Validate tensor shapes after model inference

### **Priority 3: User Experience**

9. **Progress Logging**
   - Add estimated time remaining to tqdm bars
   - Log intermediate PSNR values (not just final average)
   - Option to save restoration progress images

10. **Unified Entry Point**
    - Rename `Untitled5.ipynb` → `demo.ipynb` or `quickstart.ipynb`
    - Add markdown cells explaining each step
    - Include example outputs (images)

11. **Command-Line Help**
    - Improve `main.py` help text with examples
    - Add `--list-degradations` flag to show all options
    - Add `--dry-run` to validate config without running

12. **Visualization Tools**
    - Add `scripts/compare_results.py` to create side-by-side comparisons
    - Generate HTML report with all degradations

### **Priority 4: Testing & Validation**

13. **Unit Tests**
    - Test degradation operators (H_funcs) with known inputs
    - Test data transforms (forward + inverse should be identity)
    - Test config parsing

14. **Integration Test**
    - Small test dataset (10 images)
    - Quick test with `--timesteps 5` to verify pipeline

15. **Documentation Tests**
    - Verify all commands in README.md actually work
    - Test on fresh environment (Docker or Colab)

### **Priority 5: Repository Hygiene**

16. **Add .gitignore**
    ```
    exp/
    __pycache__/
    *.pyc
    .ipynb_checkpoints/
    *.pt
    *.ckpt
    *.pth
    .DS_Store
    ```

17. **License Clarity**
    - Add LICENSE file (appears to be missing or not in visible files)
    - Document licenses of used models and datasets

18. **Version Control**
    - Rename `Untitled5.ipynb` to meaningful name
    - Remove duplicate cells (1 and 2 are identical)
    - Add version tags for releases

### **Immediate Action Items**

**For a developer wanting to run this today:**

1. **Determine target platform:**
   - GPU (Linux): Use environment.yml as-is
   - GPU (Windows): May need adjustments
   - CPU (macOS/Linux): Follow Priority 1, item 2

2. **Minimal setup:**
   ```bash
   # GPU
   conda env create -f environment.yml
   conda activate ddrm
   
   # CPU (macOS)
   conda create -n ddrm python=3.9
   conda activate ddrm
   pip install torch torchvision pyyaml tqdm pillow numpy scipy requests lmdb
   ```

3. **Download one model:**
   ```bash
   mkdir -p exp/logs/imagenet
   wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt \
     -O exp/logs/imagenet/256x256_diffusion_uncond.pt
   ```

4. **Prepare test data:**
   ```bash
   mkdir -p exp/datasets/imagenet/imagenet
   # Copy 10 test images (JPEG) to this folder
   ls exp/datasets/imagenet/imagenet/*.jpeg | xargs -I {} basename {} .jpeg > exp/imagenet_val_1k.txt
   ```

5. **Run quick test:**
   ```bash
   python main.py --ni --config imagenet_256.yml --doc imagenet \
     --timesteps 10 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05 -i test_output
   ```

6. **Check outputs:**
   ```bash
   ls exp/image_samples/test_output/
   # Should see: orig_0.png, y0_0.png, 0_-1.png, etc.
   ```

---

## 13. Technical Notes

### **Algorithm Implementation**

The core DDRM algorithm (from paper Algorithm 1) is in `functions/denoising.py:11-107`:

1. **Initialization** (lines 14-32):
   - Compute SVD-related quantities via H_funcs
   - Initialize x_T ~ p(x_T | x_0, y) using singulars and noise

2. **Reverse Loop** (lines 41-100):
   - For t from T to 1:
     - Predict noise: e_t = model(x_t, t)
     - Estimate x_0: x_0,t = (x_t - √(1-ᾱ_t) e_t) / √ᾱ_t
     - **Degradation Correction** (lines 67-94):
       - Case 1: Missing/unobserved dimensions → standard DDIM step (etaC)
       - Case 2: Less noisy than y (σ_t < σ_0) → data consistency (etaA)
       - Case 3: Noisier than y (σ_t > σ_0) → blend observation and prediction (etaB)
     - Combine and project: x_{t-1} = V(Vt(combined)) * √ᾱ_{t-1}

**Key Insight:** Avoids explicit matrix inversion by working in V-space (right singular vectors).

### **Supported Configurations**

Based on configs/ analysis:

| Dataset | Model Type | Resolution | Class Cond | Notes |
|---------|------------|------------|------------|-------|
| ImageNet | openai | 256 | No | Fastest, best quality |
| ImageNet | openai | 256 | Yes | With classifier guidance |
| ImageNet | openai | 512 | Yes | High-res, slow |
| CelebA-HQ | simple | 256 | No | Faces only |
| LSUN Bedroom | simple | 256 | No | Indoor scenes |
| LSUN Cat | simple | 256 | No | Cat images |
| LSUN Church | simple | 256 | No | Architecture |

### **Performance Characteristics**

**Bottlenecks (static analysis):**
1. Model inference: Dominant cost (40 forward passes for --timesteps 40)
2. SVD operations: Minimal (avoided via efficient implementations)
3. I/O: Negligible (batch processing)

**Memory (estimated):**
- Model weights: ~1.2 GB (256x256) to ~2.4 GB (512x512)
- Activation memory: ~500 MB per image @ 256x256 (depends on batch size)
- **Total:** 6-10 GB VRAM recommended for batch_size=8

### **Extension Points**

1. **New Degradations:**
   - Subclass `H_functions` in `functions/svd_replacement.py`
   - Implement V, Vt, U, Ut, singulars, add_zeros
   - Register in `runners/diffusion.py:199-262`

2. **New Models:**
   - Add model type to configs (e.g., `type: "custom"`)
   - Load in `runners/diffusion.py:sample()` before line 165

3. **New Metrics:**
   - Add computation in `runners/diffusion.py:320-323`
   - Consider torch-fidelity for FID, IS

---

## Conclusion

This DDRM implementation is a **research-grade inference codebase** for image restoration using diffusion models. It is:

✅ **Well-structured** with modular degradation operators  
✅ **Flexible** supporting 15+ degradation types and multiple datasets  
✅ **Efficient** avoiding explicit SVD via analytical operators  

❌ **Lacks documentation** (minimal docstrings, unclear setup)  
❌ **Platform-specific** (Linux/GPU focused, macOS requires manual adaptation)  
❌ **Limited evaluation** (only PSNR, no perceptual metrics)  

**Best for:** Researchers familiar with diffusion models and PyTorch, running on Linux with NVIDIA GPUs.

**Not ideal for:** Production deployment, macOS users, beginners needing hand-holding.

**To make production-ready:** Follow Priority 1-3 recommendations above, especially adding requirements.txt, error handling, and comprehensive documentation.

---

**End of PROJECT_OVERVIEW.md**

