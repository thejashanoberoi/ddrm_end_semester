# Model Usage and Dependencies Analysis

**Analysis Date:** December 4, 2025  
**Analysis Type:** Static code analysis  
**Question:** How are the three models (Guided Diffusion, Custom DDPM, Classifier) used? Are they interdependent or can run individually?

---

## TL;DR - Quick Answer

**The three models are MUTUALLY EXCLUSIVE, not interdependent:**

1. **Custom DDPM Model** (`type: "simple"`) - Used for CelebA, LSUN datasets
2. **OpenAI Guided Diffusion** (`type: "openai"`) - Used for ImageNet
3. **Classifier** - OPTIONAL add-on, only with Guided Diffusion + `class_cond: true`

**Key Finding:** You choose ONE diffusion model based on config, and optionally add the classifier for class-conditional generation.

---

## Detailed Analysis

### 1. Model Selection Logic (Mutually Exclusive)

**Location:** `runners/diffusion.py:93-165`

The code uses an **if-elif** structure, meaning only ONE diffusion model is loaded:

```python
cls_fn = None
if self.config.model.type == 'simple':
    # Load Custom DDPM Model
    model = Model(self.config)
    # ... load checkpoint ...
    
elif self.config.model.type == 'openai':
    # Load OpenAI Guided Diffusion Model
    model = create_model(**config_dict)
    # ... load checkpoint ...
    
    # OPTIONALLY load classifier if class_cond is True
    if self.config.model.class_cond:
        classifier = create_classifier(...)
        # ... define cls_fn ...

self.sample_sequence(model, cls_fn)
```

**Implication:** The models are **alternatives**, not used together.

---

### 2. Configuration-Based Selection

#### Option A: Custom DDPM Model (type: "simple")

**Used by:**
- `configs/celeba_hq.yml` - CelebA-HQ faces
- `configs/bedroom.yml` - LSUN bedroom scenes
- `configs/cat.yml` - LSUN cat images
- `configs/church.yml` - LSUN church images

**Example:** `configs/celeba_hq.yml:14-15`
```yaml
model:
    type: "simple"
```

**Model class:** `models/diffusion.py::Model`  
**Architecture:** Custom U-Net with:
- Base channels: 128
- Channel multipliers: [1,1,2,2,4,4]
- Attention at resolution 16
- Time embedding via dense layers

**Checkpoint sources:**
- CelebA-HQ: AWS S3 bucket
- LSUN: Heidelberg University servers

**No classifier used** - `cls_fn` remains `None`

---

#### Option B: OpenAI Guided Diffusion (type: "openai")

**Used by:**
- `configs/imagenet_256.yml` - ImageNet 256x256 (unconditional)
- `configs/imagenet_256_cc.yml` - ImageNet 256x256 (conditional)
- `configs/imagenet_512_cc.yml` - ImageNet 512x512 (conditional)

**Example:** `configs/imagenet_256.yml:14-15`
```yaml
model:
    type: "openai"
    ...
    class_cond: false  # No classifier
```

**Model class:** `guided_diffusion/unet.py::UNetModel`  
**Architecture:** OpenAI's U-Net with:
- Model channels: 256
- Attention at resolutions [32,16,8]
- FP16 precision support
- ResNet blocks with scale-shift normalization

**Checkpoint source:** OpenAI Azure Blob Storage

**Classifier optional** - depends on `class_cond` flag

---

### 3. Classifier Usage (Optional Add-On)

**Conditions for loading classifier:**
1. Model type must be `"openai"` (line 120)
2. Config flag `class_cond: true` (line 142)

**Code:** `runners/diffusion.py:142-163`
```python
if self.config.model.class_cond:
    # Load classifier checkpoint
    classifier = create_classifier(...)
    classifier.load_state_dict(...)
    classifier.eval()
    
    # Define classifier guidance function
    def cond_fn(x, t, y):
        # Compute gradient of log p(y|x_t) w.r.t. x_t
        logits = classifier(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale
    
    cls_fn = cond_fn  # Pass to sampling
```

**Classifier model:** `guided_diffusion/unet.py::EncoderUNetModel`  
**Purpose:** Provides class gradients for guided generation  
**Checkpoint source:** OpenAI Azure Blob Storage

**When classifier is NOT used:**
- `type: "simple"` (Custom DDPM) - `cls_fn` stays `None`
- `type: "openai"` with `class_cond: false` - `cls_fn` stays `None`

---

### 4. Model Usage in Sampling

**Location:** `functions/denoising.py:51-56`

During reverse diffusion, the selected model is called at each timestep:

```python
if cls_fn == None:
    # Standard unconditional denoising
    et = model(xt, t)
else:
    # Class-conditional denoising with classifier guidance
    et = model(xt, t, classes)
    et = et[:, :3]  # Take first 3 channels (RGB)
    et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x, t, classes)
```

**Key observations:**

1. **Only ONE diffusion model** (`model`) is called
2. **Classifier (`cls_fn`)** modifies the noise prediction if present
3. **Classes argument** only used when classifier is active

---

## Dependency Matrix

| Configuration | Diffusion Model | Classifier | Interdependent? |
|---------------|-----------------|------------|-----------------|
| celeba_hq.yml | Custom DDPM | ❌ None | No - standalone |
| bedroom.yml | Custom DDPM | ❌ None | No - standalone |
| imagenet_256.yml | OpenAI Guided | ❌ None | No - standalone |
| imagenet_256_cc.yml | OpenAI Guided | ✅ Used | **Dependent** - classifier needs diffusion model |
| imagenet_512_cc.yml | OpenAI Guided | ✅ Used | **Dependent** - classifier needs diffusion model |

---

## Can They Run Individually?

### ✅ YES - Diffusion Models Run Independently

**Custom DDPM:**
```bash
python main.py --config celeba_hq.yml --doc celeba \
  --timesteps 20 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05
```
- Loads **only** Custom DDPM model
- No OpenAI model loaded
- No classifier loaded
- Fully independent

**OpenAI Guided Diffusion (unconditional):**
```bash
python main.py --config imagenet_256.yml --doc imagenet \
  --timesteps 20 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05
```
- Loads **only** OpenAI UNet model
- No Custom DDPM loaded
- No classifier loaded
- Fully independent

---

### ⚠️ PARTIAL - Classifier Depends on Diffusion Model

**OpenAI Guided Diffusion + Classifier:**
```bash
python main.py --config imagenet_256_cc.yml --doc imagenet \
  --timesteps 20 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05
```
- Loads OpenAI UNet model **FIRST**
- Then loads Classifier **SECOND**
- Classifier is **dependent** on diffusion model being present
- Cannot use classifier without a diffusion model

**Why dependent?**
- Classifier provides gradient guidance to the diffusion process
- Requires the diffusion model's intermediate states (x_t)
- Acts as a modifier, not a standalone generator

---

### ❌ NO - Classifier Cannot Run Alone

**Impossible configuration:**
```python
# This cannot happen in the code
if classifier_only:  # No such branch exists
    classifier = create_classifier(...)
    # ... what would you sample from?
```

The classifier **requires**:
1. A diffusion model to generate intermediate states
2. Timestep information from the diffusion process
3. The DDRM algorithm to coordinate the sampling

---

## Architecture Decision Tree

```
START: main.py --config {CONFIG}.yml
    |
    v
Read config.model.type
    |
    +-- type == "simple" --> Load Custom DDPM Model
    |                        - Models: CelebA, LSUN
    |                        - Checkpoint: AWS S3 / Heidelberg
    |                        - cls_fn = None
    |                        - DONE (standalone)
    |
    +-- type == "openai" --> Load OpenAI UNet Model
                             - Models: ImageNet
                             - Checkpoint: OpenAI Azure
                             |
                             v
                        Check config.model.class_cond
                             |
                             +-- class_cond == false
                             |       - cls_fn = None
                             |       - DONE (standalone)
                             |
                             +-- class_cond == true
                                     - Load Classifier
                                     - Define cls_fn (gradient function)
                                     - DONE (diffusion + classifier)
```

---

## Memory and Performance Implications

### Scenario 1: Custom DDPM (standalone)
```
GPU Memory Usage:
- Custom DDPM weights: ~500 MB
- Activations: ~500 MB per image
Total: ~1-2 GB VRAM
```

### Scenario 2: OpenAI Guided (standalone)
```
GPU Memory Usage:
- OpenAI UNet weights: ~1.2 GB (256x256) or ~2.4 GB (512x512)
- Activations: ~500-1000 MB per image
Total: ~2-4 GB VRAM
```

### Scenario 3: OpenAI Guided + Classifier
```
GPU Memory Usage:
- OpenAI UNet weights: ~1.2 GB
- Classifier weights: ~500 MB
- Activations (both models): ~1000 MB
- Gradient computation: +200 MB
Total: ~3-5 GB VRAM
```

**Note:** These are estimates based on model sizes. Actual usage depends on batch size and image resolution.

---

## Code Evidence Summary

### Evidence 1: Mutually Exclusive Loading
**File:** `runners/diffusion.py:93-120`
```python
if self.config.model.type == 'simple':
    model = Model(self.config)
    # ... load Custom DDPM ...
elif self.config.model.type == 'openai':
    model = create_model(**config_dict)
    # ... load OpenAI UNet ...
```
**Conclusion:** Only ONE `model` variable exists. Cannot load both simultaneously.

---

### Evidence 2: Classifier is Conditional Add-On
**File:** `runners/diffusion.py:142-163`
```python
if self.config.model.class_cond:
    classifier = create_classifier(...)
    def cond_fn(x, t, y):
        # ... use classifier ...
    cls_fn = cond_fn
```
**Conclusion:** Classifier is only loaded when explicitly enabled AND using OpenAI model.

---

### Evidence 3: Single Model Inference
**File:** `functions/denoising.py:51-56`
```python
if cls_fn == None:
    et = model(xt, t)  # Just the diffusion model
else:
    et = model(xt, t, classes)  # Diffusion model + classifier correction
    et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x, t, classes)
```
**Conclusion:** Only ONE diffusion model is ever called. Classifier optionally modifies its output.

---

### Evidence 4: Config Files Confirm Separation
**Configs using Custom DDPM:**
- `celeba_hq.yml`: `type: "simple"`, no `class_cond` field
- `bedroom.yml`: `type: "simple"`, no `class_cond` field

**Configs using OpenAI (no classifier):**
- `imagenet_256.yml`: `type: "openai"`, `class_cond: false`

**Configs using OpenAI + Classifier:**
- `imagenet_256_cc.yml`: `type: "openai"`, `class_cond: true`
- `imagenet_512_cc.yml`: `type: "openai"`, `class_cond: true`

**Conclusion:** Configurations explicitly choose one path, no overlap.

---

## Practical Usage Examples

### Use Case 1: Face Restoration (Custom DDPM)
```bash
python main.py --config celeba_hq.yml --doc celeba \
  --timesteps 40 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05
```
**Models loaded:** Custom DDPM only  
**Purpose:** Restore degraded face images  
**Independent:** Yes, runs standalone

---

### Use Case 2: General Image Restoration (OpenAI)
```bash
python main.py --config imagenet_256.yml --doc imagenet \
  --timesteps 40 --eta 0.85 --etaB 1 --deg deblur_uni --sigma_0 0.0
```
**Models loaded:** OpenAI UNet only  
**Purpose:** Restore general images (unconditional)  
**Independent:** Yes, runs standalone

---

### Use Case 3: Class-Specific Restoration (OpenAI + Classifier)
```bash
python main.py --config imagenet_256_cc.yml --doc imagenet \
  --timesteps 40 --eta 0.85 --etaB 1 --deg sr4 --sigma_0 0.05
```
**Models loaded:** OpenAI UNet + Classifier  
**Purpose:** Restore images with class guidance (e.g., "make it look more like a dog")  
**Independent:** No, classifier depends on UNet  
**Note:** Requires class labels in dataset

---

## Conclusion

### Summary Table

| Model | Can Run Alone? | Needs Other Models? | Use Cases |
|-------|----------------|---------------------|-----------|
| **Custom DDPM** | ✅ Yes | ❌ No | CelebA faces, LSUN scenes |
| **OpenAI Guided Diffusion** | ✅ Yes | ❌ No | ImageNet, general images |
| **Classifier** | ❌ No | ✅ Yes (needs OpenAI) | Class-conditional generation |

### Final Answer

**The three models are NOT interdependent in the sense that you load all three together.**

Instead:
1. **Choose ONE diffusion model** (Custom DDPM OR OpenAI Guided Diffusion) based on dataset
2. **Optionally add Classifier** if using OpenAI + want class conditioning
3. **Never load** Custom DDPM + OpenAI together
4. **Never use** Classifier with Custom DDPM
5. **Classifier always depends** on a diffusion model (OpenAI), but not vice versa

**Architectural pattern:** Plugin architecture where:
- Base = ONE diffusion model (required)
- Plugin = Classifier (optional, only with OpenAI base)

---

**Generated via static code analysis - no execution required.**

