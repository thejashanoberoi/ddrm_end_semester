# Command Analysis: Model Configuration Identification

**Command:**
```bash
python main.py --ni --config imagenet_256.yml --doc imagenet \
  --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_uni --sigma_0 0.0 -i demo_deblur
```

---

## Answer: **Configuration 2 - OpenAI Guided Diffusion (Unconditional)**

---

## Analysis Breakdown

### 1. Config File Analysis

**Config file:** `imagenet_256.yml`

**Key settings from static analysis:**

```yaml
model:
    type: "openai"              # ← OpenAI UNet model
    class_cond: false           # ← No classifier (unconditional)
    num_channels: 256
    num_heads: 4
    attention_resolutions: "32,16,8"
    use_fp16: true
    use_scale_shift_norm: true
```

### 2. Model Loading Logic

From `runners/diffusion.py:93-165` (static analysis):

```python
if self.config.model.type == 'simple':
    # Config 1: Custom DDPM
    model = Model(self.config)
    
elif self.config.model.type == 'openai':    # ← This branch executes
    # Config 2 or 3: OpenAI UNet
    model = create_model(**config_dict)
    
    if self.config.model.class_cond:        # ← False, so skipped
        # Config 3: Load classifier
        classifier = create_classifier(...)
        cls_fn = cond_fn
    # Config 2: cls_fn remains None
```

**Result:** Only the OpenAI UNet model loads. No classifier.

---

## Configuration Identity

### ✅ **This is Configuration 2**

| Aspect | Value |
|--------|-------|
| **Configuration** | Config 2: OpenAI Guided Diffusion (Unconditional) |
| **Model Type** | `type: "openai"` |
| **Architecture** | OpenAI U-Net |
| **Source File** | `guided_diffusion/unet.py:396+` |
| **Classifier** | ❌ None (`class_cond: false`) |
| **Class Conditioning** | ❌ No |
| **Parameters** | ~280M |
| **Base Channels** | 256 |
| **Attention** | 4-heads @ 3 levels [32,16,8] |
| **FP16** | ✅ Enabled |
| **Independence** | ✅ Fully standalone |

---

## Why It's Config 2 (Not Config 1 or 3)

### ❌ Not Config 1 (Custom DDPM)
- Config 1 uses `type: "simple"`
- This uses `type: "openai"` ✓
- Config 1 loads `models/diffusion.py::Model`
- This loads `guided_diffusion/unet.py::UNetModel` ✓

### ❌ Not Config 3 (OpenAI + Classifier)
- Config 3 has `class_cond: true`
- This has `class_cond: false` ✓
- Config 3 loads classifier in addition to UNet
- This only loads UNet ✓

### ✅ It's Config 2 (OpenAI UNet, Unconditional)
- `type: "openai"` ✓
- `class_cond: false` ✓
- Only loads one model (UNet) ✓
- No classifier guidance ✓

---

## Task Being Performed

**Degradation:** `--deg deblur_uni`
- **Type:** Uniform deblurring
- **Operator:** `Deblurring` class from `functions/svd_replacement.py:383+`
- **Kernel:** Uniform 9-tap blur (all coefficients = 1/9)
- **Noise:** `sigma_0 = 0.0` (no added noise, clean blur)

**Output:** `demo_deblur/`
- Original images: `orig_*.png`
- Degraded (blurred): `y0_*.png`
- Restored (deblurred): `*_-1.png`

---

## Model Loaded (Static Code Trace)

### Step 1: Parse Config
```python
# main.py:17-152
config = parse_args_and_config()
# Loads: configs/imagenet_256.yml
```

### Step 2: Initialize Runner
```python
# main.py:170
runner = Diffusion(args, config, device)
# Creates diffusion schedule (betas, alphas)
```

### Step 3: Load Model
```python
# runners/diffusion.py:120-141
elif self.config.model.type == 'openai':
    from guided_diffusion.script_util import create_model
    
    # Create OpenAI UNet with config params
    model = create_model(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        attention_resolutions=[32,16,8],
        num_heads=4,
        use_fp16=True,
        use_scale_shift_norm=True,
        # ... other params
    )
    
    # Convert to FP16 for speed
    model.convert_to_fp16()
    
    # Load pretrained weights
    ckpt = "exp/logs/imagenet/256x256_diffusion_uncond.pt"
    model.load_state_dict(torch.load(ckpt))
    
    model.eval()
    model = torch.nn.DataParallel(model)
```

### Step 4: Check for Classifier
```python
# runners/diffusion.py:142
if self.config.model.class_cond:  # False, so this block is skipped
    # Would load classifier here (Config 3)
    pass

# cls_fn remains None (Config 2)
```

### Step 5: Run Sampling
```python
# runners/diffusion.py:165
self.sample_sequence(model, cls_fn=None)
# Uses model without classifier guidance
```

---

## Sampling Behavior

In `functions/denoising.py:51-56`:

```python
for timestep in reversed(seq):
    if cls_fn == None:               # ← True for Config 2
        et = model(xt, t)            # Just UNet prediction
    else:                            # Config 3 would go here
        et = model(xt, t, classes)
        et = et - gradient_correction  # Add classifier guidance
```

**For this command:**
- Only UNet predicts noise at each timestep
- No classifier gradient correction
- Unconditional denoising

---

## Visual Representation

```
Command Parameters
       │
       ├─ config: imagenet_256.yml
       │      │
       │      ├─ model.type: "openai"        → Use OpenAI UNet
       │      └─ model.class_cond: false     → No classifier
       │
       ├─ deg: deblur_uni                    → Task: deblurring
       ├─ timesteps: 20                      → Faster (reduced from 1000)
       └─ sigma_0: 0.0                       → No noise in degradation
              │
              ▼
    ┌─────────────────────┐
    │   Model Loading:    │
    │                     │
    │  ✅ OpenAI UNet     │  (~280M params)
    │  ❌ Classifier      │  (not loaded)
    └─────────────────────┘
              │
              ▼
        Configuration 2
     (OpenAI Unconditional)
              │
              ▼
         Deblur images
      Save to demo_deblur/
```

---

## Comparison with Other Configs

### If you wanted Config 1 (Custom DDPM):
```bash
python main.py --ni --config celeba_hq.yml --doc celeba \
  --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_uni --sigma_0 0.0 -i demo_deblur
```
Changes: `celeba_hq.yml` (has `type: "simple"`)

### If you wanted Config 3 (OpenAI + Classifier):
```bash
python main.py --ni --config imagenet_256_cc.yml --doc imagenet \
  --timesteps 20 --eta 0.85 --etaB 1 --deg deblur_uni --sigma_0 0.0 -i demo_deblur
```
Changes: `imagenet_256_cc.yml` (has `class_cond: true`)

---

## Summary

**Your command uses:**
- ✅ **Configuration 2: OpenAI Guided Diffusion (Unconditional)**
- ✅ Model: OpenAI U-Net (280M parameters)
- ✅ Classifier: None (unconditional generation)
- ✅ Task: Uniform deblurring
- ✅ Speed: Fast (20 timesteps instead of 1000)
- ✅ Independence: Fully standalone (no dependencies)

**Key identifiers:**
1. `--config imagenet_256.yml` → contains `type: "openai"`
2. Config has `class_cond: false` → no classifier
3. Loads: OpenAI UNet only
4. Sampling: Unconditional (no guidance)

---

**Analysis method:** Static code analysis  
**Files examined:** 
- `configs/imagenet_256.yml`
- `runners/diffusion.py:93-165`
- `functions/denoising.py:51-56`

**Related documentation:**
- `docs/MODEL_USAGE_ANALYSIS.md` - Full configuration comparison
- `docs/ARCHITECTURAL_DIFFERENCES.md` - Model architecture details
- `docs/ARCHITECTURE_VISUAL_SUMMARY.txt` - Visual diagrams

