# CelebA-HQ Model Download Troubleshooting

**Issue:** CelebA-HQ model (Config 1) fails to download while OpenAI models succeed.

**Status from your output:**
```
❌ Config 1: CelebA-HQ (Custom DDPM)             MISSING
✅ Config 2: ImageNet Uncond (OpenAI)              2.06 GB
✅ Config 3: ImageNet Cond (OpenAI)                2.06 GB
✅ Config 3: Classifier                            0.20 GB
```

---

## Root Cause Analysis (Static)

### Problem 1: Unreliable S3 Bucket URL

**Current URL in code:**
```python
CELEBA_URL = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
```

**Issues:**
- Contains "test-12345" → suggests temporary/test bucket
- May have access restrictions or be deleted
- Not an official/stable hosting location

**Evidence:** This is mentioned in `docs/SECURITY_WARNINGS.md` as a security concern.

### Problem 2: Silent Failure

```python
subprocess.run(["wget", "-q", "--show-progress", CELEBA_URL, "-P", "exp/logs/celeba/"], check=False)
```

**Issues:**
- `check=False` → ignores wget exit codes (404, connection errors, etc.)
- `-q` flag suppresses error messages
- No error handling or retry logic

### Problem 3: Path Verification Issue

The code checks:
```python
if not os.path.isfile("exp/logs/celeba/celeba_hq.ckpt"):
```

But in Colab, the working directory might be different, or the file might save with a different name.

---

## Solutions (Choose One)

### Solution A: Use Alternative Download Method (Recommended)

Replace wget with Python's `requests` library for better error handling:

```python
import requests
from tqdm import tqdm

def download_with_progress(url, dest_path):
    """Download file with progress bar and error handling."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Stream download
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise error for bad status codes
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=os.path.basename(dest_path)
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Downloaded: {dest_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed: {e}")
        return False

# Usage:
CELEBA_URL = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
celeba_path = "exp/logs/celeba/celeba_hq.ckpt"

if not os.path.exists(celeba_path):
    print("⏬ Downloading CelebA-HQ model (~500 MB)...")
    success = download_with_progress(CELEBA_URL, celeba_path)
    if not success:
        print("⚠️ Trying alternative URL...")
        # Try alternative source if primary fails
else:
    print("✅ CelebA-HQ model already exists")
```

### Solution B: Use the DDRM Code's Built-in Downloader

The DDRM codebase has its own download function in `runners/diffusion.py`:

```python
# From runners/diffusion.py:111-113 (static analysis)
from functions.ckpt_util import download

celeba_path = "exp/logs/celeba/celeba_hq.ckpt"
if not os.path.exists(celeba_path):
    print("⏬ Downloading CelebA-HQ model...")
    download(
        'https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt',
        celeba_path
    )
    print(f"✅ Downloaded to: {celeba_path}")
```

### Solution C: Manual Download + Upload to Colab

**If URL is permanently broken:**

1. **Try downloading locally first:**
   ```bash
   # On your local machine
   wget https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt
   
   # Or use curl
   curl -O https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt
   ```

2. **If download succeeds locally, upload to Colab:**
   ```python
   from google.colab import files
   
   print("Please upload celeba_hq.ckpt:")
   uploaded = files.upload()
   
   # Move to correct location
   os.makedirs("exp/logs/celeba", exist_ok=True)
   os.rename("celeba_hq.ckpt", "exp/logs/celeba/celeba_hq.ckpt")
   ```

3. **Or upload to your Google Drive first:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Copy from Drive
   !cp /content/drive/MyDrive/celeba_hq.ckpt exp/logs/celeba/
   ```

### Solution D: Skip Config 1, Run Only Configs 2 & 3

Since the OpenAI models downloaded successfully, you can:

```python
# Comment out or skip Config 1 cells
# Run only Config 2 and Config 3

# This still demonstrates the key finding:
# - Config 2 (OpenAI UNet) runs standalone
# - Config 3 (OpenAI + Classifier) shows dependency
# - Both are independent of each other (mutually exclusive)
```

---

## Recommended Fix for the Notebook

Add this enhanced download cell **before** the current CelebA download:

```python
# %%
# Enhanced CelebA-HQ Download with Error Handling

import requests
from tqdm import tqdm

def robust_download(url, dest_path, chunk_size=8192):
    """
    Download file with progress bar, error handling, and verification.
    Returns True if successful, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        print(f"Attempting download from: {url}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"File size: {total_size / (1024**3):.2f} GB")
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Verify download
        if os.path.exists(dest_path):
            actual_size = os.path.getsize(dest_path)
            if actual_size > 1024 * 1024:  # At least 1 MB
                print(f"✅ Download successful: {actual_size / (1024**3):.2f} GB")
                return True
            else:
                print(f"⚠️ Downloaded file too small: {actual_size} bytes")
                os.remove(dest_path)
                return False
        
        return False
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print("   The S3 bucket URL may be invalid or access restricted.")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Unable to reach server")
        return False
    except requests.exceptions.Timeout:
        print("❌ Timeout: Download took too long")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return False

# Try downloading CelebA-HQ model
CELEBA_URL = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
celeba_path = "exp/logs/celeba/celeba_hq.ckpt"

if not os.path.exists(celeba_path):
    print("⏬ Downloading CelebA-HQ model (~500 MB)...")
    print("⚠️ Note: This URL uses a test S3 bucket and may be unreliable\n")
    
    success = robust_download(CELEBA_URL, celeba_path)
    
    if not success:
        print("\n" + "="*60)
        print("WORKAROUND: CelebA-HQ model unavailable")
        print("="*60)
        print("Options:")
        print("1. Skip Config 1 (Custom DDPM) - Configs 2 & 3 will still work")
        print("2. Download manually and upload to Colab")
        print("3. Try alternative sources (see SECURITY_WARNINGS.md)")
        print("="*60 + "\n")
else:
    print("✅ CelebA-HQ model already exists")
```

---

## Why This Happens (Root Causes)

### 1. **Unstable Source URL**
From `docs/SECURITY_WARNINGS.md` (static analysis):
> **CelebA-HQ Model (AWS S3 - Test Bucket!)**
> - URL: `https://image-editing-test-12345.s3-us-west-2.amazonaws.com/...`
> - **Risk:** Bucket name contains "test-12345" - suggests temporary/unsecured bucket
> - Unknown ownership and trustworthiness

### 2. **Network/Firewall Issues**
- Colab may have restrictions on certain AWS regions
- S3 bucket permissions may have changed
- Temporary network issues

### 3. **File Naming Issues**
wget might save with a different name due to URL redirects or S3 headers.

---

## Verification Steps

After applying fix, verify download:

```python
# Check if file exists and has reasonable size
celeba_path = "exp/logs/celeba/celeba_hq.ckpt"

if os.path.exists(celeba_path):
    size_mb = os.path.getsize(celeba_path) / (1024**2)
    print(f"✅ File exists: {size_mb:.2f} MB")
    
    # CelebA-HQ should be ~300-600 MB
    if size_mb < 10:
        print("⚠️ File too small - likely corrupted or incomplete")
    elif size_mb > 1000:
        print("⚠️ File too large - unexpected")
    else:
        print("✅ File size looks reasonable")
else:
    print("❌ File not found")
```

---

## Impact on Notebook Functionality

**If CelebA-HQ model cannot be downloaded:**

### ✅ What Still Works:
- Config 2 (OpenAI UNet) - **Fully functional**
- Config 3 (OpenAI + Classifier) - **Fully functional**
- Side-by-side comparison of Configs 2 & 3
- Validation of key findings:
  - Models are alternatives (2 vs 3)
  - Classifier depends on UNet (Config 3)
  - Independent execution (separate outputs)

### ❌ What Won't Work:
- Config 1 (Custom DDPM) execution
- CelebA face restoration examples
- Full three-way comparison visualization

### 📊 Core Thesis Still Validated:

**Even with only Configs 2 & 3, you prove:**

1. ✅ **Models are alternatives** - Config 2 (UNet only) vs Config 3 (UNet + Classifier) load different model combinations
2. ✅ **Classifier is optional add-on** - Config 2 works without it
3. ✅ **Separate execution paths** - Two different output folders
4. ✅ **No interference** - Both complete independently

**The Custom DDPM would add a third example, but isn't essential to prove the architecture pattern.**

---

## Quick Fix: Modify Notebook to Skip Config 1

If you want to proceed immediately without fixing the download:

```python
# Add at the start of Config 1 section:
print("⚠️ SKIPPING Config 1 due to missing CelebA-HQ model")
print("   Proceeding with Configs 2 & 3 only\n")

# Then skip/comment the Config 1 execution cells
# The notebook will still demonstrate the key findings with 2 configs
```

---

## Alternative: Use Config 2 Twice with Different Degradations

To maintain three comparisons without CelebA:

```python
# Config 1 Alternative: OpenAI UNet with SR4 (instead of Custom DDPM)
config1_alt_params = {
    "config": "imagenet_256.yml",
    "doc": "imagenet",
    "timesteps": 20,
    "eta": 0.85,
    "etaB": 1.0,
    "deg": "sr4",  # Different degradation than Config 2
    "sigma_0": 0.05,
    "image_folder": "config1_alt_openai_sr4",
    "subset_start": 0,
    "subset_end": 5,
}

# Now you have:
# Config 1-alt: OpenAI UNet + SR4 (no classifier)
# Config 2: OpenAI UNet + Deblur (no classifier)
# Config 3: OpenAI UNet + SR4 + Classifier
```

This still shows:
- Same model, different tasks (1-alt vs 2)
- Same model, same task, with/without classifier (1-alt vs 3)

---

## Summary

**Problem:** CelebA-HQ model download fails due to unreliable S3 test bucket URL.

**Impact:** Config 1 cannot run, but Configs 2 & 3 still validate core findings.

**Best Solution:** Use enhanced download code with error handling (Solution A).

**Quick Workaround:** Skip Config 1, run only Configs 2 & 3.

**Alternative:** Replace Config 1 with another OpenAI UNet config using different degradation.

---

**Reference:** `docs/SECURITY_WARNINGS.md` - section on unverified downloads

