# Security and Privacy Warnings

**Project:** DDRM (Denoising Diffusion Restoration Models)  
**Analysis Date:** December 4, 2025  
**Analysis Type:** Static code review (no execution)

---

## ⚠️ Critical Warnings

### 1. Unverified Model Downloads (MEDIUM Severity)

**Issue:** Pre-trained models are downloaded from external servers without cryptographic verification.

**Affected Code:**

#### OpenAI Models (Azure Blob Storage)
```python
# runners/diffusion.py:130-145
ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (...))
if not os.path.exists(ckpt):
    download('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (...), ckpt)
```

**Risk:** 
- No SHA256 or MD5 verification
- Man-in-the-middle attacks could serve malicious model weights
- Compromised Azure account could replace models
- Malicious model weights could execute arbitrary code via pickle exploits

#### CelebA-HQ Model (AWS S3 - Test Bucket!)
```python
# runners/diffusion.py:111-113
ckpt = os.path.join(self.args.exp, "logs/celeba/celeba_hq.ckpt")
if not os.path.exists(ckpt):
    download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', ckpt)
```

**Risk:**
- Bucket name contains "test-12345" - suggests temporary/unsecured bucket
- No checksum verification
- AWS S3 buckets can be public or have misconfigured permissions
- Unknown ownership and trustworthiness

**Remediation:**

```python
# Add to functions/ckpt_util.py

CHECKPOINT_HASHES = {
    '256x256_diffusion_uncond.pt': 'sha256:abc123...xyz',  # Replace with actual hash
    'celeba_hq.ckpt': 'sha256:def456...uvw',
    # ... add all checkpoints
}

import hashlib

def verify_checkpoint(path, expected_hash):
    """Verify file SHA256 hash."""
    if not expected_hash.startswith('sha256:'):
        raise ValueError("Only SHA256 supported")
    
    expected = expected_hash.split(':')[1]
    sha256 = hashlib.sha256()
    
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    
    actual = sha256.hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch! Expected {expected}, got {actual}")
    
    return True

# Then in download():
def download(url, local_path, chunk_size=1024, expected_hash=None):
    # ... existing download code ...
    
    if expected_hash:
        verify_checkpoint(local_path, expected_hash)
```

---

### 2. Pickle Deserialization Vulnerability (HIGH Severity)

**Issue:** PyTorch models are loaded using `torch.load()` without `weights_only=True` parameter.

**Affected Code:**

```python
# runners/diffusion.py:116, 139
model.load_state_dict(torch.load(ckpt, map_location=self.device))
```

**Risk:**
- Pickle can execute arbitrary Python code during deserialization
- If checkpoint files are compromised, code execution is possible
- PyTorch's `torch.load()` uses pickle internally

**Remediation:**

```python
# Use PyTorch 1.13+ safe loading (if available)
state_dict = torch.load(ckpt, map_location=self.device, weights_only=True)

# OR for older PyTorch versions, use safetensors library:
# pip install safetensors
from safetensors.torch import load_file
state_dict = load_file(ckpt)
model.load_state_dict(state_dict)
```

---

### 3. External Dependency Risks (MEDIUM Severity)

**Issue:** Project depends on multiple external servers for critical assets.

**External Resources:**

| Resource | URL | Purpose | Mitigation Status |
|----------|-----|---------|-------------------|
| OpenAI Models | `openaipublic.blob.core.windows.net` | Diffusion models | ❌ No verification |
| CelebA-HQ | `image-editing-test-12345.s3-us-west-2.amazonaws.com` | Face model | ❌ Test bucket! |
| Heidelberg DDPM | `heibox.uni-heidelberg.de` | LSUN/CIFAR models | ✅ MD5 verified |
| DDRM Datasets | `github.com/jiamings/ddrm-exp-datasets` | Demo images | ⚠️ Git clone |
| ImageNet List | `github.com/XingangPan/deep-generative-prior` | Metadata | ⚠️ Raw GitHub |

**Risks:**
- Single point of failure (server downtime breaks experiments)
- Potential for supply chain attacks
- GitHub raw files can change without notice
- S3 test buckets may be deleted without warning

**Remediation:**

1. **Mirror critical assets** to institutional/personal storage
2. **Add checksums** for all downloaded files
3. **Version pin** all external resources
4. **Implement retry logic** with exponential backoff
5. **Cache downloads** to avoid repeated requests

```python
# Example caching strategy
import os
from urllib.parse import urlparse

def cached_download(url, cache_dir='~/.cache/ddrm'):
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Use URL hash as filename to avoid collisions
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    cached_path = os.path.join(cache_dir, url_hash)
    
    if os.path.exists(cached_path):
        print(f"Using cached file: {cached_path}")
        return cached_path
    
    # Download and cache
    download(url, cached_path)
    return cached_path
```

---

## 🔒 Privacy Concerns

### 1. Dataset Licensing and Attribution

**Datasets Used:**

| Dataset | License | Restrictions | Attribution Required |
|---------|---------|--------------|---------------------|
| ImageNet | Custom | Academic use, no redistribution | Yes |
| CelebA | Custom | Non-commercial, academic use | Yes |
| LSUN | Unspecified | Check per-category | Yes |

**Issues:**
- No LICENSE file documenting dataset/model sources
- README.md doesn't mention licensing restrictions
- Users may inadvertently violate licenses

**Remediation:**

Create `LICENSE.md`:

```markdown
# License and Attribution

## Code License
[Specify: MIT, Apache 2.0, or other]

## Model Licenses
- OpenAI Guided Diffusion Models: [Check OpenAI's terms]
- DDPM Models (Heidelberg): [Check paper/repo]
- CelebA-HQ Model: Unknown (hosted on test S3 bucket)

## Dataset Licenses
- ImageNet: Academic use only, no redistribution
  - Cite: Deng et al., ImageNet: A Large-Scale Hierarchical Image Database, CVPR 2009
- CelebA: Non-commercial use only
  - Cite: Liu et al., Deep Learning Face Attributes in the Wild, ICCV 2015
- LSUN: Check individual category licenses
  - Cite: Yu et al., LSUN: Construction of a Large-scale Image Dataset, arXiv 2015

## Usage Restrictions
1. **Academic Research Only** - Do not use for commercial purposes
2. **Cite Original Papers** - Respect attribution requirements
3. **No Model Redistribution** - Download from original sources
```

### 2. Sensitive Data Handling

**Issue:** CelebA dataset contains faces of real people.

**Risks:**
- GDPR/privacy law compliance if processing EU citizen data
- Facial recognition concerns
- Potential misuse for deepfakes

**Recommendations:**
1. Add ethics statement to README
2. Warn users about responsible use
3. Don't redistribute processed face images
4. Follow your institution's IRB/ethics board guidelines

---

## 🌐 Network Security

### 1. HTTPS Verification

**Current Status:** ✅ All URLs use HTTPS

**Code Review:**
```python
# functions/ckpt_util.py:38-48
with requests.get(url, stream=True) as r:
    # Implicitly uses verify=True (default)
```

**Recommendation:** Make SSL verification explicit:

```python
with requests.get(url, stream=True, verify=True, timeout=30) as r:
    r.raise_for_status()  # Check HTTP errors
```

### 2. URL Injection

**Current Status:** ✅ No user-controlled URLs found

All URLs are hardcoded in source or configs. Low risk of injection attacks.

---

## 🔧 Secure Configuration

### 1. File System Permissions

**Issue:** Downloaded files not checked for permissions.

**Recommendation:**

```python
import os

def secure_download(url, path):
    download(url, path)
    
    # Set restrictive permissions (owner read-write only)
    os.chmod(path, 0o600)  # -rw-------
```

### 2. Directory Traversal

**Current Status:** ✅ Low risk

Paths constructed from args/config but not from external user input during runtime.

**Existing Safeguards:**
```python
# main.py:119-127
if not os.path.exists(args.image_folder):
    os.makedirs(args.image_folder)
```

**Recommendation:** Add path validation:

```python
import os.path

def safe_path(base, user_path):
    """Prevent directory traversal."""
    resolved = os.path.abspath(os.path.join(base, user_path))
    if not resolved.startswith(os.path.abspath(base)):
        raise ValueError("Invalid path: directory traversal detected")
    return resolved
```

---

## 📊 Summary and Risk Matrix

| Issue | Severity | Likelihood | Impact | Priority |
|-------|----------|------------|--------|----------|
| Unverified model downloads | MEDIUM | MEDIUM | HIGH | **P1** |
| Pickle deserialization | HIGH | LOW | CRITICAL | **P1** |
| External dependency failures | MEDIUM | HIGH | MEDIUM | **P2** |
| Dataset license violations | LOW | MEDIUM | MEDIUM | **P2** |
| Privacy/GDPR concerns | LOW | LOW | HIGH | **P3** |

**Risk Score Calculation:**
- **P1 (Critical):** Address immediately before production use
- **P2 (High):** Address before public release
- **P3 (Medium):** Address in next maintenance cycle

---

## ✅ Actionable Remediation Plan

### Phase 1: Immediate (Before Running Code)

1. ✅ **Review this document** - Understand all risks
2. ✅ **Use isolated environment** - Conda/virtualenv, not system Python
3. ✅ **Download only from official sources** - Verify URLs match README.md
4. ⚠️ **Monitor downloads** - Check file sizes match expectations

### Phase 2: Short-term (Before Sharing)

1. **Add checksum verification** - Implement SHA256 checks for all downloads
2. **Create LICENSE.md** - Document all licenses and attributions
3. **Add ethics statement** - Responsible AI use guidelines
4. **Implement retry logic** - Handle network failures gracefully

### Phase 3: Long-term (Production Hardening)

1. **Mirror all assets** - Host critical files on institutional servers
2. **Use safetensors** - Eliminate pickle vulnerability
3. **Add integrity monitoring** - Detect file tampering
4. **Implement audit logging** - Track all downloads and model loads
5. **Security scan dependencies** - Use tools like `safety`, `bandit`

---

## 🛠️ Security Tools and Commands

### Check for Known Vulnerabilities

```bash
# Install security scanners
pip install safety bandit

# Check dependencies for CVEs
safety check

# Static security analysis
bandit -r . -ll

# Check for secrets (install separately)
# pip install detect-secrets
detect-secrets scan --baseline .secrets.baseline
```

### Verify Downloaded Files

```bash
# Compute SHA256 hash
shasum -a 256 exp/logs/imagenet/256x256_diffusion_uncond.pt

# Compare with published hash (if available)
echo "EXPECTED_HASH exp/logs/imagenet/256x256_diffusion_uncond.pt" | shasum -a 256 -c
```

### Monitor Network Activity

```bash
# On Linux: Monitor outbound connections
sudo netstat -tnp | grep python

# On macOS: Use Activity Monitor or lsof
lsof -i -P | grep python
```

---

## 📞 Reporting Security Issues

If you discover security vulnerabilities in this code:

1. **Do NOT open public GitHub issues** (responsible disclosure)
2. **Contact the original authors** via email (check paper for contacts)
3. **Contact your institution's security team** if handling sensitive data
4. **Document the issue** privately with steps to reproduce

---

## 📚 References and Resources

- [PyTorch Security Best Practices](https://pytorch.org/docs/stable/notes/security.html)
- [OWASP Machine Learning Security Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)
- [Model Serialization Security (Safetensors)](https://github.com/huggingface/safetensors)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [ImageNet Terms of Use](https://image-net.org/download)
- [CelebA Dataset Page](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

---

**End of SECURITY_WARNINGS.md**

*This document was generated via static code analysis. Actual security posture may vary depending on deployment environment and usage patterns. Always perform your own security assessment before production deployment.*

