# Static Analysis Methodology Documentation
# Generated: December 4, 2025
# Project: DDRM (Denoising Diffusion Restoration Models)

This document describes the static analysis commands and techniques used to
generate PROJECT_OVERVIEW.md and project_summary.json without executing any code.

## Tools and Techniques Used

### 1. File Discovery
Commands used to identify all relevant files in the repository:

```bash
# Find all Python files
find . -name "*.py" -type f

# Find all Jupyter notebooks
find . -name "*.ipynb" -type f

# Find all YAML configuration files
find . -name "*.yml" -type f
find . -name "*.yaml" -type f

# Find all NumPy binary files
find . -name "*.npy" -type f

# List directory structure
tree -L 3
# OR
find . -type d | head -50
```

### 2. Text Content Analysis

#### Reading entire files:
```bash
# View Python files
cat main.py
cat runners/diffusion.py
cat functions/denoising.py
cat models/diffusion.py

# View configuration files
cat environment.yml
cat configs/imagenet_256.yml
cat configs/celeba_hq.yml

# View README
cat README.md
```

#### Reading specific line ranges (simulation):
```bash
# Lines 1-100 of a file
head -n 100 main.py

# Lines 100-200 of a file
sed -n '100,200p' runners/diffusion.py

# Lines 200-end of a file
tail -n +200 functions/svd_replacement.py
```

### 3. Pattern Matching and Search

#### Find import statements:
```bash
# All imports across Python files
grep -r "^import " *.py */*.py
grep -r "^from " *.py */*.py

# Specific library usage
grep -r "torch\|pytorch" *.py */*.py
grep -r "numpy" *.py */*.py
grep -r "PIL\|Image" *.py */*.py
```

#### Find class definitions:
```bash
# All class definitions
grep -rn "^class " *.py */*.py

# Find classes inheriting from specific base
grep -rn "class.*nn.Module" *.py */*.py
grep -rn "class.*H_functions" functions/
```

#### Find function definitions:
```bash
# All function definitions
grep -rn "^def " *.py */*.py

# Find specific functions
grep -rn "def sample" runners/
grep -rn "def forward" models/ guided_diffusion/
```

#### Find method calls:
```bash
# Model loading
grep -rn "load_state_dict\|torch.load" *.py */*.py

# Model inference
grep -rn "model(" *.py */*.py

# File I/O
grep -rn "save_image\|torch.save" *.py */*.py
grep -rn "open(.*['\"]r" *.py */*.py
```

#### Find URLs and external resources:
```bash
# HTTP/HTTPS URLs
grep -rn "https\?://" *.py */*.py

# S3 buckets
grep -rn "s3\|amazonaws" *.py */*.py

# GitHub repositories
grep -rn "github.com" *.py */*.py *.ipynb
```

### 4. Jupyter Notebook Analysis

#### Extract notebook structure (without nbformat library):
```bash
# View raw notebook JSON
cat Untitled5.ipynb | jq '.cells[] | {cell_type, source}'

# Count cells
cat Untitled5.ipynb | jq '.cells | length'

# Extract code cells only
cat Untitled5.ipynb | jq '.cells[] | select(.cell_type=="code") | .source'

# Extract markdown cells
cat Untitled5.ipynb | jq '.cells[] | select(.cell_type=="markdown") | .source'
```

#### Extract notebook structure (with Python):
```python
import json

with open('Untitled5.ipynb') as f:
    nb = json.load(f)

# Print cell types and first line
for i, cell in enumerate(nb['cells']):
    print(f"Cell {i}: {cell['cell_type']}")
    if cell['source']:
        print(f"  First line: {cell['source'][0][:60]}")
```

### 5. Dependency Analysis

#### Conda environment inspection:
```bash
# Extract package names and versions
grep "^  - " environment.yml | sort

# Find GPU-specific packages
grep -i "cuda\|gpu" environment.yml

# Find Linux-specific packages
grep -i "linux\|libgcc\|ld_impl" environment.yml

# Count total packages
grep "^  - " environment.yml | wc -l
```

#### Find pip packages:
```bash
# Extract pip section
sed -n '/^  - pip:/,/^[^ ]/p' environment.yml
```

### 6. Configuration File Analysis

#### Parse YAML configs:
```bash
# View specific config sections
grep -A 10 "^data:" configs/imagenet_256.yml
grep -A 10 "^model:" configs/celeba_hq.yml
grep -A 10 "^diffusion:" configs/bedroom.yml

# Extract specific values
grep "batch_size:" configs/*.yml
grep "image_size:" configs/*.yml
grep "timesteps:" configs/*.yml
```

#### Using Python for YAML parsing:
```python
import yaml

with open('configs/imagenet_256.yml') as f:
    config = yaml.safe_load(f)

print("Image size:", config['data']['image_size'])
print("Batch size:", config['sampling']['batch_size'])
print("Model type:", config['model']['type'])
```

### 7. Code Structure Analysis

#### Count lines of code:
```bash
# Total Python LOC
find . -name "*.py" -exec wc -l {} + | tail -1

# Per-directory LOC
wc -l functions/*.py
wc -l models/*.py
wc -l runners/*.py
```

#### Find docstrings:
```bash
# Functions with docstrings (triple-quoted strings after def)
grep -A 3 "^def " *.py */*.py | grep '"""'

# Classes with docstrings
grep -A 3 "^class " *.py */*.py | grep '"""'
```

#### Identify code patterns:
```bash
# Training loops (NOT FOUND in this repo)
grep -rn "optimizer.step()\|loss.backward()" *.py */*.py

# Evaluation metrics
grep -rn "psnr\|ssim\|fid\|accuracy" *.py */*.py

# Device handling
grep -rn "\.to(device)\|\.cuda()\|\.cpu()" *.py */*.py
```

### 8. Call Graph Construction

#### Manual tracing (example):
```bash
# Start from main
grep -n "def main" main.py
# -> calls parse_args_and_config()
grep -n "parse_args_and_config" main.py
# -> creates Diffusion runner
grep -n "Diffusion(" main.py
# -> calls runner.sample()
grep -n "def sample" runners/diffusion.py
# -> calls sample_sequence()
grep -n "def sample_sequence" runners/diffusion.py
# -> calls efficient_generalized_steps()
grep -n "efficient_generalized_steps" runners/diffusion.py functions/denoising.py
```

### 9. Data Flow Tracing

#### Identify I/O operations:
```bash
# File reads
grep -rn "open(.*'r'\|torch.load\|Image.open\|np.load" *.py */*.py

# File writes
grep -rn "open(.*'w'\|torch.save\|save_image\|np.save" *.py */*.py

# Network I/O
grep -rn "requests.get\|urllib\|download" *.py */*.py
```

#### Trace data transformations:
```bash
# Dataset loading
grep -rn "get_dataset\|DataLoader" *.py */*.py

# Transforms
grep -rn "data_transform\|inverse_data_transform" *.py */*.py

# Model inputs/outputs
grep -rn "model(.*)" runners/ functions/
```

### 10. Security Analysis

#### Find hardcoded credentials (NONE FOUND):
```bash
# API keys, passwords, tokens
grep -rni "api_key\|password\|token\|secret\|credential" *.py */*.py

# AWS keys
grep -rni "aws_access_key\|aws_secret" *.py */*.py
```

#### Find external URLs:
```bash
# All HTTP/HTTPS URLs
grep -rnoP 'https?://[^\s"\047]+' *.py */*.py *.ipynb | sort -u
```

## Static Analysis Limitations

The following CANNOT be determined without execution:

1. **Runtime values**: Actual batch sizes, image counts, tensor shapes
2. **Performance**: Execution time, memory usage, GPU utilization
3. **Numerical behavior**: NaN/Inf occurrences, convergence, accuracy
4. **Dynamic imports**: `importlib` or `__import__()` usage
5. **Generated configs**: Runtime config generation or modification
6. **Network behavior**: Actual download success, response times
7. **File system state**: Whether files exist, are readable, correct format
8. **Environment variables**: Values of os.environ lookups
9. **Conditional execution**: Which branches actually execute
10. **Error handling**: What exceptions occur in practice

## Reproducibility

To reproduce this analysis:

1. **Clone the repository** (if not already local)
2. **Navigate to project root**
3. **Run the commands above** in sequence or as needed
4. **Use jq for JSON parsing**: `brew install jq` (macOS) or `apt install jq` (Linux)
5. **Use Python for complex parsing**: Python 3.7+ with json, yaml libraries
6. **Combine results** into structured documentation

## Tools Summary

- **grep**: Pattern matching in files
- **find**: File discovery
- **sed**: Stream editing, line extraction
- **wc**: Word/line counting
- **jq**: JSON querying and formatting
- **cat/head/tail**: File viewing
- **tree**: Directory structure visualization
- **Python**: Complex parsing (JSON, YAML, AST analysis)

## Notes

- All analysis performed WITHOUT executing Python code or notebooks
- No imports of project modules
- No model loading or inference
- No GPU/CUDA required for analysis
- Safe to run on any system with basic Unix tools

## References

- [jq Manual](https://stedolan.github.io/jq/manual/)
- [grep Manual](https://www.gnu.org/software/grep/manual/grep.html)
- [Python AST Module](https://docs.python.org/3/library/ast.html) (for advanced analysis)
- [nbformat Documentation](https://nbformat.readthedocs.io/) (for notebook parsing)

---
End of README_ANALYSIS.txt

