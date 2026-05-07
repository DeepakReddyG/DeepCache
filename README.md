# DeepCache Replication — CS Final Project

> **Replication of:** *DeepCache: Accelerating Diffusion Models for Free*  
> Xinyin Ma, Gongfan Fang, Xinchao Wang — NUS (CVPR 2024)  
> Original paper: [arXiv:2312.00858](https://arxiv.org/abs/2312.00858) · Original repo: [horseee/DeepCache](https://github.com/horseee/DeepCache)

---

## What We Replicated

We independently replicated the core DeepCache results using the official implementation. DeepCache accelerates diffusion model inference by caching temporally redundant high-level U-Net features across consecutive denoising steps, requiring **no retraining**.

**Our key results (matching the paper):**

| Configuration | Time | Throughput | Speedup |
|---|---|---|---|
| Original SD v1.5 (50 PLMS steps) | 7.30s | 7.10 it/s | 1.00× |
| DeepCache SD v1.5 (N=5) | **3.17s** | **17.22 it/s** | **2.30×** |

CLIP Score drop: only **0.05** (29.51 → 29.46 on PartiPrompts).

---

## Repository Structure

```
.
├── README.md                  ← this file
├── DeepCache.ipynb            ← main experiment notebook (run on Google Colab)
├── requirements.txt           ← pip dependencies
├── environment.yml            ← conda environment (alternative)
└── experiments/
    ├── README.md              ← DDPM and LDM reproduction commands
    ├── generate.py            ← Stable Diffusion sample generation script
    ├── clip_score.py          ← CLIP Score evaluation script
    └── ldm/
        └── environment.yaml   ← separate conda env for LDM experiments
```

---

## Dependencies

**Hardware:** NVIDIA GPU with ≥8 GB VRAM (experiments run on a T4 GPU via Google Colab).  
**Python:** 3.10+

### Key packages

| Package | Version |
|---|---|
| `diffusers` | 0.24.0 |
| `torch` | ≥2.0.0 (CUDA) |
| `transformers` | ≥4.35.0 |
| `accelerate` | ≥0.24.0 |
| `matplotlib` | ≥3.7.0 |

---

## Setup Instructions

### Option A — Google Colab (recommended, matches our setup)

Open `DeepCache.ipynb` directly in [Google Colab](https://colab.research.google.com/). Make sure to select a **T4 GPU runtime** (`Runtime → Change runtime type → T4 GPU`), then run all cells in order.

The notebook handles all installation steps automatically.

### Option B — pip (local)

```bash
# 1. Clone the official DeepCache repo (required for the DeepCache pipeline module)
git clone https://github.com/horseee/DeepCache.git
cd DeepCache

# 2. Install dependencies
pip install diffusers==0.24.0
pip install matplotlib transformers accelerate
pip install -e .
```

Or using the requirements file at the root of this repo:

```bash
pip install -r requirements.txt
```

### Option C — Conda

```bash
conda env create -f environment.yml
conda activate deepcache
pip install -e .
```

---

## Reproducing Our Results

### Primary experiment — SD v1.5 timing benchmark (our 2.30× result)

**Fastest path:** open `DeepCache.ipynb` in Google Colab and run all cells.  
The notebook reproduces the exact timing result reported in our paper:

- **Cell 0–3:** install packages and clone the repo  
- **Cell 4–9:** run the original SD v1.5 pipeline, record baseline time  
- **Cell 10–13:** enable DeepCache (`cache_interval=5`), run accelerated pipeline  
- **Cell 14:** display side-by-side output images with timing labels

Expected output (matching our measured results):

```
Original Pipeline:  7.30 seconds  (7.10 it/s)
DeepCache Pipeline: 3.17 seconds  (17.22 it/s)
Speedup Ratio = 2.30×
```

### Stable Diffusion v1.5 — standalone script

```bash
# Baseline vs DeepCache visual comparison
python main.py \
  --model_type sd1.5 \
  --prompt "a photo of an astronaut on a moon" \
  --seed 42 \
  --cache_interval 5 \
  --cache_branch_id 0
```

Outputs: `text2img_origin.png` and `text2img_deepcache.png`.

### Benchmark table (timing + similarity metrics)

```bash
python evaluate.py \
  --model runwayml/stable-diffusion-v1-5 \
  --num_prompts 5 \
  --cache_interval 5 \
  --cache_branch_id 0
```

Outputs: `benchmark_results.csv` containing per-prompt and average speedup, PSNR, and LPIPS scores.

### One-command full reproduction

```bash
bash reproduce_results.sh
```

This runs the visual comparison and benchmark table generation together.

### LDM-4-G on ImageNet (Table 2 in our report)

See `experiments/README.md` for the full set of DDPM and LDM reproduction commands. The LDM experiments require a separate Conda environment:

```bash
conda env create -f experiments/ldm/environment.yaml
conda activate ldm
# then follow experiments/README.md
```

---

## Using DeepCache in Your Own Code

```python
import torch
from diffusers import StableDiffusionPipeline
from DeepCache import DeepCacheSDHelper

# Load the standard pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    torch_dtype=torch.float16
).to("cuda:0")

# Wrap with DeepCache
helper = DeepCacheSDHelper(pipe=pipe)
helper.set_params(
    cache_interval=5,   # N — our primary setting
    cache_branch_id=0,  # skip branch m
)
helper.enable()

# Generate
image = pipe("a photo of an astronaut on a moon", output_type='pt').images[0]

helper.disable()
```

**Key parameters:**

| Parameter | Description | Our setting |
|---|---|---|
| `cache_interval` | N — how many steps reuse cached features | `5` |
| `cache_branch_id` | skip branch index m (deeper = more cached) | `0` |
| `uniform` | whether to use uniform (True) or non-uniform (False) scheduling | `True` |

---

## Key Findings

- At `cache_interval=5`, DeepCache achieves **2.30× wall-clock speedup** on SD v1.5 with only a **0.05 CLIP Score drop** — matching the paper exactly.
- DeepCache outperforms all BK-SDM variants (which require retraining) on both speed and CLIP Score simultaneously.
- Larger `N` gives more speedup but degrades quality; `N ∈ [3, 5]` is the practical sweet spot.
- DeepCache is additive with fast samplers (PLMS, DDIM) — they can be stacked for compounding gains.

---

## Supported Models

The official DeepCache codebase supports:

- Stable Diffusion v1.5
- Stable Diffusion v2.1
- Stable Diffusion XL (SDXL)
- Stable Video Diffusion (SVD)
- SD Inpainting / Img2Img pipelines
- DDPM (CIFAR-10, LSUN)
- LDM-4-G (ImageNet)

---

## Citation

If you use the original DeepCache method, please cite:

```bibtex
@inproceedings{ma2023deepcache,
  title     = {DeepCache: Accelerating Diffusion Models for Free},
  author    = {Ma, Xinyin and Fang, Gongfan and Wang, Xinchao},
  booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024}
}
```

---

## Contribution Statement

| Member | Contribution |
|---|---|
| **Student A** | Environment setup, notebook implementation, SD v1.5 timing experiments, CLIP Score evaluation |
| **Student B** | LDM-4-G ImageNet experiments, cache interval sweep (N ∈ {2,3,5,10,20}), benchmark table generation |
| **Student C** | Literature review, report writing, fast-sampler compatibility analysis, repository documentation |
