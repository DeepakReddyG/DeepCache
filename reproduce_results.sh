#!/usr/bin/env bash
# =============================================================================
# reproduce_results.sh
#
# One-command script to regenerate all key figures and tables from the report.
# Verified on Google Colab (Python 3.12, CUDA 12.8).
#
# Usage:
#   bash reproduce_results.sh
#
# Outputs:
#   text2img_origin.png       — baseline SD v1.5 image
#   text2img_deepcache.png    — same image accelerated by DeepCache
#   benchmark_results.csv     — timing + quality metrics table (PSNR, LPIPS)
#   coco2017_ckpt/            — SD experiment samples (CLIP Score evaluation)
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# 0. Sanity checks
# -----------------------------------------------------------------------------
echo "============================================================"
echo " DeepCache – Reproduce Results"
echo "============================================================"

python - <<'EOF'
import sys, torch
print(f"Python  : {sys.version.split()[0]}")
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda or 'not available'}")
print(f"Device  : {'cuda:0 (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
EOF

echo ""

# -----------------------------------------------------------------------------
# 1. Visual comparison  →  text2img_origin.png  &  text2img_deepcache.png
#
#    Generates one baseline image and one DeepCache-accelerated image from the
#    same prompt and seed so quality differences are directly visible.
#    Expected speedup: ~2.3x on SD v1.5 with 50 PLMS steps.
# -----------------------------------------------------------------------------
echo "[Step 1/3] Generating baseline vs DeepCache images (SD v1.5) ..."
python main.py \
  --model_type    sd1.5 \
  --prompt        "a photo of an astronaut on a moon" \
  --seed          42 \
  --cache_interval  3 \
  --cache_branch_id 0

echo "  → Saved: text2img_origin.png, text2img_deepcache.png"
echo ""

# -----------------------------------------------------------------------------
# 2. Benchmark table  →  benchmark_results.csv
#
#    Runs baseline and DeepCache over 5 prompts and records:
#      - Average inference time (seconds/image)
#      - Speedup ratio
#      - PSNR  (higher = more similar to baseline; expected ~30–40 dB)
#      - LPIPS (lower = more perceptually similar; expected <0.1)
#    Results are appended to benchmark_results.csv.
# -----------------------------------------------------------------------------
echo "[Step 2/3] Running benchmark (5 prompts, SD v1.5) ..."
python evaluate.py \
  --model           runwayml/stable-diffusion-v1-5 \
  --num_prompts     5 \
  --cache_interval  3 \
  --cache_branch_id 0

echo "  → Saved / updated: benchmark_results.csv"
echo ""

# -----------------------------------------------------------------------------
# 3. CLIP Score evaluation  →  coco2017_ckpt/  (paper-style SD experiment)
#
#    Step 3a: Generate 50-step DeepCache samples on COCO-2017 captions.
#    Step 3b: Compute CLIP Score over the saved samples.
#    Note: requires the COCO-2017 captions dataset to be reachable by
#          experiments/generate.py (downloaded automatically on first run).
# -----------------------------------------------------------------------------
echo "[Step 3/3] Running COCO-2017 CLIP Score experiment ..."

echo "  [3a] Generating samples ..."
python experiments/generate.py \
  --dataset         coco2017 \
  --layer           0 \
  --block           0 \
  --update_interval 2 \
  --uniform \
  --steps           50 \
  --batch_size      16

echo "  [3b] Computing CLIP Score ..."
python experiments/clip_score.py coco2017_ckpt

echo "  → Results printed above; samples saved in coco2017_ckpt/"
echo ""

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo "============================================================"
echo " All steps complete."
echo " Key outputs:"
echo "   text2img_origin.png      – baseline image"
echo "   text2img_deepcache.png   – DeepCache image"
echo "   benchmark_results.csv    – speedup / PSNR / LPIPS table"
echo "   coco2017_ckpt/           – CLIP Score samples"
echo "============================================================"
