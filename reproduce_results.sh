#!/usr/bin/env bash

set -euo pipefail

# Core demo artifact: baseline vs DeepCache image outputs.
python main.py \
  --model_type sd1.5 \
  --prompt "a photo of an astronaut on a moon" \
  --seed 42 \
  --cache_interval 3 \
  --cache_branch_id 0

# Benchmark table artifact: timing and image-similarity metrics.
python evaluate.py \
  --model runwayml/stable-diffusion-v1-5 \
  --num_prompts 5 \
  --cache_interval 3 \
  --cache_branch_id 0

# Paper-style Stable Diffusion experiment artifacts.
python experiments/generate.py \
  --dataset coco2017 \
  --layer 0 \
  --block 0 \
  --update_interval 2 \
  --uniform \
  --steps 50 \
  --batch_size 16

python experiments/clip_score.py coco2017_ckpt
