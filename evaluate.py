import argparse
import time
import csv
import os
import torch
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from DeepCache import DeepCacheSDHelper

# For evaluation metrics
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("WARNING: 'lpips' package not found. Run `pip install lpips` for LPIPS metric computation.")

def set_seed(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def calculate_psnr(img1, img2):
    # img1 and img2 are PIL images
    img1_np = np.array(img1).astype(np.float32) / 255.0
    img2_np = np.array(img2).astype(np.float32) / 255.0
    mse = np.mean((img1_np - img2_np) ** 2)
    if mse == 0:
        return float('inf')
    return -10 * np.log10(mse)

def pil_to_tensor(img):
    # Convert PIL Image to PyTorch tensor [-1, 1] for LPIPS
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = img_np * 2.0 - 1.0
    # HWC to CHW
    img_np = img_np.transpose(2, 0, 1)
    return torch.from_numpy(img_np).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--num_prompts", type=int, default=5, help="Number of prompts to evaluate")
    parser.add_argument("--cache_interval", type=int, default=3)
    parser.add_argument("--cache_branch_id", type=int, default=0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda:0"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"Loading Pipeline from {args.model} on {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device)

    # Some sample prompts for evaluation
    sample_prompts = [
        "a photo of an astronaut riding a horse on mars",
        "A high tech solarpunk utopia in the Amazon rainforest",
        "A cute corgi wearing a top hat and a monocle",
        "A beautiful landscape of a mountain range at sunset, highly detailed",
        "A cyberpunk city street at night with neon lights and rain",
        "An oil painting of a cozy cabin in the snowy woods",
        "A futuristic sports car driving on a glowing bridge",
        "A portrait of an old wizard with a long beard and a glowing staff",
        "A spaceship flying through an asteroid field",
        "A fantasy castle floating in the clouds"
    ]
    
    prompts_to_test = sample_prompts[:args.num_prompts]
    
    if LPIPS_AVAILABLE:
        print("Loading LPIPS VGG model...")
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    baseline_times = []
    deepcache_times = []
    psnr_scores = []
    lpips_scores = []

    print(f"\nStarting Evaluation for {len(prompts_to_test)} prompts...\n")

    for i, prompt in enumerate(prompts_to_test):
        print(f"[{i+1}/{len(prompts_to_test)}] Prompt: '{prompt}'")
        
        # --- Baseline ---
        set_seed(42)
        start_time = time.time()
        baseline_img = pipe(prompt).images[0]
        t_base = time.time() - start_time
        baseline_times.append(t_base)
        
        # --- DeepCache ---
        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(
            cache_interval=args.cache_interval,
            cache_branch_id=args.cache_branch_id,
        )
        helper.enable()
        
        set_seed(42)
        start_time = time.time()
        deepcache_img = pipe(prompt).images[0]
        t_deep = time.time() - start_time
        deepcache_times.append(t_deep)
        
        helper.disable()
        
        # --- Metrics ---
        psnr = calculate_psnr(baseline_img, deepcache_img)
        psnr_scores.append(psnr)
        
        l_score = -1.0
        if LPIPS_AVAILABLE:
            t1 = pil_to_tensor(baseline_img).to(device)
            t2 = pil_to_tensor(deepcache_img).to(device)
            with torch.no_grad():
                l_score = lpips_fn(t1, t2).item()
            lpips_scores.append(l_score)
            
        print(f"  Baseline Time: {t_base:.2f}s | DeepCache Time: {t_deep:.2f}s | PSNR: {psnr:.2f} | LPIPS: {l_score if l_score != -1.0 else 'N/A'}")

    # --- Summary ---
    avg_base_time = np.mean(baseline_times)
    avg_deep_time = np.mean(deepcache_times)
    speedup = avg_base_time / avg_deep_time if avg_deep_time > 0 else 0
    
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model:           {args.model}")
    print(f"Cache Interval:  {args.cache_interval}")
    print(f"Cache Branch ID: {args.cache_branch_id}")
    print(f"Num Prompts:     {len(prompts_to_test)}")
    print("-"*50)
    print(f"Avg Baseline Time:  {avg_base_time:.3f} s/image")
    print(f"Avg DeepCache Time: {avg_deep_time:.3f} s/image")
    print(f"Speedup Ratio:      {speedup:.2f}x")
    print(f"Avg PSNR:           {np.mean(psnr_scores):.2f} dB (Higher is better)")
    if LPIPS_AVAILABLE:
        print(f"Avg LPIPS:          {np.mean(lpips_scores):.4f} (Lower is better, <0.1 is almost identical)")
    print("="*50)

    # --- Export to CSV ---
    csv_file = "benchmark_results.csv"
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model", "Cache Interval", "Cache Branch ID", "Num Prompts", "Avg Baseline Time", "Avg DeepCache Time", "Speedup Ratio", "Avg PSNR", "Avg LPIPS"])
        writer.writerow([
            args.model,
            args.cache_interval,
            args.cache_branch_id,
            len(prompts_to_test),
            f"{avg_base_time:.3f}",
            f"{avg_deep_time:.3f}",
            f"{speedup:.2f}",
            f"{np.mean(psnr_scores):.2f}",
            f"{np.mean(lpips_scores):.4f}" if LPIPS_AVAILABLE else "N/A"
        ])
    print(f"\nResults successfully exported and appended to {csv_file}")

if __name__ == "__main__":
    main()
