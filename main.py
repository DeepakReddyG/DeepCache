import argparse
import os
import time

import torch
from torchvision.utils import save_image

from DeepCache import DeepCacheSDHelper

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, ".hf-cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HF_CACHE_DIR, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_CACHE_DIR, "transformers"))


def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_runtime_device():
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default = "sd1.5")
    parser.add_argument("--prompt", type=str, default='a photo of an astronaut on a moon')
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cache_interval", type=int, default=3)
    parser.add_argument("--cache_branch_id", type=int, default=0)
    args = parser.parse_args()
    device, dtype = get_runtime_device()
    fp16_kwargs = {"variant": "fp16"} if dtype == torch.float16 else {}

    if args.model_type.lower() == 'sdxl':
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-1.0',
            torch_dtype=dtype,
            use_safetensors=True,
            cache_dir=HF_CACHE_DIR,
            **fp16_kwargs,
        ).to(device)
    elif args.model_type.lower() == 'sd1.5':
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5', torch_dtype=dtype, cache_dir=HF_CACHE_DIR
        ).to(device)
    elif args.model_type.lower() == 'sd2.1':
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            'stabilityai/stable-diffusion-2-1-base', torch_dtype=dtype, cache_dir=HF_CACHE_DIR
        ).to(device)
    elif args.model_type.lower() == 'svd':
        from diffusers import StableVideoDiffusionPipeline 
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=dtype,
            cache_dir=HF_CACHE_DIR,
            **fp16_kwargs,
        )
        if device == "cuda:0":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
    elif args.model_type.lower() == 'sd-inpaint':
        from diffusers import StableDiffusionInpaintPipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            'runwayml/stable-diffusion-inpainting', torch_dtype=dtype, cache_dir=HF_CACHE_DIR
        ).to(device)
    elif args.model_type.lower() == 'sdxl-inpaint':
        from diffusers import StableDiffusionXLInpaintPipeline
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            'diffusers/stable-diffusion-xl-1.0-inpainting-0.1', torch_dtype=dtype, cache_dir=HF_CACHE_DIR
        ).to(device)
    elif args.model_type.lower() == 'sd-img2img':
        from diffusers import StableDiffusionImg2ImgPipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=dtype,
            use_safetensors=True,
            cache_dir=HF_CACHE_DIR,
            **fp16_kwargs,
        )
        if device == "cuda:0":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
    else:
        raise NotImplementedError

    prompt = args.prompt
    seed = args.seed

    if args.model_type.lower() == 'svd':
        import time
        from diffusers.utils import load_image, export_to_video
        image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")

        print("Running Original Pipeline...")
        set_random_seed(42)
        start_time = time.time()
        frames = pipe(
            image, 
            decode_chunk_size=8,
        ).frames[0]
        origin_time = time.time() - start_time
        export_to_video(frames, "{}_origin.mp4".format('rocket'), fps=7)

        print("Enable DeepCache...")
        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(
            cache_interval=args.cache_interval,
            cache_branch_id=args.cache_branch_id,
        )
        helper.enable()

        print("Running Pipeline with DeepCache...")
        set_random_seed(42)
        start_time = time.time()
        frames = pipe(
            image, 
            decode_chunk_size=8,
        ).frames[0]
        deepcache_time = time.time() - start_time
        export_to_video(frames, "{}_deepcache.mp4".format('rocket'), fps=7)
        helper.disable()
    
    elif 'inpaint' in args.model_type.lower():
        from diffusers.utils import load_image
        img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
        image = load_image(img_url)
        mask_image = load_image(mask_url)
        prompt = "a tiger sitting on a park bench"

        # warmup
        _ = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]

        set_random_seed(seed)
        start_time = time.time()
        image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
        origin_time = time.time() - start_time
        image.save("inpaint_origin.png")

        print("Enable DeepCache...")
        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(
            cache_interval=args.cache_interval,
            cache_branch_id=args.cache_branch_id,
        )
        helper.enable()

        print("Running Pipeline with DeepCache...")
        start_time = time.time()
        set_random_seed(seed)
        deepcache_image= pipe(
            prompt=prompt,image=image, mask_image=mask_image
        ).images[0]
        deepcache_time = time.time() - start_time
        deepcache_image.save("inpaint_deepcache.png")

    elif args.model_type.lower() == 'sd-img2img':
        from diffusers.utils import make_image_grid, load_image
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
        init_image = load_image(url)
        init_image.save("img2img_init.png")

        prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
        # Warmup
        image = pipe(prompt, image=init_image).images[0]

        set_random_seed(seed)
        start_time = time.time()
        image = pipe(prompt, image=init_image).images[0]
        origin_time = time.time() - start_time
        image.save("img2img_ori.png")

        print("Enable DeepCache...")
        helper = DeepCacheSDHelper(pipe=pipe)
        start_time = time.time()
        helper.set_params(
            cache_interval=args.cache_interval,
            cache_branch_id=args.cache_branch_id,
        )
        helper.enable()

        print("Running Pipeline with DeepCache...")
        set_random_seed(seed)
        start_time = time.time()
        deepcache_img = pipe(prompt, image=init_image).images[0]
        deepcache_time = time.time() - start_time
        deepcache_img.save("img2img_deepcache.png")
    else:
        import time
        print("Warmup GPU...")
        for _ in range(1):
            set_random_seed(seed)
            _ = pipe(prompt)

        print("Running Original Pipeline...")
        set_random_seed(seed)
        start_time = time.time()
        pipeline_output = pipe(
            prompt, 
            output_type='pt'
        ).images[0]
        origin_time = time.time() - start_time
        save_image([pipeline_output], 'text2img_origin.png')

        print("Enable DeepCache...")
        helper = DeepCacheSDHelper(pipe=pipe)
        start_time = time.time()
        helper.set_params(
            cache_interval=args.cache_interval,
            cache_branch_id=args.cache_branch_id,
        )
        helper.enable()

        print("Running Pipeline with DeepCache...")
        set_random_seed(seed)
        deepcache_pipeline_output = pipe(
                prompt,
                output_type='pt'
        ).images[0]
        deepcache_time = time.time() - start_time

        save_image([deepcache_pipeline_output], 'text2img_deepcache.png')
        helper.disable()

    print("Done! Original Pipeline: {:.2f} seconds, DeepCache: {:.2f} seconds. Speedup Ratio = {:.2f}".format(origin_time, deepcache_time, origin_time/deepcache_time))
