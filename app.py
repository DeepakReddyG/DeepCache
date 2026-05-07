import streamlit as st
import time
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from DeepCache import DeepCacheSDHelper

def generate_heatmap(img1, img2):
    # Convert to float to avoid overflow
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    # Calculate absolute difference
    diff = np.abs(arr1 - arr2)
    # Average across RGB channels to get a single intensity
    diff_gray = np.mean(diff, axis=-1)
    # Amplify difference by 5x and clip
    diff_gray = np.clip(diff_gray * 5, 0, 255).astype(np.uint8)
    # Create an RGB image where difference maps to Red channel
    heatmap = np.zeros_like(arr1, dtype=np.uint8)
    heatmap[:, :, 0] = diff_gray
    return Image.fromarray(heatmap)

st.set_page_config(page_title="DeepCache Web Demo", layout="wide")

st.title("DeepCache: Accelerating Diffusion Models for Free 🚀")
st.markdown("This interactive demo compares the standard Stable Diffusion pipeline with the DeepCache-accelerated pipeline.")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

prompt = st.sidebar.text_area("Prompt", value="a photo of an astronaut on a moon", height=100)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)
model_id = st.sidebar.selectbox("Model", ["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1-base"])
cache_interval = st.sidebar.slider("Cache Interval", min_value=1, max_value=10, value=3, help="Interval for updating the cache. Higher means faster but potentially lower quality.")
cache_branch_id = st.sidebar.slider("Cache Branch ID", min_value=0, max_value=2, value=0, help="Specifies the selected skip branch. 0 is the shallowest.")

# --- Model Loading ---
@st.cache_resource
def load_model(model_id):
    if torch.cuda.is_available():
        device = "cuda:0"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
        
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=dtype
    ).to(device)
    return pipe

def set_seed(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

with st.spinner(f"Loading Model {model_id}... (This may take a minute)"):
    try:
        pipe = load_model(model_id)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- Generation Logic ---
if st.button("Generate Images", type="primary"):
    col1, col2 = st.columns(2)
    
    # 1. Run Baseline
    with col1:
        st.subheader("Baseline (Original)")
        with st.spinner("Generating with original pipeline..."):
            set_seed(seed)
            start_time = time.time()
            baseline_image = pipe(prompt).images[0]
            baseline_time = time.time() - start_time
            
        st.image(baseline_image, use_container_width=True)
        st.success(f"⏱️ Time taken: {baseline_time:.2f} seconds")

    # 2. Run DeepCache
    with col2:
        st.subheader(f"DeepCache (Interval={cache_interval}, Branch={cache_branch_id})")
        with st.spinner("Generating with DeepCache pipeline..."):
            helper = DeepCacheSDHelper(pipe=pipe)
            helper.set_params(
                cache_interval=cache_interval,
                cache_branch_id=cache_branch_id,
            )
            helper.enable()
            
            set_seed(seed)
            start_time = time.time()
            deepcache_image = pipe(prompt).images[0]
            deepcache_time = time.time() - start_time
            
            helper.disable()

        st.image(deepcache_image, use_container_width=True)
        st.success(f"⏱️ Time taken: {deepcache_time:.2f} seconds")

    # --- Summary Statistics ---
    st.markdown("---")
    st.header("Results Summary")
    speedup = baseline_time / deepcache_time if deepcache_time > 0 else 0
    st.metric(label="Speedup Ratio", value=f"{speedup:.2f}x")

    # --- Difference Map ---
    st.markdown("---")
    st.subheader("Visual Difference Map (Heatmap)")
    st.markdown("This heatmap highlights the pixel-wise differences between the Original and DeepCache outputs. The differences are amplified **5x** and displayed in **red** for visibility. Since DeepCache is 'almost lossless', the heatmap should ideally be mostly dark.")
    
    heatmap_img = generate_heatmap(baseline_image, deepcache_image)
    
    # Center the heatmap in the middle
    col_hm1, col_hm2, col_hm3 = st.columns([1, 2, 1])
    with col_hm2:
        st.image(heatmap_img, use_container_width=True, caption="Red = Pixel Difference (Amplified 5x)")
