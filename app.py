import streamlit as st
from PIL import Image
from io import BytesIO
from torchvision import transforms
import os
from models.clip_encoder import CLIPFeatureExtractor
from src.inference import generate_caption_openrouter

from dotenv import load_dotenv
load_dotenv()

# ✅ Set your OpenRouter API key here (DO NOT expose in UI)
API_KEY = os.getenv("OPEN_ROUTER_KEY")

st.set_page_config(page_title="🧠🖼 Medical & Normal Image Captioning", layout="centered")
st.title("🧠🖼 Medical & Normal Image Captioning")

# Sidebar - mode select
mode = st.sidebar.radio("Choose captioning mode", ["🏥 Medical", "🏞 Normal"])
uploaded = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        image = Image.open(BytesIO(uploaded.read())).convert("RGB")
    except Exception as e:
        st.error(f"❌ Error loading image: {e}")
        st.stop()

    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # Extract image features
    extractor = CLIPFeatureExtractor()
    features = extractor.get_features(image)

    # Prepare the prompt for the LLM
    prompt = (
    f"You are a professional image captioning assistant. "
    f"Look at the image features provided and generate a clear, descriptive, and context-appropriate caption. "
    f"This image belongs to the {'medical imaging domain (e.g., chest X-ray, radiology)' if mode == '🏥 Medical' else 'natural image domain (e.g., objects, scenes)'}.\n\n"
    f"Generate one meaningful caption:"
    )

    # Generate caption using OpenRouter-powered LLM
    caption = generate_caption_openrouter(prompt, API_KEY)

    st.markdown("### 📝 Generated Caption")
    st.success(caption)

else:
    st.info("Please upload an image to generate a caption.")