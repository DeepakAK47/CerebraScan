"""CerebraScan Streamlit application."""

from io import BytesIO
from pathlib import Path
import sys

import albumentations as A
import numpy as np
import streamlit as st
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, UnidentifiedImageError

# --- PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_trained_model, predict_single

# --- CONFIG ---
WEIGHTS_PATH = PROJECT_ROOT / "models" / "weights" / "best_model.pth"
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@st.cache_resource
def get_model():
    """Load the model once and cache it."""
    return load_trained_model(WEIGHTS_PATH, DEVICE)


def preprocess(image: Image.Image):
    """Convert uploaded image to tensor + numpy preview."""
    preview = np.asarray(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    transform = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
    return transform(image=preview)["image"], preview


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay the predicted mask on the MRI image."""
    overlay = image.copy()
    # Bright red color for tumor
    overlay[mask > 0] = [255, 72, 92]
    return (0.62 * image + 0.38 * overlay).astype(np.uint8)


def png_bytes(image: np.ndarray) -> bytes:
    """Convert numpy array to PNG byte stream for downloading."""
    output = BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def main():
    # Page Config
    st.set_page_config(
        page_title="CerebraScan | Brain Tumor Segmentation",
        page_icon="🧠",
        layout="wide",
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .block-container { max-width: 1300px; padding-top: 2rem; }
        .hero {
            padding: 1.5rem 2rem; border-radius: 16px;
            background: linear-gradient(120deg, #0E1A35, #1B4375);
            color: white; margin-bottom: 2rem; text-align: center;
        }
        .hero h1 { margin: 0; font-size: 2.8rem; font-weight: 700; }
        .hero p { color: #B0C4DE; margin: .8rem 0 0; font-size: 1.1rem; }
        [data-testid="stMetricValue"] { color: #1B4375; }
        .stProgress > div > div > div > div { background-color: #1B4375; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Hero Banner
    st.markdown(
        '<div class="hero">'
        '<h1>🧠 CerebraScan</h1>'
        "<p>AI-assisted brain MRI segmentation for faster visual review.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Analysis Settings")
        threshold = st.slider(
            "Segmentation Threshold",
            min_value=0.1, max_value=0.9, value=0.5, step=0.05,
            help="Higher values produce a more conservative tumor mask.",
        )
        
        st.markdown("---")
        st.markdown("**Model Info**")
        st.caption("Architecture: ResU-Net")
        st.caption("Encoder: ResNet34 (ImageNet)")
        st.caption(f"Runtime: `{DEVICE.upper()}`")
        st.caption(f"Input Size: `{IMG_SIZE} × {IMG_SIZE}`")
        
        st.markdown("---")
        st.info("⚠️ For research and academic purposes only. This tool is not a medical diagnosis.")

    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload an MRI slice",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="The image is resized to 256 × 256 for inference.",
    )

    if uploaded_file is None:
        st.markdown("### 🚀 Start an Analysis")
        st.write("Upload a JPG, PNG, or TIFF MRI slice to generate a tumor segmentation.")
        return

    # Inference
    try:
        image = Image.open(uploaded_file)
        tensor, input_image = preprocess(image)
        
        with st.spinner("Running segmentation..."):
            model = get_model()
            mask, probabilities = predict_single(tensor, model, DEVICE, threshold)
            
    except (OSError, RuntimeError, ValueError, FileNotFoundError, UnidentifiedImageError) as error:
        st.error(f"❌ Analysis could not be completed: {error}")
        st.stop()

    # Results Calculation
    tumor_pixels = int(mask.sum())
    tumor_pct = tumor_pixels / mask.size * 100
    confidence = float(probabilities[mask > 0].mean()) if tumor_pixels else float(probabilities.mean())
    overlay = make_overlay(input_image, mask)
    mask_image = (mask * 255).astype(np.uint8)

    # Result Banner
    if tumor_pixels > 0:
        st.success("✅ Analysis Complete: Tumor region detected.")
    else:
        st.success("✅ Analysis Complete: No tumor detected above threshold.")

    # Metrics
    metrics = st.columns(3)
    metrics[0].metric("Tumor Area", f"{tumor_pct:.2f}%")
    metrics[1].metric("Tumor Pixels", f"{tumor_pixels:,}")
    metrics[2].metric("Mean Confidence", f"{confidence:.1%}")

    st.markdown("---")

    # Visual Review
    st.markdown("### 🖼️ Visual Review")
    col1, col2, col3 = st.columns(3)
    col1.image(input_image, caption="Input MRI", use_column_width=True)
    col2.image(mask_image, caption="Predicted Mask", use_column_width=True)
    col3.image(overlay, caption="Tumor Overlay", use_column_width=True)

    # Downloads
    st.markdown("### 💾 Download Results")
    download_col1, download_col2 = st.columns(2)
    download_col1.download_button(
        "📥 Download Mask (PNG)", 
        png_bytes(mask_image), 
        "cerebrascan-mask.png", 
        "image/png",
        use_container_width=True
    )
    download_col2.download_button(
        "📥 Download Overlay (PNG)", 
        png_bytes(overlay), 
        "cerebrascan-overlay.png", 
        "image/png",
        use_container_width=True
    )


if __name__ == "__main__":
    main()