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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_trained_model, predict_single

WEIGHTS_PATH = PROJECT_ROOT / "models" / "weights" / "best_model.pth"
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@st.cache_resource
def get_model():
    return load_trained_model(WEIGHTS_PATH, DEVICE)


def preprocess(image: Image.Image):
    preview = np.asarray(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    transform = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
    return transform(image=preview)["image"], preview


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    overlay[mask > 0] = [255, 72, 92]
    return (0.62 * image + 0.38 * overlay).astype(np.uint8)


def png_bytes(image: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(image).save(output, format="PNG")
    return output.getvalue()


def main():
    st.set_page_config(page_title="CerebraScan", page_icon="🧠", layout="wide")
    st.markdown(
        """<style>
        .block-container {max-width: 1400px; padding-top: 2rem;}
        .hero {padding: 1.4rem 1.7rem; border-radius: 18px;
               background: linear-gradient(120deg,#101b37,#173e63);
               color: white; margin-bottom: 1.3rem;}
        .hero h1 {margin: 0; font-size: 2.5rem;}
        .hero p {color: #c7d9ef; margin: .45rem 0 0;}
        [data-testid="stMetric"] {background: #f5f8fc; border-radius: 12px; padding: 12px;}
        </style>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero"><h1>🧠 CerebraScan</h1>'
        "<p>AI-assisted brain MRI segmentation for faster visual review.</p></div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Analysis settings")
        threshold = st.slider(
            "Segmentation threshold", 0.1, 0.9, 0.5, 0.05,
            help="Higher values produce a more conservative tumor mask.",
        )
        st.caption("Model: ResU-Net · ResNet34 encoder")
        st.caption(f"Runtime: {DEVICE.upper()} · Input: {IMG_SIZE} × {IMG_SIZE}")
        st.divider()
        st.info("For research support only. This tool is not a medical diagnosis.")

    uploaded_file = st.file_uploader(
        "Upload an MRI slice",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="The image is resized to 256 × 256 for inference.",
    )
    if uploaded_file is None:
        st.markdown("### Start an analysis")
        st.write("Upload a JPG, PNG, or TIFF MRI slice to generate a segmentation.")
        return

    try:
        image = Image.open(uploaded_file)
        tensor, input_image = preprocess(image)
        with st.spinner("Running segmentation..."):
            model = get_model()
            mask, probabilities = predict_single(tensor, model, DEVICE, threshold)
    except (OSError, RuntimeError, ValueError, FileNotFoundError, UnidentifiedImageError) as error:
        st.error(f"Analysis could not be completed: {error}")
        st.stop()

    tumor_pixels = int(mask.sum())
    tumor_pct = tumor_pixels / mask.size * 100
    confidence = float(probabilities[mask > 0].mean()) if tumor_pixels else float(probabilities.mean())
    overlay = make_overlay(input_image, mask)
    mask_image = (mask * 255).astype(np.uint8)

    st.success("Analysis complete")
    metrics = st.columns(3)
    metrics[0].metric("Tumor area", f"{tumor_pct:.2f}%")
    metrics[1].metric("Tumor pixels", f"{tumor_pixels:,}")
    metrics[2].metric("Mean confidence", f"{confidence:.1%}")

    st.markdown("### Visual review")
    col1, col2, col3 = st.columns(3)
    col1.image(input_image, caption="Input MRI", use_container_width=True)
    col2.image(mask_image, caption="Predicted tumor mask", use_container_width=True)
    col3.image(overlay, caption="MRI + segmentation overlay", use_container_width=True)

    download_col1, download_col2 = st.columns(2)
    download_col1.download_button(
        "Download mask", png_bytes(mask_image), "cerebrascan-mask.png", "image/png"
    )
    download_col2.download_button(
        "Download overlay", png_bytes(overlay), "cerebrascan-overlay.png", "image/png"
    )


if __name__ == "__main__":
    main()
