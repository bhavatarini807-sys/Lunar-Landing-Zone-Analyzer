
import os
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="AI Lunar Landing Zone Safety Analyzer",
    page_icon="🌕",
    layout="wide"
)

st.title("🌕 AI Lunar Landing Zone Safety Analyzer")
st.write("Upload a lunar surface image to analyze the landing zone.")

# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------
MODEL_PATH = "lunar_model.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

if model is None:
    st.error(
        "Model file not found! Place "
        "lunar_landing_safety_model.keras "
        "in the same folder as app.py."
    )
    st.stop()

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Lunar Image",
    type=[
        "jpg", "jpeg", "png",
        "webp", "bmp", "tif", "tiff"
    ]
)
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Resize for model prediction
    resized_img = cv2.resize(img_array, (224, 224))
    img_input = resized_img.astype("float32") / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # -------------------------------
    # MODEL PREDICTION
    # -------------------------------
    with st.spinner("Analyzing lunar surface..."):
        prediction = model.predict(img_input, verbose=0)
        score = float(prediction[0][0])

    if score >= 0.5:
        landing_zone = "HAZARDOUS"
        recommendation = "NOT RECOMMENDED FOR LANDING"
    else:
        landing_zone = "SAFE"
        recommendation = (
            "POTENTIALLY SUITABLE - "
            "FURTHER CHECKS NEEDED"
        )

    st.subheader("Analysis Results")

    # -------------------------------
    # RESULT SUMMARY
    # -------------------------------
    if landing_zone == "SAFE":
        st.success("Landing Zone: SAFE")
    else:
        st.error("Landing Zone: HAZARDOUS")

    st.write(f"**Prediction Score:** {score:.4f}")
    st.write(f"**Recommendation:** {recommendation}")

    # -------------------------------
    # TERRAIN HEATMAP
    # Same OpenCV method as Colab
    # -------------------------------
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (224, 224))

    hazard_map = 255 - gray
    hazard_map = cv2.GaussianBlur(
        hazard_map, (21, 21), 0
    )

    # Normalize heatmap for visualization
    heatmap = cv2.normalize(
        hazard_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype("uint8")

    heatmap_color = cv2.applyColorMap(
        heatmap, cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(
        heatmap_color, cv2.COLOR_BGR2RGB
    )

    # Resize original image for overlay
    original_small = cv2.resize(
        img_array, (224, 224)
    )

    overlay = cv2.addWeighted(
        original_small, 0.55,
        heatmap_color, 0.45,
        0
    )

    # -------------------------------
    # DISPLAY SMALLER IMAGES
    # -------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Input Image")
        st.image(
            img_array,
            caption="Original Lunar Image",
            width=220
        )

    with col2:
        st.subheader("Terrain Heatmap")
        st.image(
            heatmap_color,
            caption="Heatmap Visualization",
            width=220
        )

    with col3:
        st.subheader("Overlay Output")
        st.image(
            overlay,
            caption="Image + Heatmap",
            width=220
        )

    # -------------------------------
    # OPTIONAL HEATMAP LEGEND
    # -------------------------------
    st.caption(
        "Heatmap colors represent grayscale intensity "
        "after inversion and smoothing; they are not "
        "a direct measurement of landing risk."
    )

    st.info(
        "This is an AI-assisted preliminary result. "
        "A SAFE prediction does not guarantee a safe landing."
    )


