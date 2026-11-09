
import os
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CNN Based Lunar Landing Classification",
    page_icon="🌕",
    layout="wide"
)

st.title(
    "🌕 CNN – BASED CLASSIFICATION OF SAFE AND HAZARDOUS LUNAR LANDING ZONES"
)

st.write(
    "Upload a lunar surface image to classify "
    "the landing zone as SAFE or HAZARDOUS."
)

# ==========================================
# MODEL CONFIGURATION
# ==========================================

MODEL_PATH = "lunar_landing_safety_model.keras"

IMG_SIZE = 224
THRESHOLD = 0.5

# IMPORTANT:
# hazardous = 0, safe = 1
# Sigmoid output represents SAFE probability

POSITIVE_CLASS = "safe"

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return tf.keras.models.load_model(
        MODEL_PATH
    )

try:

    model = load_model()

except Exception as e:

    st.error(
        f"Model loading error: {e}"
    )

    st.stop()

if model is None:

    st.error(
        f"Model file not found: {MODEL_PATH}"
    )

    st.info(
        "Place the trained .keras model "
        "in the same folder as app.py."
    )

    st.stop()

# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Lunar Surface Image",
    type=[
        "jpg", "jpeg", "png",
        "webp", "bmp", "tif", "tiff"
    ]
)

if uploaded_file is not None:

    try:

        # ======================================
        # IMAGE PREPROCESSING
        # ======================================

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        img_array = np.array(
            image
        )

        resized_img = cv2.resize(
            img_array,
            (IMG_SIZE, IMG_SIZE)
        )

        img_input = (
            resized_img.astype("float32")
            / 255.0
        )

        img_input = np.expand_dims(
            img_input,
            axis=0
        )

        # ======================================
        # CNN MODEL PREDICTION
        # ======================================

        with st.spinner(
            "Analyzing lunar landing zone..."
        ):

            prediction = model.predict(
                img_input,
                verbose=0
            )

            raw_score = float(
                prediction[0][0]
            )

        # ======================================
        # PROBABILITY CALCULATION
        # ======================================

        if POSITIVE_CLASS == "safe":

            safe_probability = raw_score

            hazard_probability = (
                1 - raw_score
            )

        else:

            hazard_probability = raw_score

            safe_probability = (
                1 - raw_score
            )

        # ======================================
        # CLASSIFICATION
        # ======================================

        if safe_probability >= THRESHOLD:

            landing_zone = "SAFE"

            recommendation = (
                "POTENTIALLY SUITABLE "
                "FOR LANDING"
            )

        else:

            landing_zone = "HAZARDOUS"

            recommendation = (
                "NOT RECOMMENDED "
                "FOR LANDING"
            )

        # ======================================
        # DISPLAY RESULTS
        # ======================================

        st.subheader(
            "CNN Classification Results"
        )

        if landing_zone == "SAFE":

            st.success(
                "🟢 LANDING ZONE: SAFE"
            )

        else:

            st.error(
                "🔴 LANDING ZONE: HAZARDOUS"
            )

        # ======================================
        # PROBABILITY METRICS
        # ======================================

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Safe Probability",
                f"{safe_probability * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Hazard Probability",
                f"{hazard_probability * 100:.2f}%"
            )

        st.write(
            f"**Raw Model Output:** "
            f"{raw_score:.4f}"
        )

        st.write(
            f"**Classification Threshold:** "
            f"{THRESHOLD}"
        )

        st.write(
            f"**Recommendation:** "
            f"{recommendation}"
        )

        # ======================================
        # TERRAIN HEATMAP
        # ======================================

        gray = cv2.cvtColor(
            img_array,
            cv2.COLOR_RGB2GRAY
        )

        gray = cv2.resize(
            gray,
            (IMG_SIZE, IMG_SIZE)
        )

        hazard_map = 255 - gray

        hazard_map = cv2.GaussianBlur(
            hazard_map,
            (21, 21),
            0
        )

        heatmap = cv2.normalize(
            hazard_map,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype("uint8")

        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB
        )

        # ======================================
        # OVERLAY IMAGE
        # ======================================

        original_small = cv2.resize(
            img_array,
            (IMG_SIZE, IMG_SIZE)
        )

        overlay = cv2.addWeighted(
            original_small,
            0.55,
            heatmap_color,
            0.45,
            0
        )

        # ======================================
        # DISPLAY IMAGES
        # ======================================

        st.subheader(
            "Lunar Terrain Visualization"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.image(
                img_array,
                caption="Original Lunar Image",
                width=220
            )

        with col2:

            st.image(
                heatmap_color,
                caption="Terrain Heatmap",
                width=220
            )

        with col3:

            st.image(
                overlay,
                caption="Image + Heatmap",
                width=220
            )

        # ======================================
        # DOWNLOAD HEATMAP
        # ======================================

        heatmap_bgr = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_RGB2BGR
        )

        success, encoded = cv2.imencode(
            ".png",
            heatmap_bgr
        )

        if success:

            st.download_button(
                label="Download Heatmap",
                data=encoded.tobytes(),
                file_name="lunar_heatmap.png",
                mime="image/png"
            )

        # ======================================
        # DISCLAIMER
        # ======================================

        st.caption(
            "Heatmap represents grayscale intensity "
            "after inversion and smoothing. "
            "It is not a direct measurement "
            "of landing risk."
        )

        st.warning(
            "This CNN-based classification is an "
            "experimental prediction and must not "
            "be used as the sole basis for an "
            "actual lunar landing decision."
        )

    except Exception as e:

        st.error(
            f"Image processing error: {e}"
        )
