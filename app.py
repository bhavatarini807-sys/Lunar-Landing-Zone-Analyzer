
import os
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# ==========================================
# 1. PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="AI Lunar Landing Zone Analyzer",
    page_icon="🌕",
    layout="wide"
)

st.title("🌕 AI Lunar Landing Zone Safety Analyzer")
st.write(
    "Upload a lunar surface image to analyze the landing zone."
)

IMG_SIZE = (224, 224)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "lunar_model.keras"
)


# ==========================================
# 2. LOAD CNN MODEL
# ==========================================

@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found! Place "
            "'lunar_landing_safety_model.keras' "
            "in the same folder as app.py."
        )

    return tf.keras.models.load_model(MODEL_PATH)


# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================

def prepare_image(image):
    img = np.array(image.convert("RGB"))

    img_resized = cv2.resize(img, IMG_SIZE)

    img_input = img_resized.astype("float32") / 255.0

    img_input = np.expand_dims(img_input, axis=0)

    return img, img_input


# ==========================================
# 4. TERRAIN HEATMAP
# Same logic as Google Colab
# ==========================================

def create_terrain_heatmap(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(gray, IMG_SIZE)

    # Highlight darker regions
    hazard_map = 255 - gray

    # Smooth the visualization
    hazard_map = cv2.GaussianBlur(
        hazard_map,
        (21, 21),
        0
    )

    return hazard_map


# ==========================================
# 5. MAIN APPLICATION
# ==========================================

try:
    model = load_model()

    st.success("AI model loaded successfully!")

    uploaded_file = st.file_uploader(
        "Upload a lunar surface image",
        type=["jpg", "jpeg", "png", "jfif"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        img, img_input = prepare_image(image)

        st.subheader("Uploaded Lunar Image")

        st.image(
            img,
            caption=uploaded_file.name,
            use_container_width=True
        )

        if st.button(
            "Analyze Landing Zone",
            type="primary"
        ):

            # ==========================================
            # CNN PREDICTION
            # ==========================================

            with st.spinner("Analyzing lunar terrain..."):

                prediction = model.predict(
                    img_input,
                    verbose=0
                )

                score = float(
                    np.asarray(prediction).reshape(-1)[0]
                )

            # Same threshold logic as Colab
            if score >= 0.5:
                result = "HAZARDOUS"
                recommendation = (
                    "NOT RECOMMENDED FOR LANDING"
                )
            else:
                result = "SAFE"
                recommendation = (
                    "POTENTIALLY SUITABLE - "
                    "FURTHER CHECKS NEEDED"
                )

            # ==========================================
            # DISPLAY PREDICTION
            # ==========================================

            st.divider()
            st.subheader("🤖 AI Prediction Result")

            st.write(
                f"**Image Name:** {uploaded_file.name}"
            )

            st.metric(
                "Prediction Score",
                f"{score:.4f}"
            )

            if result == "HAZARDOUS":
                st.error("🔴 Landing Zone: HAZARDOUS")
            else:
                st.success("🟢 Landing Zone: SAFE")

            st.info(
                f"Recommendation: {recommendation}"
            )

            # ==========================================
            # DISPLAY TERRAIN HEATMAP
            # ==========================================

            st.divider()
            st.subheader(
                "🔥 Terrain Visualization Heatmap"
            )

            hazard_map = create_terrain_heatmap(img)

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    img,
                    caption="Uploaded Lunar Image",
                    use_container_width=True
                )

            with col2:
                fig, ax = plt.subplots(figsize=(6, 6))

                heat = ax.imshow(
                    hazard_map,
                    cmap="jet"
                )

                ax.set_title(
                    "Terrain Visualization Heatmap"
                )

                ax.axis("off")

                fig.colorbar(
                    heat,
                    ax=ax,
                    label="Relative Intensity"
                )

                fig.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

            # ==========================================
            # HEATMAP OVERLAY
            # ==========================================

            st.subheader(
                "🗺️ Heatmap Overlay on Lunar Image"
            )

            fig, ax = plt.subplots(figsize=(8, 6))

            ax.imshow(img)

            ax.imshow(
                hazard_map,
                cmap="jet",
                alpha=0.45
            )

            ax.set_title(
                "Lunar Terrain Heatmap Overlay"
            )

            ax.axis("off")

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

            st.caption(
                "The heatmap highlights darker image regions "
                "using grayscale intensity. It is a terrain "
                "visualization, not a verified AI hazard map."
            )

            st.divider()

            st.write(
                "⚠️ Research demonstration only. "
                "The CNN prediction depends on its training data "
                "and correct class mapping."
            )

except Exception as e:

    st.error("Model loading or prediction failed.")

    st.code(
        f"{type(e).__name__}: {e}"
    )
