
import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import gdown

st.set_page_config(page_title="AI Lunar Landing Zone Analyzer", page_icon="🌕")

st.title("🌕 AI Lunar Landing Zone Safety Analyzer")
st.write("Upload a lunar surface image to analyze the landing zone.")

MODEL_PATH = "lunar_landing_safety_model.keras"

# Paste your Google Drive FILE ID below
DRIVE_FILE_ID = "1AWV2dwrafu2pfnDtoKzNPd_qe6SsxCL1"
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

if DRIVE_FILE_ID == "1AWV2dwrafu2pfnDtoKzNPd_qe6SsxCL1":
    st.warning("Add your Google Drive file ID in the code first.")
else:
    try:
        model = load_model()
        uploaded_file = st.file_uploader(
            "Upload lunar surface image",
            type=["jpg", "jpeg", "png", "jfif"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Lunar Image", use_container_width=True)

            if st.button("Analyze Landing Zone"):
                resized = image.resize((224, 224))
                array = np.array(resized, dtype=np.float32) / 255.0
                array = np.expand_dims(array, axis=0)

                score = float(np.array(model.predict(array)).reshape(-1)[0])
                st.write(f"Prediction Score: {score:.4f}")

                if score >= 0.5:
                    st.success("Landing Zone: Safe")
                else:
                    st.error("Landing Zone: Hazardous")

                st.caption("Research demo only. Verify class mapping and preprocessing against the model training code.")
    except Exception as e:
        st.error("Model loading or prediction failed.")
        st.code(str(e))
