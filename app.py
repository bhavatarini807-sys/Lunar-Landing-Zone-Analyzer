import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(
    page_title="AI Lunar Landing Zone Analyzer",
    page_icon="🌕",
    layout="centered"
)

st.title("🌕 AI Lunar Landing Zone Safety Analyzer")
st.write("Upload a lunar surface image to analyze the landing zone.")

MODEL_PATH = "lunar_landing_safety_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found!")
    st.info("Place lunar_landing_safety_model.keras in this same folder.")
else:
    try:
        model = load_model()
        uploaded_file = st.file_uploader(
            "Upload Lunar Surface Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Lunar Image",
                     use_container_width=True)

            if st.button("Analyze Landing Zone"):
                resized_image = image.resize((224, 224))
                image_array = np.array(resized_image) / 255.0
                image_array = np.expand_dims(image_array, axis=0)

                prediction = float(
                    np.array(model.predict(image_array)).reshape(-1)[0]
                )

                st.subheader("Analysis Result")
                st.write(f"Prediction Score: {prediction:.4f}")

                if prediction >= 0.5:
                    st.success("Landing Zone: Safe")
                    st.write("Recommendation: Candidate zone classified as safe.")
                else:
                    st.error("Landing Zone: Hazardous")
                    st.write("Recommendation: Avoid this candidate zone.")

                st.caption(
                    "Research prototype only. Confirm class mapping and "
                    "preprocessing against the model's training code. "
                    "This result is not suitable for real lunar landing decisions."
                )

    except Exception as e:
        st.error("Could not load the model or analyze the image.")
        st.write(str(e))
