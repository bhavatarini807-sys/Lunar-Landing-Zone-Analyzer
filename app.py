
import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="AI Lunar Landing Zone Analyzer",
    page_icon="🌕",
    layout="centered"
)

st.title("🌕 AI Lunar Landing Zone Safety Analyzer")
st.write(
    "Upload a lunar surface image to analyze the landing zone."
)

# Model file must be in the same folder as app.py
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "lunar_model.keras"
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found. Place "
            "lunar_landing_safety_model.keras "
            "in the same folder as app.py."
        )

    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    st.success("AI model loaded successfully!")

    uploaded_file = st.file_uploader(
        "Upload a lunar surface image",
        type=["jpg", "jpeg", "png", "jfif"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Lunar Surface Image",
            use_container_width=True
        )

        if st.button("Analyze Landing Zone"):
            with st.spinner("Analyzing lunar terrain..."):
                resized_image = image.resize((224, 224))
                image_array = np.array(
                    resized_image, dtype=np.float32
                ) / 255.0

                image_array = np.expand_dims(
                    image_array, axis=0
                )

                prediction = model.predict(
                    image_array, verbose=0
                )

                score = float(
                    np.array(prediction).reshape(-1)[0]
                )

                st.write(f"Prediction Score: {score:.4f}")

                # Assumes a sigmoid output where
                # 1 = Safe and 0 = Hazardous.
                if score >= 0.5:
                    st.success("🟢 Landing Zone: SAFE")
                    st.write(
                        "Recommendation: Candidate safe zone. "
                        "Further terrain verification is required."
                    )
                else:
                    st.error("🔴 Landing Zone: HAZARDOUS")
                    st.write(
                        "Recommendation: Avoid this zone "
                        "and inspect another location."
                    )

                st.caption(
                    "Research demonstration only. "
                    "Confirm class mapping and preprocessing "
                    "match the model's training code."
                )

except Exception as e:
    st.error("Model loading or prediction failed.")
    st.code(str(e))
