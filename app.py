
import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Lunar Landing Zone Analyzer",
    page_icon="🌕",
    layout="wide"
)

st.title("🌕 AI Lunar Landing Zone Safety Analyzer")
st.write("Upload a lunar surface image to analyze the landing zone.")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "lunar_model.keras"
)

IMG_SIZE = (224, 224)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Place lunar_landing_safety_model.keras "
            "in the same folder as app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(image):
    resized = image.resize(IMG_SIZE)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        if isinstance(layer, tf.keras.Model):
            found = find_last_conv_layer(layer)
            if found is not None:
                return found
    return None


def make_gradcam_heatmap(img_array, model):
    conv_layer = find_last_conv_layer(model)

    if conv_layer is None:
        raise ValueError("No convolutional layer found.")

    # Build a model returning feature maps and final prediction.
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            img_array, training=False
        )
        score = predictions[:, 0]

    gradients = tape.gradient(score, conv_outputs)

    if gradients is None:
        raise ValueError(
            "Grad-CAM gradients unavailable for this model."
        )

    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    feature_maps = conv_outputs[0]

    heatmap = tf.reduce_sum(
        feature_maps * weights, axis=-1
    )
    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()


def make_overlay(image, heatmap):
    original = np.asarray(image.resize(IMG_SIZE))
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], IMG_SIZE
    ).numpy().squeeze()

    fig, ax = plt.subplots()
    ax.imshow(original)
    ax.imshow(heatmap_resized, cmap="jet", alpha=0.45)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


try:
    model = load_model()
    st.success("AI model loaded successfully!")

    uploaded_file = st.file_uploader(
        "Upload a lunar surface image",
        type=["jpg", "jpeg", "png", "jfif"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded Lunar Surface Image",
            use_container_width=True
        )

        if st.button("Analyze Landing Zone"):
            img_array = prepare_image(image)

            with st.spinner("Analyzing lunar terrain..."):
                prediction = model.predict(
                    img_array, verbose=0
                )
                score = float(
                    np.asarray(prediction).reshape(-1)[0]
                )

            st.subheader("Prediction Result")
            st.metric("Prediction Score", f"{score:.4f}")

            # Assumption: sigmoid output 1 = Safe, 0 = Hazardous.
            if score >= 0.5:
                st.success("🟢 Landing Zone: SAFE")
                st.write(
                    "Recommendation: Candidate safe zone. "
                    "Further terrain verification is required."
                )
            else:
                st.error("🔴 Landing Zone: HAZARDOUS")
                st.write(
                    "Recommendation: Avoid this zone and "
                    "inspect another location."
                )

            st.subheader("🔥 Grad-CAM Heatmap Visualization")

            try:
                heatmap = make_gradcam_heatmap(
                    img_array, model
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        image.resize(IMG_SIZE),
                        caption="Input Image",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        heatmap,
                        caption="AI Heatmap",
                        clamp=True,
                        use_container_width=True
                    )

                st.pyplot(
                    make_overlay(image, heatmap),
                    use_container_width=True
                )

                st.caption(
                    "Red/yellow regions indicate stronger influence "
                    "on the model's prediction. The heatmap is an "
                    "explanation visualization, not a verified "
                    "rock or crater detection map."
                )

            except Exception as heatmap_error:
                st.warning(
                    "Prediction completed, but the heatmap "
                    "could not be generated for this model."
                )
                st.code(str(heatmap_error))

            st.caption(
                "Research demonstration only. Confirm the model's "
                "class mapping and preprocessing against its "
                "training code."
            )

except Exception as e:
    st.error("Model loading or prediction failed.")
    st.code(str(e))
