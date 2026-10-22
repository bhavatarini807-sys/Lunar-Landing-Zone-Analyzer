
import os
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# ---------------- PAGE SETTINGS ----------------
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


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found. Keep "
            "'lunar_landing_safety_model.keras' "
            "in the same folder as app.py."
        )

    return tf.keras.models.load_model(MODEL_PATH)


# ---------------- PREPARE IMAGE ----------------
def prepare_image(image):
    resized = image.resize(IMG_SIZE)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


# ---------------- FIND LAST CONVOLUTION LAYER ----------------
def find_last_conv_layer(container):
    for layer in reversed(container.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

        if isinstance(layer, tf.keras.Model):
            result = find_last_conv_layer(layer)
            if result is not None:
                return result

    return None


# ---------------- GRAD-CAM HEATMAP ----------------
def make_gradcam_heatmap(img_array, model):
    conv_layer = find_last_conv_layer(model)

    if conv_layer is None:
        raise ValueError(
            "No Conv2D layer found in the saved model."
        )

    # Rebuild a differentiable forward pass without using
    # nested layers' .output property.
    input_tensor = tf.keras.Input(shape=(224, 224, 3))

    def forward(container, x, target):
        activation = None

        for layer in container.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            if isinstance(layer, tf.keras.Model):
                x, nested_activation = forward(
                    layer, x, target
                )
                if nested_activation is not None:
                    activation = nested_activation
            else:
                x = layer(x)

                if layer is target:
                    activation = x

        return x, activation

    predictions, activations = forward(
        model, input_tensor, conv_layer
    )

    if activations is None:
        raise ValueError(
            "Could not connect the selected convolutional layer."
        )

    grad_model = tf.keras.Model(
        inputs=input_tensor,
        outputs=[activations, predictions]
    )

    with tf.GradientTape() as tape:
        conv_outputs, model_predictions = grad_model(
            img_array, training=False
        )

        # Assumes a single sigmoid output.
        score = model_predictions[:, 0]

    gradients = tape.gradient(score, conv_outputs)

    if gradients is None:
        raise ValueError(
            "Gradients are unavailable for this model architecture."
        )

    # Average gradients across the feature-map dimensions.
    weights = tf.reduce_mean(gradients, axis=(1, 2))

    heatmap = tf.reduce_sum(
        conv_outputs[0] * weights[0],
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


# ---------------- CREATE HEATMAP OVERLAY ----------------
def create_overlay(image, heatmap):
    original = np.asarray(image.resize(IMG_SIZE))

    resized_heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        IMG_SIZE
    ).numpy().squeeze()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(original)
    ax.imshow(
        resized_heatmap,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )
    ax.axis("off")
    fig.tight_layout(pad=0)

    return fig


# ---------------- MAIN APPLICATION ----------------
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

        if st.button("Analyze Landing Zone", type="primary"):
            img_array = prepare_image(image)

            with st.spinner("Analyzing lunar terrain..."):
                prediction = model.predict(
                    img_array, verbose=0
                )

                score = float(
                    np.asarray(prediction).reshape(-1)[0]
                )

            # Prediction result
            st.subheader("Prediction Result")
            st.metric("Prediction Score", f"{score:.4f}")

            # Assumption: model output 1 = Safe, 0 = Hazardous.
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

            # Grad-CAM heatmap
            st.divider()
            st.subheader("🔥 Grad-CAM Heatmap Visualization")

            try:
                heatmap = make_gradcam_heatmap(
                    img_array, model
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        image.resize(IMG_SIZE),
                        caption="Original Image",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        heatmap,
                        caption="Grad-CAM Heatmap",
                        clamp=True,
                        use_container_width=True
                    )

                overlay_fig = create_overlay(image, heatmap)
                st.pyplot(overlay_fig, use_container_width=True)
                plt.close(overlay_fig)

                st.caption(
                    "The heatmap highlights image regions that "
                    "influence the model's prediction. It does "
                    "not independently confirm rocks or craters."
                )

            except Exception as heatmap_error:
                st.warning(
                    "Prediction completed, but heatmap generation "
                    "failed for this model."
                )
                st.code(
                    f"{type(heatmap_error).__name__}: "
                    f"{heatmap_error}"
                )

            st.caption(
                "Research demo only. Verify class mapping and "
                "preprocessing against the original training code."
            )

except Exception as error:
    st.error("Model loading failed.")
    st.code(f"{type(error).__name__}: {error}")
