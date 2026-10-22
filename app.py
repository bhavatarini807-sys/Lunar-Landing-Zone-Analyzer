
import os
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title="AI Lunar Landing Zone Analyzer",
    page_icon="🌕",
    layout="wide"
)

st.title("🌕 AI Lunar Landing Zone Safety Analyzer")
st.write(
    "Upload a lunar surface image to analyze the landing zone."
)

# Keep this model file in the same folder as app.py
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "lunar_model.keras"
)

IMG_SIZE = (224, 224)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found: "
            "lunar_landing_safety_model.keras"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    # Build the model before accessing its layers
    dummy_input = tf.zeros((1, 224, 224, 3))
    _ = model(dummy_input, training=False)

    return model


def prepare_image(image):
    resized = image.resize(IMG_SIZE)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def find_last_conv_layer(model):
    # Search from the last layer backwards
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

        if isinstance(layer, tf.keras.Model):
            found = find_last_conv_layer(layer)
            if found is not None:
                return found

    return None


def make_gradcam_heatmap(img_array, model):
    # Ensure the main model has been called
    _ = model(img_array, training=False)

    conv_layer = find_last_conv_layer(model)

    if conv_layer is None:
        raise ValueError(
            "No convolutional layer found for Grad-CAM."
        )

    # Locate the model containing the selected convolutional layer
    nested_model = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                _ = layer(img_array, training=False)
                if conv_layer in layer.layers:
                    nested_model = layer
                    break
            except Exception:
                continue

    if nested_model is not None:
        # Nested Sequential/Functional model
        inner_grad_model = tf.keras.Model(
            inputs=nested_model.inputs,
            outputs=[
                conv_layer.output,
                nested_model.output
            ]
        )

        with tf.GradientTape() as tape:
            conv_outputs, nested_predictions = inner_grad_model(
                img_array, training=False
            )

            # Pass the nested output through remaining layers
            x = nested_predictions

            start_index = model.layers.index(nested_model) + 1

            for layer in model.layers[start_index:]:
                x = layer(x, training=False)

            predictions = x
            score = predictions[:, 0]

        gradients = tape.gradient(score, conv_outputs)

    else:
        # Convolutional layer belongs directly to the main model
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
            "Gradients unavailable. Model architecture "
            "may require a custom Grad-CAM implementation."
        )

    # Calculate Grad-CAM weights and heatmap
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


def create_overlay(image, heatmap):
    original = np.asarray(image.resize(IMG_SIZE))

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        IMG_SIZE
    ).numpy().squeeze()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(original)
    ax.imshow(
        heatmap_resized,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )
    ax.axis("off")
    fig.tight_layout(pad=0)

    return fig


# Load model and display application
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

            st.subheader("Prediction Result")
            st.metric("Prediction Score", f"{score:.4f}")

            # Assumption: sigmoid output 1 = Safe, 0 = Hazardous
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

                st.pyplot(
                    create_overlay(image, heatmap),
                    use_container_width=True
                )

                plt.close("all")

                st.caption(
                    "Red/yellow regions show areas that influence "
                    "the model's prediction. This is an explanation "
                    "heatmap, not verified rock or crater detection."
                )

            except Exception as heatmap_error:
                st.warning(
                    "Prediction completed, but the heatmap "
                    "could not be generated."
                )
                st.code(str(heatmap_error))

            st.caption(
                "Research demonstration only. Confirm the output "
                "class mapping and image preprocessing against "
                "the original model training code."
            )

except Exception as e:
    st.error("Model loading failed.")
    st.code(str(e))
