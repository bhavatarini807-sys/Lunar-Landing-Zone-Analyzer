"""
CNN - BASED CLASSIFICATION OF SAFE AND HAZARDOUS LUNAR LANDING ZONES
--------------------------------------------------------------------
Streamlit application that classifies a lunar surface image as SAFE or
HAZARDOUS using a trained CNN (lunar_landing_safety_model.keras).

Run with:
    streamlit run app.py

Class mapping (IMPORTANT)
-------------------------
Keras `image_dataset_from_directory` and `flow_from_directory` assign class
indices in ALPHABETICAL folder order. With folders named "hazardous" and
"safe":

    hazardous -> 0
    safe      -> 1

For a single sigmoid output unit, the model output is P(class index 1),
i.e. P(SAFE). The prediction rule is therefore:

    score >= 0.5  -> class 1 -> SAFE
    score <  0.5  -> class 0 -> HAZARDOUS

If your folders were named differently (e.g. "safe" and "unsafe", where
safe=0), switch the mapping in the sidebar.
"""

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import matplotlib

matplotlib.use("Agg")
from matplotlib import cm  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
PROJECT_TITLE = "CNN – BASED CLASSIFICATION OF SAFE AND HAZARDOUS LUNAR LANDING ZONES"
MODEL_FILENAME = "lunar_landing_safety_model.keras"
THRESHOLD = 0.5
ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]
DEFAULT_SIZE = 224

MAPPING_OPTIONS = {
    "hazardous = 0, safe = 1  (alphabetical folders, default)": ["HAZARDOUS", "SAFE"],
    "safe = 0, hazardous = 1": ["SAFE", "HAZARDOUS"],
}

st.set_page_config(
    page_title=PROJECT_TITLE,
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 1.9rem; font-weight: 800; letter-spacing: 0.5px;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center; opacity: 0.75; margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1.2rem 1.5rem; border-radius: 14px; text-align: center;
        font-size: 1.8rem; font-weight: 800; color: white; margin-bottom: 0.8rem;
    }
    .safe-box { background: linear-gradient(135deg, #1b8a4b, #2ecc71); }
    .haz-box  { background: linear-gradient(135deg, #b02a2a, #e74c3c); }
    .rec-box {
        padding: 1rem 1.2rem; border-radius: 12px; border-left: 6px solid;
        margin-top: 0.5rem;
    }
    .rec-safe { background: rgba(46,204,113,0.12); border-color: #2ecc71; }
    .rec-haz  { background: rgba(231,76,60,0.12);  border-color: #e74c3c; }
    .rec-warn { background: rgba(241,196,15,0.14); border-color: #f1c40f; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def find_model_path():
    """Look for the model next to app.py, in the working dir, or in ./models."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / MODEL_FILENAME,
        Path.cwd() / MODEL_FILENAME,
        here / "models" / MODEL_FILENAME,
        Path.cwd() / "models" / MODEL_FILENAME,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


@st.cache_resource(show_spinner="Loading CNN model...")
def load_model(path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False)


def get_model_input_info(model):
    """Return (height, width, channels) expected by the model."""
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    h = shape[1] if len(shape) > 1 and shape[1] else DEFAULT_SIZE
    w = shape[2] if len(shape) > 2 and shape[2] else DEFAULT_SIZE
    c = shape[3] if len(shape) > 3 and shape[3] else 3
    return int(h), int(w), int(c)


def model_has_internal_scaling(model):
    """True if the model already contains a Rescaling/Normalization layer."""

    def scan(layers):
        for layer in layers:
            name = layer.__class__.__name__
            if name in ("Rescaling", "Normalization"):
                return True
            if hasattr(layer, "layers") and scan(layer.layers):
                return True
        return False

    try:
        return scan(model.layers)
    except Exception:
        return False


# --------------------------------------------------------------------------
# Image handling
# --------------------------------------------------------------------------
def load_image(uploaded_file) -> Image.Image:
    """Open any supported upload and return a clean 8-bit RGB PIL image."""
    data = uploaded_file.getvalue()
    img = Image.open(io.BytesIO(data))
    img.load()

    # Multi-frame TIFF / animated WEBP: use the first frame
    try:
        img.seek(0)
    except Exception:
        pass

    img = ImageOps.exif_transpose(img)

    # Handle 16-bit / float TIFF modes by scaling to 8-bit
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.array(img).astype(np.float64)
        lo, hi = arr.min(), arr.max()
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        img = Image.fromarray(arr.astype(np.uint8))

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, rgba)

    return img.convert("RGB")


def preprocess(img_rgb: Image.Image, height, width, channels, scale_mode, model):
    """Resize and normalise exactly as during training."""
    img = img_rgb.convert("L") if channels == 1 else img_rgb
    img = img.resize((width, height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]

    if scale_mode == "Auto-detect":
        raw = model_has_internal_scaling(model)
    elif scale_mode == "Raw pixels (0–255)":
        raw = True
    else:
        raw = False

    if not raw:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def predict(model, x, class_names):
    """
    Returns dict with raw score, safe/hazardous probabilities and label.
    Sigmoid output = P(class index 1). Threshold 0.5 decides the class.
    """
    out = np.asarray(model.predict(x, verbose=0)).reshape(-1)

    if out.size == 1:
        score = float(out[0])
        p_class1 = score
        p_class0 = 1.0 - score
    else:  # defensive: 2-unit softmax model
        score = float(out[1])
        p_class1 = float(out[1])
        p_class0 = float(out[0])

    probs = {class_names[0]: p_class0, class_names[1]: p_class1}
    pred_idx = 1 if score >= THRESHOLD else 0
    return {
        "score": score,
        "pred_idx": pred_idx,
        "label": class_names[pred_idx],
        "p_safe": probs["SAFE"],
        "p_hazardous": probs["HAZARDOUS"],
    }


# --------------------------------------------------------------------------
# Heatmaps
# --------------------------------------------------------------------------
def grad_cam(model, x, pred_idx):
    """Grad-CAM on the last 4-D (feature map) layer. Returns 2-D map in [0,1]."""
    import tensorflow as tf

    last_layer = None
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                last_layer = layer
                break
        except Exception:
            continue
    if last_layer is None:
        return None

    grad_model = tf.keras.Model(model.inputs, [last_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(tf.convert_to_tensor(x))
        preds = tf.reshape(preds, (tf.shape(preds)[0], -1))
        score = preds[:, 0] if preds.shape[-1] == 1 else preds[:, 1]
        target = score if pred_idx == 1 else 1.0 - score

    grads = tape.gradient(target, conv_out)
    if grads is None:
        return None
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    if cam.max() <= 0:
        return None
    return cam / cam.max()


def terrain_roughness_map(img_rgb: Image.Image):
    """
    Fallback terrain map: combines local slope (gradient magnitude) and
    local brightness variation, which highlight craters, ridges and boulders.
    """
    gray = img_rgb.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    gy, gx = np.gradient(arr)
    mag = np.sqrt(gx**2 + gy**2)

    blur = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=7)), dtype=np.float32) / 255.0
    local_var = np.abs(arr - blur)

    rough = 0.6 * mag / (mag.max() + 1e-8) + 0.4 * local_var / (local_var.max() + 1e-8)
    rough_img = Image.fromarray((rough * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=max(arr.shape) / 60)
    )
    rough = np.asarray(rough_img, dtype=np.float32)
    rough = (rough - rough.min()) / (rough.max() - rough.min() + 1e-8)
    return rough


def build_heatmap(model, x, img_rgb, pred_idx, method):
    """Return (heat in [0,1] resized to image size, method name used)."""
    heat, used = None, None
    if method in ("Grad-CAM (model attention)", "Auto"):
        try:
            cam = grad_cam(model, x, pred_idx)
            if cam is not None:
                heat, used = cam, "Grad-CAM (model attention)"
        except Exception:
            heat = None
    if heat is None:
        heat, used = terrain_roughness_map(img_rgb), "Terrain roughness (slope + texture)"

    heat_img = Image.fromarray((np.clip(heat, 0, 1) * 255).astype(np.uint8)).resize(
        img_rgb.size, Image.BICUBIC
    )
    heat = np.asarray(heat_img, dtype=np.float32) / 255.0
    return heat, used


def colorize(heat):
    rgba = cm.get_cmap("jet")(heat) if hasattr(cm, "get_cmap") else matplotlib.colormaps["jet"](heat)
    return Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))


def make_overlay(img_rgb, heat_rgb, alpha):
    return Image.blend(img_rgb, heat_rgb, alpha)


# --------------------------------------------------------------------------
# Recommendation
# --------------------------------------------------------------------------
def recommendation(label, p_safe, p_haz):
    conf = max(p_safe, p_haz)
    if conf < 0.60:
        return (
            "rec-warn",
            "⚠️ LOW CONFIDENCE – MANUAL REVIEW ADVISED",
            "The model is uncertain about this terrain. Obtain additional imagery "
            "(different sun angle or higher resolution) and have a mission analyst verify "
            "the site before any landing decision.",
        )
    if label == "SAFE":
        level = "HIGH" if conf >= 0.85 else "MODERATE"
        return (
            "rec-safe",
            f"✅ LANDING RECOMMENDED – {level} CONFIDENCE",
            "The terrain appears smooth and suitable for landing, with no dominant "
            "hazards such as steep craters, large boulders or rugged slopes detected. "
            "This site can be shortlisted as a candidate landing zone.",
        )
    level = "HIGH" if conf >= 0.85 else "MODERATE"
    return (
        "rec-haz",
        f"🚫 LANDING NOT RECOMMENDED – {level} CONFIDENCE",
        "The terrain shows features associated with landing risk (craters, boulders, "
        "steep or rugged surface). Avoid this zone and consider an alternative, smoother site.",
    )


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.markdown(f'<div class="main-title">🌕 {PROJECT_TITLE}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload a lunar surface image to assess landing-zone safety '
    "using a trained Convolutional Neural Network.</div>",
    unsafe_allow_html=True,
)

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Settings")

    mapping_label = st.selectbox(
        "Training class mapping",
        list(MAPPING_OPTIONS.keys()),
        index=0,
        help="Must match the class indices used during training. Keras assigns "
        "indices alphabetically by folder name. The sigmoid output is P(class 1).",
    )
    CLASS_NAMES = MAPPING_OPTIONS[mapping_label]

    scale_mode = st.selectbox(
        "Input scaling",
        ["Auto-detect", "Scale to 0–1 (÷255)", "Raw pixels (0–255)"],
        help="Use the same scaling as training. Auto-detect feeds raw pixels if the "
        "model has its own Rescaling layer, otherwise divides by 255.",
    )

    heat_method = st.selectbox(
        "Heatmap method",
        ["Auto", "Grad-CAM (model attention)", "Terrain roughness"],
        help="Auto tries Grad-CAM and falls back to a terrain roughness map.",
    )
    alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.45, 0.05)

    st.divider()
    st.markdown("**Decision rule**")
    st.markdown(
        f"- Sigmoid output = P(`{CLASS_NAMES[1]}`)\n"
        f"- Threshold = **{THRESHOLD}**\n"
        f"- score ≥ {THRESHOLD} → **{CLASS_NAMES[1]}**\n"
        f"- score < {THRESHOLD} → **{CLASS_NAMES[0]}**"
    )

# ---- Model ----
model_path = find_model_path()

if model_path is None:
    st.error(
        f"❌ Model file **{MODEL_FILENAME}** was not found.\n\n"
        "Place it in the same folder as `app.py` (or in a `models/` sub-folder), "
        "or upload it below."
    )
    uploaded_model = st.file_uploader("Upload the trained model (.keras)", type=["keras"], key="model_up")
    if uploaded_model is None:
        st.stop()
    tmp_dir = tempfile.mkdtemp()
    model_path = os.path.join(tmp_dir, MODEL_FILENAME)
    with open(model_path, "wb") as f:
        f.write(uploaded_model.getvalue())

try:
    model = load_model(model_path)
except ImportError:
    st.error("TensorFlow is not installed. Run: `pip install tensorflow`")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load the model: {e}")
    st.stop()

H, W, C = get_model_input_info(model)
with st.sidebar:
    st.success("Model loaded")
    st.caption(f"Input size: {H}×{W}×{C}")

# ---- Upload ----
uploaded = st.file_uploader(
    "Upload a lunar surface image",
    type=ALLOWED_TYPES,
    help="Supported: " + ", ".join(t.upper() for t in ALLOWED_TYPES),
)

if uploaded is None:
    st.info("👆 Upload an image to begin the analysis.")
    st.stop()

# ---- Process ----
try:
    img_rgb = load_image(uploaded)
except Exception as e:
    st.error(f"❌ Could not read the image. The file may be corrupted or unsupported. ({e})")
    st.stop()

try:
    with st.spinner("Analyzing terrain..."):
        x = preprocess(img_rgb, H, W, C, scale_mode, model)
        result = predict(model, x, CLASS_NAMES)
        heat, heat_used = build_heatmap(model, x, img_rgb, result["pred_idx"], heat_method)
        heat_rgb = colorize(heat)
        overlay = make_overlay(img_rgb, heat_rgb, alpha)
except Exception as e:
    st.error(f"❌ Image processing / prediction failed: {e}")
    st.stop()

# ---- Visuals ----
st.subheader("🔍 Terrain Analysis")
c1, c2, c3 = st.columns(3)
with c1:
    st.image(img_rgb, caption="Original Lunar Image", use_container_width=True)
with c2:
    st.image(heat_rgb, caption=f"Terrain Heatmap – {heat_used}", use_container_width=True)
with c3:
    st.image(overlay, caption="Heatmap Overlay", use_container_width=True)

st.divider()

# ---- Result ----
st.subheader("📋 Classification Result")
left, right = st.columns([1, 1])

is_safe = result["label"] == "SAFE"
with left:
    box_cls = "safe-box" if is_safe else "haz-box"
    icon = "✅" if is_safe else "⚠️"
    st.markdown(
        f'<div class="result-box {box_cls}">{icon} {result["label"]}</div>',
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns(2)
    m1.metric("Safe Probability", f"{result['p_safe'] * 100:.2f}%")
    m2.metric("Hazardous Probability", f"{result['p_hazardous'] * 100:.2f}%")

    st.write("**Safe**")
    st.progress(float(np.clip(result["p_safe"], 0, 1)))
    st.write("**Hazardous**")
    st.progress(float(np.clip(result["p_hazardous"], 0, 1)))

with right:
    st.markdown("**Model Details**")
    d1, d2 = st.columns(2)
    d1.metric("Prediction Score", f"{result['score']:.4f}")
    d2.metric("Threshold", f"{THRESHOLD}")
    st.caption(
        f"Raw sigmoid output = P({CLASS_NAMES[1]}) = {result['score']:.4f}. "
        f"Since {result['score']:.4f} {'≥' if result['score'] >= THRESHOLD else '<'} {THRESHOLD}, "
        f"the image is classified as **{result['label']}**."
    )

    css, title, text = recommendation(result["label"], result["p_safe"], result["p_hazardous"])
    st.markdown(
        f'<div class="rec-box {css}"><b>🚀 Landing Recommendation</b><br>'
        f"<b>{title}</b><br>{text}</div>",
        unsafe_allow_html=True,
    )

with st.expander("ℹ️ How to read the heatmap"):
    st.markdown(
        "- **Red / yellow** areas are the regions that most influenced the model's decision "
        "(Grad-CAM) or the roughest terrain (slope + texture fallback).\n"
        "- **Blue** areas contributed least.\n"
        "- The heatmap is an explanation aid; the final classification comes from the CNN score."
    )

st.caption(
    "This tool is an academic decision-support system. Real landing-site selection requires "
    "additional elevation, slope and illumination data."
)
