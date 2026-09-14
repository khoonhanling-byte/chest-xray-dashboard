from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps

from utils.gradcam import make_gradcam


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
THRESHOLD = 0.50

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_FILES = {
    "VGG16":
        BASE_DIR / "models" / "vgg16_best.keras",

    "DenseNet121":
        BASE_DIR / "models" / "densenet121_best.keras",

    "ResNet50":
        BASE_DIR / "models" / "resnet50_best.keras",
}


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>

    .analyse-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 5px;
    }

    .analyse-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 25px;
    }

    .result-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .normal-text {
        color: #15803d;
        font-size: 1.8rem;
        font-weight: 700;
    }

    .pneumonia-text {
        color: #b45309;
        font-size: 1.8rem;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="analyse-title">🩻 Analyse X-ray</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="analyse-subtitle">
        Upload a chest X-ray and analyse it using one of the
        trained transfer-learning models.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_name):
    """
    Load and cache selected trained model.
    """

    model_path = MODEL_FILES[
        model_name
    ]

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )


    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):
    """
    Same preprocessing used during final model training:

    1. Grayscale
    2. Resize to 224 x 224
    3. CLAHE
    4. Three identical channels
    5. Divide by 255
    """

    image = ImageOps.exif_transpose(
        image
    )


    # --------------------------------------------------------
    # Original image preview
    # --------------------------------------------------------

    original = image.convert(
        "RGB"
    )

    # Resize preview so before/after images
    # have the same displayed dimensions.
    original_preview = original.resize(
        IMG_SIZE
    )


    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = np.array(
        image.convert("L"),
        dtype=np.uint8
    )


    # --------------------------------------------------------
    # Resize to 224 x 224
    # --------------------------------------------------------

    gray = cv2.resize(
        gray,
        IMG_SIZE,
        interpolation=cv2.INTER_LINEAR
    )


    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )


    # --------------------------------------------------------
    # Three identical channels
    # --------------------------------------------------------

    enhanced_rgb = np.stack(
        [enhanced] * 3,
        axis=-1
    )


    # --------------------------------------------------------
    # Normalisation [0, 1]
    # --------------------------------------------------------

    model_input = (
        enhanced_rgb.astype(np.float32)
        / 255.0
    )


    # Add batch dimension
    model_input = np.expand_dims(
        model_input,
        axis=0
    )


    return (
        original_preview,
        enhanced_rgb,
        model_input
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_image(
    model,
    model_input
):
    """
    Binary sigmoid classification.

    Class:
        0 = Normal
        1 = Pneumonia
    """

    prediction = model.predict(
        model_input,
        verbose=0
    )


    pneumonia_probability = float(
        np.asarray(
            prediction
        ).reshape(-1)[0]
    )


    pneumonia_probability = float(
        np.clip(
            pneumonia_probability,
            0.0,
            1.0
        )
    )


    normal_probability = (
        1.0
        - pneumonia_probability
    )


    if pneumonia_probability >= THRESHOLD:

        predicted_class = (
            "Pneumonia-like"
        )

        confidence = (
            pneumonia_probability
        )

    else:

        predicted_class = (
            "Normal-like"
        )

        confidence = (
            normal_probability
        )


    return {
        "prediction":
            predicted_class,

        "confidence":
            confidence,

        "normal_probability":
            normal_probability,

        "pneumonia_probability":
            pneumonia_probability
    }


# ============================================================
# SESSION HISTORY
# ============================================================

if "analysis_history" not in st.session_state:

    st.session_state.analysis_history = []


# ============================================================
# MODEL SELECTION
# ============================================================

model_name = st.selectbox(
    "Select model",
    [
        "VGG16",
        "DenseNet121",
        "ResNet50"
    ]
)


if model_name == "VGG16":

    st.caption(
        "Recommended model — VGG16 achieved the "
        "best overall result in the final experiment."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload chest X-ray",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp"
    ]
)


if uploaded_file is None:

    st.info(
        "Upload a chest X-ray image to begin the analysis."
    )

    st.stop()


# ============================================================
# PROCESS IMAGE
# ============================================================

try:

    uploaded_image = Image.open(
        uploaded_file
    )


    (
        original_image,
        enhanced_image,
        model_input
    ) = preprocess_image(
        uploaded_image
    )


except Exception as error:

    st.error(
        "The uploaded image could not be processed."
    )

    st.exception(error)

    st.stop()


# ============================================================
# IMAGE PREVIEW
# ============================================================

st.subheader(
    "Image Preview"
)


col1, col2 = st.columns(2)


with col1:

    st.markdown(
        "**Original X-ray**"
    )

    st.image(
        original_image,
        use_container_width=True
    )


with col2:

    st.markdown(
        "**CLAHE-enhanced X-ray**"
    )

    st.image(
        enhanced_image,
        use_container_width=True
    )


# ============================================================
# ANALYSE BUTTON
# ============================================================

analyse = st.button(
    "Analyse Image",
    type="primary",
    use_container_width=True
)


if analyse:

    try:

        with st.spinner(
            f"Analysing with {model_name}..."
        ):

            # ------------------------------------------------
            # Load model
            # ------------------------------------------------

            model = load_model(
                model_name
            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            result = predict_image(
                model,
                model_input
            )


            # ------------------------------------------------
            # Grad-CAM
            # ------------------------------------------------

            gradcam_result = make_gradcam(
                model=model,
                model_input=model_input,
                display_image=enhanced_image,
                threshold=THRESHOLD,
                alpha=0.45
            )


        # ====================================================
        # RESULT
        # ====================================================

        st.subheader(
            "Prediction Result"
        )


        if result["prediction"] == "Normal-like":

            result_class = (
                "normal-text"
            )

        else:

            result_class = (
                "pneumonia-text"
            )


        st.markdown(
        f"""<div class="result-box">
        <div class="{result_class}">{result["prediction"]}</div>
         <br>
        Model: <b>{model_name}</b>
         &nbsp;&nbsp; | &nbsp;&nbsp;
        Confidence: <b>{result["confidence"] * 100:.2f}%</b>
        </div>""",
        unsafe_allow_html=True
          ) 
         

        # ====================================================
        # PROBABILITIES
        # ====================================================

        p1, p2 = st.columns(2)


        with p1:

            st.metric(
                "Normal probability",
                f'{result["normal_probability"] * 100:.2f}%'
            )


        with p2:

            st.metric(
                "Pneumonia probability",
                f'{result["pneumonia_probability"] * 100:.2f}%'
            )


        if result["confidence"] < 0.70:

            st.warning(
                "The model produced a relatively "
                "low-confidence classification."
            )


        # ====================================================
        # GRAD-CAM
        # ====================================================

        st.subheader(
            "Grad-CAM Explanation"
        )


        g1, g2 = st.columns(2)


        with g1:

            st.markdown(
                "**Model input**"
            )

            st.image(
                enhanced_image,
                caption="CLAHE-enhanced input",
                use_container_width=True
            )


        with g2:

            st.markdown(
                "**Grad-CAM overlay**"
            )

            st.image(
                gradcam_result["overlay"],
                caption=f"{model_name} attention map",
                use_container_width=True
            )


        st.info(
            "Blue represents lower model influence, "
            "followed by green and yellow, while red "
            "represents the strongest influence."
        )


        st.caption(
            "Grad-CAM shows regions that influenced the "
            "model prediction. It does not confirm the "
            "location of pneumonia."
        )


        # ====================================================
        # SAVE SESSION HISTORY
        # ====================================================

        st.session_state.analysis_history.append(
            {
                "Time":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "Image":
                    uploaded_file.name,

                "Model":
                    model_name,

                "Prediction":
                    result["prediction"],

                "Confidence":
                    round(
                        result["confidence"] * 100,
                        2
                    ),

                "Normal Probability":
                    round(
                        result[
                            "normal_probability"
                        ] * 100,
                        2
                    ),

                "Pneumonia Probability":
                    round(
                        result[
                            "pneumonia_probability"
                        ] * 100,
                        2
                    ),

                "Threshold":
                    THRESHOLD
            }
        )


    except Exception as error:

        st.error(
            "The image analysis could not be completed."
        )

        st.exception(error)


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "This dashboard is an academic research prototype. "
    "The model prediction, confidence and Grad-CAM "
    "visualisation do not replace professional medical diagnosis."
)