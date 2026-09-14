from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from config import BASE_DIR, DEFAULT_THRESHOLD, MODEL_FILES
from utils.gradcam import make_gradcam_heatmap, overlay_heatmap
from utils.prediction import get_recommended_model, load_model_cached, model_exists, predict_binary
from utils.preprocessing import preprocess_image
from utils.ui import apply_global_styles, disclaimer, hero, result_banner, section_title

st.set_page_config(page_title="Test Model", page_icon="🧪", layout="wide")
apply_global_styles()
hero(
    "Test Model with Labelled Samples",
    "Use known test-set examples to compare the ground-truth label with the AI prediction",
)

samples_csv = BASE_DIR / "data" / "test_samples.csv"
if not samples_csv.exists():
    st.info(
        "Labelled test samples are not installed yet. Run the updated Colab notebook and reinstall the exported dashboard bundle."
    )
    disclaimer()
    st.stop()

samples = pd.read_csv(samples_csv)
required_cols = {"sample_id", "actual_label", "image_path"}
if not required_cols.issubset(samples.columns):
    st.error("test_samples.csv is missing required columns.")
    st.stop()

section_title(1, "Choose a labelled test example")
class_options = sorted(samples["actual_label"].dropna().unique().tolist())
selected_class = st.selectbox("Ground-truth class", class_options)
class_rows = samples[samples["actual_label"] == selected_class].reset_index(drop=True)

if "test_sample_index" not in st.session_state:
    st.session_state.test_sample_index = 0

c1, c2 = st.columns([1, 1])
with c1:
    sample_choice = st.selectbox(
        "Sample",
        class_rows["sample_id"].tolist(),
        index=min(st.session_state.test_sample_index, max(len(class_rows) - 1, 0)),
    )
with c2:
    st.write("")
    st.write("")
    if st.button("🎲 Random sample", use_container_width=True):
        if len(class_rows) > 0:
            st.session_state.test_sample_index = random.randrange(len(class_rows))
            st.rerun()

row = class_rows[class_rows["sample_id"] == sample_choice].iloc[0]
image_path = BASE_DIR / row["image_path"]
if not image_path.exists():
    st.error(f"Sample image is missing: {image_path}")
    st.stop()

pil_image = Image.open(image_path).convert("RGB")
original, enhanced, model_input = preprocess_image(pil_image)

left, right = st.columns(2, gap="large")
with left:
    st.markdown("#### Labelled test image")
    st.image(original, use_container_width=True)
with right:
    st.markdown("#### CLAHE-enhanced model input")
    st.image(enhanced, use_container_width=True)

st.markdown(f'<div class="callout"><b>Ground truth:</b> {selected_class}</div>', unsafe_allow_html=True)

st.divider()
section_title(2, "Choose a model and run the test")
recommended = get_recommended_model()
model_name = st.selectbox(
    "Model",
    list(MODEL_FILES.keys()),
    index=list(MODEL_FILES.keys()).index(recommended) if recommended in MODEL_FILES else 0,
)
threshold = DEFAULT_THRESHOLD

if not model_exists(model_name):
    st.warning(f"{model_name} model file is missing.")
    st.stop()

if st.button("Run labelled test", type="primary", use_container_width=True):
    with st.spinner(f"Analysing with {model_name}..."):
        model = load_model_cached(model_name)
        result = predict_binary(model, model_input, threshold)

    section_title(3, "Compare prediction with the known label")
    result_banner(result["display_class"], result["confidence"])

    predicted_label = "PNEUMONIA" if result["display_class"] == "Pneumonia-like" else "NORMAL"
    correct = predicted_label.upper() == str(selected_class).upper()

    a, b, c = st.columns(3)
    with a:
        st.metric("Ground truth", selected_class)
    with b:
        st.metric("Model prediction", predicted_label)
    with c:
        st.metric("Outcome", "Correct" if correct else "Incorrect")

    if correct:
        st.success("The model correctly classified this labelled test image.")
    else:
        st.error("The model misclassified this labelled test image.")

    try:
        heatmap = make_gradcam_heatmap(
            model,
            model_input,
            explain_positive=result["display_class"] == "Pneumonia-like",
        )
        overlay = overlay_heatmap(enhanced, heatmap)
        st.markdown("#### Grad-CAM explanation")
        st.image(overlay, caption=f"{model_name} attention map", use_container_width=True)
    except Exception as exc:
        st.info(f"Grad-CAM unavailable for this sample: {exc}")

st.divider()
disclaimer()
