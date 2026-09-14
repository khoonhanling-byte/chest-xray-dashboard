import streamlit as st

from config import CLASS_NAMES, IMAGE_SIZE
from utils.ui import apply_global_styles, disclaimer, hero

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
apply_global_styles()
hero("About the Project", "Deep Learning in Medical Images Analysis Using Transfer Learning")

st.subheader("Project scope")
st.markdown(
    f"""
- **Task:** binary chest X-ray classification ({CLASS_NAMES[0]} vs {CLASS_NAMES[1]})
- **Input size:** {IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}
- **Models:** VGG16, ResNet50 and DenseNet121
- **Preprocessing:** CLAHE + normalization + 3-channel conversion
- **Classifier:** modified transfer-learning head with dense layer, batch normalization, dropout and sigmoid output
- **Training:** Stage 1 frozen backbone, Stage 2 partial fine-tuning
"""
)

st.subheader("Two-stage training")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Stage 1 — feature extraction**\n\n- Backbone frozen\n- Train classifier head\n- Larger learning rate")
with c2:
    st.markdown("**Stage 2 — fine-tuning**\n\n- Upper backbone layers trainable\n- Smaller learning rate\n- Carefully adapt to chest X-rays")

st.subheader("How to interpret the dashboard")
st.markdown(
    """
- **Prediction:** which learned class the selected model favours.
- **Model confidence:** strength of the model output for the predicted class, not medical certainty.
- **Grad-CAM:** regions that influenced the model output; not confirmed disease locations.
- **Model agreement:** whether the three AI models produced the same class; not a clinical consensus.
- **Labelled Test Model page:** for research verification using known test-set labels.
"""
)

st.subheader("Limitations")
st.markdown(
    """
- The system only distinguishes **Normal** and **Pneumonia**.
- It was trained on a specific chest X-ray dataset and may not generalise equally to all hospitals, devices or patient populations.
- Non-chest-X-ray inputs can produce meaningless predictions.
- The dashboard is a research prototype and is not intended for clinical diagnosis.
"""
)

disclaimer()
