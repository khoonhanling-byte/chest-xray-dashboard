import streamlit as st

from config import APP_TITLE, MODEL_FILES
from utils.prediction import get_recommended_model, model_status
from utils.ui import apply_global_styles, disclaimer, hero, model_status_card, section_title

st.set_page_config(page_title=APP_TITLE, page_icon="🩻", layout="wide")
apply_global_styles()

recommended_model = get_recommended_model()
status = model_status()

hero(
    APP_TITLE,
    "Upload a chest X-ray, enhance it with CLAHE, analyse it with transfer learning, and inspect the model's visual explanation.",
)

left, right = st.columns([1.18, 0.82], gap="large")

with left:
    st.markdown('<div class="section-kicker">Interactive analysis</div>', unsafe_allow_html=True)
    st.header("From X-ray to understandable AI output")
    st.write(
        "The main analysis is designed for non-technical users, while labelled testing, error inspection, "
        "model comparison and technical evaluation support the research side of the FYP."
    )
    b1, b2 = st.columns(2)
    with b1:
        st.page_link("pages/1_Analyse_Xray.py", label="Start X-ray analysis", icon="🩻", use_container_width=True)
    with b2:
        st.page_link("pages/2_Compare_Models.py", label="Compare all models", icon="🧠", use_container_width=True)

with right:
    st.markdown(
        f"""
        <div class="soft-card">
            <div class="section-kicker">Recommended model</div>
            <div style="font-size:1.6rem;font-weight:900;color:#14213d">{recommended_model}</div>
            <p>Automatically selected from the exported model summary using the strongest F1-score.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="flow-wrap">
        <div class="flow-item">① Upload X-ray</div>
        <div class="flow-item">② CLAHE enhancement</div>
        <div class="flow-item">③ AI classification</div>
        <div class="flow-item">④ Grad-CAM explanation</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()
section_title(None, "Core dashboard features", "User-facing")
f1, f2, f3 = st.columns(3, gap="large")
with f1:
    st.markdown(
        '<div class="feature-card"><div class="feature-icon">🩻</div><div class="feature-title">Original vs CLAHE</div><div class="feature-text">See the uploaded X-ray and the contrast-enhanced image used by the model.</div></div>',
        unsafe_allow_html=True,
    )
with f2:
    st.markdown(
        '<div class="feature-card"><div class="feature-icon">🧠</div><div class="feature-title">Three AI models</div><div class="feature-text">Use VGG16, DenseNet121 or ResNet50, or compare all three on one X-ray.</div></div>',
        unsafe_allow_html=True,
    )
with f3:
    st.markdown(
        '<div class="feature-card"><div class="feature-icon">🔥</div><div class="feature-title">Grad-CAM explanation</div><div class="feature-text">Show which image regions influenced the selected model output.</div></div>',
        unsafe_allow_html=True,
    )

st.divider()
section_title(None, "Research tools", "New in this version")
r1, r2, r3 = st.columns(3, gap="large")
with r1:
    st.markdown(
        '<div class="feature-card"><div class="feature-icon">🧪</div><div class="feature-title">Labelled Test Model</div><div class="feature-text">Use known test-set examples and immediately see whether the model was correct.</div></div>',
        unsafe_allow_html=True,
    )
    st.page_link("pages/3_Test_Model.py", label="Open labelled test", icon="🧪", use_container_width=True)
with r2:
    st.markdown(
        '<div class="feature-card"><div class="feature-icon">🔎</div><div class="feature-title">Misclassified cases</div><div class="feature-text">Inspect false negatives and false positives instead of relying only on a single accuracy number.</div></div>',
        unsafe_allow_html=True,
    )
    st.page_link("pages/4_Misclassified_Cases.py", label="Inspect model errors", icon="🔎", use_container_width=True)
with r3:
    st.markdown(
        '<div class="feature-card"><div class="feature-icon">📈</div><div class="feature-title">ROC & Precision–Recall</div><div class="feature-text">Review threshold-based performance, confusion matrices and training curves.</div></div>',
        unsafe_allow_html=True,
    )
    st.page_link("pages/5_Model_Performance.py", label="View model performance", icon="📊", use_container_width=True)

st.divider()
section_title(None, "Model availability", "System status")
cols = st.columns(3, gap="large")
model_details = {
    "VGG16": "Simple sequential CNN baseline",
    "DenseNet121": "Dense feature reuse",
    "ResNet50": "Residual feature learning",
}
for col, model_name in zip(cols, MODEL_FILES):
    with col:
        model_status_card(model_name, status[model_name], model_details.get(model_name, ""))

if not all(status.values()):
    st.info(
        "One or more trained model files are missing. Install the exported Colab model bundle before running predictions."
    )

st.divider()
section_title(None, "More", "Project")
c1, c2 = st.columns(2)
with c1:
    st.page_link("pages/6_Analysis_History.py", label="View session history", icon="🕘", use_container_width=True)
with c2:
    st.page_link("pages/7_About.py", label="About the project", icon="ℹ️", use_container_width=True)

st.divider()
disclaimer()
