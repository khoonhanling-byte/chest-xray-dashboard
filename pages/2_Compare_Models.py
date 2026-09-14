import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from config import DEFAULT_THRESHOLD, MODEL_FILES
from utils.prediction import compare_available_models, model_status
from utils.preprocessing import preprocess_image
from utils.ui import apply_global_styles, disclaimer, hero, model_status_card, section_title

st.set_page_config(page_title="Compare Models", page_icon="🧠", layout="wide")
apply_global_styles()
hero(
    "Compare the Three Models",
    "Run one chest X-ray through VGG16, DenseNet121 and ResNet50 using the same preprocessing",
)

st.markdown(
    '<div class="callout"><b>Research comparison mode</b><br>'
    '<span class="tiny-muted">Model agreement is useful for analysis, but it is not medical consensus. '
    'Displayed inference time is an approximate single-image runtime on the current computer.</span></div>',
    unsafe_allow_html=True,
)

st.divider()
section_title(None, "Model status", "Before analysis")
status = model_status()
status_cols = st.columns(3, gap="large")
for col, model_name in zip(status_cols, MODEL_FILES):
    with col:
        model_status_card(model_name, status[model_name])

st.divider()
section_title(1, "Upload one X-ray for comparison")
uploaded = st.file_uploader("Choose a JPG, JPEG or PNG image", type=["jpg", "jpeg", "png"])

with st.expander("Advanced settings"):
    threshold = st.slider("Decision threshold", 0.10, 0.90, float(DEFAULT_THRESHOLD), 0.05)

if uploaded is None:
    st.caption("The same preprocessed image will be sent to every available model.")
    st.divider()
    disclaimer()
    st.stop()

try:
    pil_image = Image.open(uploaded).convert("RGB")
    original, enhanced, model_input = preprocess_image(pil_image)
except (UnidentifiedImageError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

section_title(2, "Review the shared model input")
preview, processed = st.columns(2, gap="large")
with preview:
    st.markdown("#### Original X-ray")
    st.image(original, caption="Resized original", clamp=True, use_container_width=True)
with processed:
    st.markdown("#### CLAHE-enhanced X-ray")
    st.image(enhanced, caption="Same input used by all models", clamp=True, use_container_width=True)

st.divider()
section_title(3, "Compare VGG16, DenseNet121 and ResNet50")

if st.button("Run three-model comparison", type="primary", use_container_width=True):
    with st.spinner("Loading and analysing with each model one at a time..."):
        rows = compare_available_models(model_input, threshold)

    df = pd.DataFrame(rows)
    ready = df[df["Status"] == "Ready"] if "Status" in df.columns else pd.DataFrame()

    if ready.empty:
        st.error("No trained models are currently available for comparison.")
    else:
        st.markdown("#### At-a-glance results")
        card_cols = st.columns(len(ready), gap="large")
        for card, (_, row) in zip(card_cols, ready.iterrows()):
            with card:
                prediction = row["Prediction"]
                emoji = "🟠" if prediction == "Pneumonia-like" else "🟢"
                st.markdown(
                    f"""
                    <div class="soft-card">
                        <div class="section-kicker">{row['Model']}</div>
                        <div style="font-size:1.25rem;font-weight:900;color:#14213d">{emoji} {prediction}</div>
                        <div style="margin-top:.5rem;color:#475569"><b>{row['Confidence'] * 100:.1f}%</b> model confidence</div>
                        <div style="margin-top:.25rem;color:#64748b"><b>{row['Inference Seconds'] * 1000:.0f} ms</b> approximate inference time</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        counts = ready["Prediction"].value_counts()
        most_common = counts.index[0]
        agreement = int(counts.iloc[0])
        st.markdown(
            f'<div class="callout"><b>Model agreement:</b> {agreement}/{len(ready)} models produced <b>{most_common}</b>.</div>',
            unsafe_allow_html=True,
        )

        fastest_row = ready.loc[ready["Inference Seconds"].idxmin()]
        st.caption(
            f"Fastest prediction in this run: {fastest_row['Model']} "
            f"({fastest_row['Inference Seconds'] * 1000:.0f} ms). "
            "Timing can vary between computers and runs."
        )

        st.markdown("#### Pneumonia-model output")
        for _, row in ready.iterrows():
            st.write(f"**{row['Model']} — {row['Pneumonia'] * 100:.1f}%**")
            st.progress(float(row["Pneumonia"]))

        st.markdown("#### Detailed comparison")
        display_df = df.copy()
        for col in ["Confidence", "Normal", "Pneumonia"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "—")
        if "Inference Seconds" in display_df.columns:
            display_df["Inference Time"] = display_df["Inference Seconds"].apply(
                lambda x: f"{x * 1000:.0f} ms" if pd.notna(x) else "—"
            )
            display_df = display_df.drop(columns=["Inference Seconds"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        chart_df = ready[["Model", "Pneumonia"]].set_index("Model") * 100
        st.bar_chart(chart_df, y="Pneumonia", y_label="Pneumonia model output (%)")

st.divider()
disclaimer()
