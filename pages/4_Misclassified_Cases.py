from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

from config import BASE_DIR
from utils.ui import apply_global_styles, disclaimer, hero

st.set_page_config(page_title="Misclassified Cases", page_icon="🔎", layout="wide")
apply_global_styles()
hero(
    "Misclassified Test Cases",
    "Inspect false negatives and false positives to understand where each model struggles",
)

cases_csv = BASE_DIR / "data" / "misclassified_cases.csv"
if not cases_csv.exists():
    st.info(
        "Misclassified-case examples are not installed yet. Run the updated Colab notebook and reinstall the exported dashboard bundle."
    )
    disclaimer()
    st.stop()

cases = pd.read_csv(cases_csv)
required_cols = {"model", "error_type", "actual_label", "predicted_label", "confidence", "image_path"}
if not required_cols.issubset(cases.columns):
    st.error("misclassified_cases.csv is missing required columns.")
    st.stop()

model = st.selectbox("Model", sorted(cases["model"].unique().tolist()))
error_type = st.radio(
    "Error type",
    ["False Negative", "False Positive"],
    horizontal=True,
)

filtered = cases[(cases["model"] == model) & (cases["error_type"] == error_type)].reset_index(drop=True)

if error_type == "False Negative":
    st.warning(
        "False negative: the actual image is Pneumonia, but the model predicted Normal. These cases are especially important to inspect."
    )
else:
    st.info("False positive: the actual image is Normal, but the model predicted Pneumonia.")

if filtered.empty:
    st.success(f"No saved {error_type.lower()} examples were found for {model} in the exported sample set.")
    disclaimer()
    st.stop()

cols_per_row = 3
for start in range(0, len(filtered), cols_per_row):
    cols = st.columns(cols_per_row, gap="large")
    for col, (_, row) in zip(cols, filtered.iloc[start:start + cols_per_row].iterrows()):
        with col:
            image_path = BASE_DIR / row["image_path"]
            if image_path.exists():
                image = Image.open(image_path).convert("RGB")
                st.image(image, use_container_width=True)
            st.markdown(
                f"**Actual:** {row['actual_label']}  \n"
                f"**Predicted:** {row['predicted_label']}  \n"
                f"**Confidence:** {float(row['confidence']) * 100:.1f}%"
            )
            if "pneumonia_probability" in row and pd.notna(row["pneumonia_probability"]):
                st.caption(f"Pneumonia-model output: {float(row['pneumonia_probability']) * 100:.1f}%")

st.divider()
disclaimer()
