from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import DATA_DIR
from utils.prediction import BACKBONE_TO_DISPLAY, get_model_summary, get_recommended_model
from utils.ui import apply_global_styles, hero, section_title

st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")
apply_global_styles()
hero(
    "Model Performance",
    "Technical evaluation for supervisor and research review",
)

summary = get_model_summary()
recommended = get_recommended_model()
auc_path = DATA_DIR / "auc_summary.csv"
auc_df = pd.read_csv(auc_path) if auc_path.exists() else pd.DataFrame()

# -------------------------------------------------------------------
# Top summary cards
# -------------------------------------------------------------------
section_title(None, "Best-model summary", "Overview")
recommended_row = None
if summary:
    for backbone, row in summary.items():
        if BACKBONE_TO_DISPLAY.get(backbone.lower()) == recommended:
            recommended_row = row
            break

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Recommended model", recommended)
with c2:
    value = "—"
    if recommended_row:
        value = f"{float(recommended_row.get('test_accuracy', 0)) * 100:.2f}%"
    st.metric("Test accuracy", value)
with c3:
    value = "—"
    if recommended_row:
        value = f"{float(recommended_row.get('recall', 0)) * 100:.2f}%"
    st.metric("Recall", value)
with c4:
    value = "—"
    if recommended_row:
        value = f"{float(recommended_row.get('f1_score', 0)) * 100:.2f}%"
    st.metric("F1-score", value)

if not auc_df.empty and {"model", "roc_auc", "average_precision"}.issubset(auc_df.columns):
    match = auc_df[auc_df["model"] == recommended]
    if not match.empty:
        st.caption(
            f"{recommended}: ROC AUC {float(match.iloc[0]['roc_auc']):.3f} • "
            f"Average Precision {float(match.iloc[0]['average_precision']):.3f}"
        )

st.divider()

# -------------------------------------------------------------------
# Dataset overview
# -------------------------------------------------------------------
section_title(None, "Dataset overview", "Data")
dataset_path = DATA_DIR / "dataset_summary.csv"
if dataset_path.exists():
    dataset_df = pd.read_csv(dataset_path)
    if {"split", "class", "count"}.issubset(dataset_df.columns):
        pivot = dataset_df.pivot_table(index="split", columns="class", values="count", aggfunc="sum", fill_value=0)
        left, right = st.columns([1.05, 0.95], gap="large")
        with left:
            st.dataframe(dataset_df, use_container_width=True, hide_index=True)
        with right:
            st.bar_chart(pivot)
else:
    st.info("Dataset summary will appear after the updated Colab dashboard bundle is installed.")

st.divider()

# -------------------------------------------------------------------
# Benchmark table
# -------------------------------------------------------------------
section_title(None, "Three-model benchmark", "Model comparison")
latest_benchmark_path = DATA_DIR / "benchmark_results.csv"
if latest_benchmark_path.exists():
    benchmark = pd.read_csv(latest_benchmark_path)
    pretty = benchmark.copy()
    if "backbone" in pretty.columns:
        pretty["Model"] = pretty["backbone"].map(lambda x: BACKBONE_TO_DISPLAY.get(str(x).lower(), str(x)))
    for col in ["test_accuracy", "precision", "recall", "f1_score"]:
        if col in pretty.columns:
            pretty[col] = (pretty[col] * 100).map(lambda x: f"{x:.2f}%")

    preferred = [
        "Model", "dense_units", "dropout_rate", "fine_tune_ratio",
        "stage1_epochs_run", "stage2_epochs_run",
        "test_accuracy", "precision", "recall", "f1_score",
    ]
    cols = [c for c in preferred if c in pretty.columns]
    st.dataframe(pretty[cols] if cols else pretty, use_container_width=True, hide_index=True)

    if {"backbone", "test_accuracy", "f1_score"}.issubset(benchmark.columns):
        chart = benchmark.copy()
        chart["Model"] = chart["backbone"].map(lambda x: BACKBONE_TO_DISPLAY.get(str(x).lower(), str(x)))
        chart_df = chart.groupby("Model")[["test_accuracy", "f1_score"]].max() * 100
        st.bar_chart(chart_df)
else:
    st.info("Latest benchmark results are not installed yet.")

st.divider()

# -------------------------------------------------------------------
# ROC and Precision-Recall curves
# -------------------------------------------------------------------
section_title(None, "Threshold-based evaluation", "Advanced evaluation")
roc_path = DATA_DIR / "roc_curves.csv"
pr_path = DATA_DIR / "pr_curves.csv"

if roc_path.exists() and pr_path.exists():
    roc_df = pd.read_csv(roc_path)
    pr_df = pd.read_csv(pr_path)

    tab1, tab2 = st.tabs(["ROC curve", "Precision–Recall curve"])

    with tab1:
        fig, ax = plt.subplots(figsize=(7, 5))
        for model, group in roc_df.groupby("model"):
            label = model
            if not auc_df.empty and "model" in auc_df.columns:
                match = auc_df[auc_df["model"] == model]
                if not match.empty:
                    label = f"{model} (AUC={float(match.iloc[0]['roc_auc']):.3f})"
            ax.plot(group["fpr"], group["tpr"], label=label)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="No-skill")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Test Set")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
        st.caption("Curves closer to the top-left generally indicate stronger discrimination across thresholds.")

    with tab2:
        fig, ax = plt.subplots(figsize=(7, 5))
        for model, group in pr_df.groupby("model"):
            label = model
            if not auc_df.empty and "model" in auc_df.columns:
                match = auc_df[auc_df["model"] == model]
                if not match.empty:
                    label = f"{model} (AP={float(match.iloc[0]['average_precision']):.3f})"
            ax.plot(group["recall"], group["precision"], label=label)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision–Recall Curve — Test Set")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
        st.caption("This curve shows the trade-off between precision and recall as the decision threshold changes.")
else:
    st.info("ROC and Precision–Recall curves will appear after installing the updated Colab export.")

st.divider()

# -------------------------------------------------------------------
# Confusion matrix
# -------------------------------------------------------------------
section_title(None, "Confusion matrix", "Errors")
cm_path = DATA_DIR / "confusion_matrices.csv"
if cm_path.exists():
    cm_df = pd.read_csv(cm_path)
    model_choices = cm_df["model"].unique().tolist()
    selected_model = st.selectbox("Choose model for confusion matrix", model_choices)
    row = cm_df[cm_df["model"] == selected_model].iloc[0]
    cm = np.array([[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]])

    fig, ax = plt.subplots(figsize=(5.5, 4.7))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["NORMAL", "PNEUMONIA"])
    ax.set_yticks([0, 1], labels=["NORMAL", "PNEUMONIA"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(f"{selected_model} — Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("True negatives", int(row["tn"]))
    m2.metric("False positives", int(row["fp"]))
    m3.metric("False negatives", int(row["fn"]))
    m4.metric("True positives", int(row["tp"]))
else:
    st.info("Per-model confusion matrices will appear after installing the updated Colab export.")

st.divider()

# -------------------------------------------------------------------
# Training curves
# -------------------------------------------------------------------
section_title(None, "Training curves", "Learning behaviour")
history_files = {
    "VGG16": DATA_DIR / "history_vgg16.csv",
    "DenseNet121": DATA_DIR / "history_densenet121.csv",
    "ResNet50": DATA_DIR / "history_resnet50.csv",
}
available_histories = {k: v for k, v in history_files.items() if v.exists()}
if available_histories:
    selected_history_model = st.selectbox("Choose model for training curves", list(available_histories.keys()))
    history_df = pd.read_csv(available_histories[selected_history_model])

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Accuracy")
        if {"epoch", "accuracy", "val_accuracy"}.issubset(history_df.columns):
            st.line_chart(history_df.set_index("epoch")[["accuracy", "val_accuracy"]])
    with right:
        st.markdown("#### Loss")
        if {"epoch", "loss", "val_loss"}.issubset(history_df.columns):
            st.line_chart(history_df.set_index("epoch")[["loss", "val_loss"]])
    st.caption("The vertical Stage column in the exported table identifies Stage 1 and Stage 2 epochs.")
    with st.expander("View training-history data"):
        st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    st.info("Best-model training histories will appear after installing the updated Colab export.")

st.divider()

# -------------------------------------------------------------------
# Preprocessing metrics
# -------------------------------------------------------------------
section_title(None, "Preprocessing measurements", "Image enhancement")
latest_preprocess_path = DATA_DIR / "latest_preprocessing_metrics.csv"
fallback_preprocess_path = DATA_DIR / "preprocessing_metrics.csv"
preprocess_path = latest_preprocess_path if latest_preprocess_path.exists() else fallback_preprocess_path
if preprocess_path.exists():
    preprocess = pd.read_csv(preprocess_path)
    st.dataframe(preprocess, use_container_width=True, hide_index=True)
    st.caption(
        "These metrics describe brightness, contrast variation, information variation and edge strength. "
        "They should be interpreted together with model-performance experiments."
    )
else:
    st.info("Preprocessing metrics are not available.")
