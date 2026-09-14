from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from config import (
    CLASS_NAMES,
    DEFAULT_THRESHOLD,
    FALLBACK_RECOMMENDED_MODEL,
    MODEL_FILES,
    MODEL_SUMMARY_FILE,
    POSITIVE_CLASS_INDEX,
)

BACKBONE_TO_DISPLAY = {
    "vgg16": "VGG16",
    "densenet121": "DenseNet121",
    "resnet50": "ResNet50",
}


def model_exists(model_name: str) -> bool:
    path = MODEL_FILES.get(model_name)
    return bool(path and Path(path).exists())


def model_status() -> dict[str, bool]:
    return {name: model_exists(name) for name in MODEL_FILES}


def get_model_summary() -> dict:
    if not MODEL_SUMMARY_FILE.exists():
        return {}
    try:
        with open(MODEL_SUMMARY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def get_recommended_model() -> str:
    """Use the highest-F1 exported backbone when summary metadata exists."""
    summary = get_model_summary()
    candidates = []

    for backbone, row in summary.items():
        if not isinstance(row, dict):
            continue
        display_name = BACKBONE_TO_DISPLAY.get(backbone.lower())
        if display_name not in MODEL_FILES:
            continue
        try:
            f1 = float(row.get("f1_score", -1.0))
        except (TypeError, ValueError):
            f1 = -1.0
        candidates.append((f1, display_name))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    return FALLBACK_RECOMMENDED_MODEL


@st.cache_resource(show_spinner=False)
def load_model_cached(model_name: str):
    """Load one selected model and cache it for fast repeated predictions."""
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}")

    path = MODEL_FILES[model_name]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model file: {path.name}. Run the Colab notebook, export the three-model bundle, "
            "then install it into this dashboard."
        )

    return tf.keras.models.load_model(path, compile=False)


def load_model_uncached(model_name: str):
    """Load a model without caching; useful for low-memory multi-model comparison."""
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}")

    path = MODEL_FILES[model_name]
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path.name}")

    return tf.keras.models.load_model(path, compile=False)


def predict_binary(model, model_input: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Run the project's one-output sigmoid classifier."""
    raw = np.asarray(model.predict(model_input, verbose=0))

    if raw.size != 1:
        raise ValueError(
            "This dashboard expects the FYP binary sigmoid model with one output value. "
            f"Received prediction shape {raw.shape}."
        )

    pneumonia_probability = float(np.clip(raw.reshape(-1)[0], 0.0, 1.0))
    normal_probability = 1.0 - pneumonia_probability

    if pneumonia_probability >= threshold:
        predicted_index = POSITIVE_CLASS_INDEX
        confidence = pneumonia_probability
    else:
        predicted_index = 1 - POSITIVE_CLASS_INDEX
        confidence = normal_probability

    return {
        "predicted_index": predicted_index,
        "predicted_class": CLASS_NAMES[predicted_index],
        "display_class": "Pneumonia-like" if predicted_index == POSITIVE_CLASS_INDEX else "Normal-like",
        "confidence": confidence,
        "normal_probability": normal_probability,
        "pneumonia_probability": pneumonia_probability,
        "threshold": threshold,
    }


def timed_prediction(model, model_input: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> tuple[dict, float]:
    """Return a prediction plus approximate single-image inference time in seconds."""
    start = time.perf_counter()
    result = predict_binary(model, model_input, threshold)
    elapsed = time.perf_counter() - start
    return result, elapsed


def compare_available_models(model_input: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """Compare all three models sequentially to reduce peak memory use."""
    results = []

    for model_name in MODEL_FILES:
        if not model_exists(model_name):
            results.append({"Model": model_name, "Status": "Model file missing"})
            continue

        model = None
        try:
            model = load_model_uncached(model_name)

            # Warm-up call to reduce one-off graph setup effects in the displayed timing.
            _ = model.predict(model_input, verbose=0)

            pred, elapsed = timed_prediction(model, model_input, threshold)
            results.append(
                {
                    "Model": model_name,
                    "Status": "Ready",
                    "Prediction": pred["display_class"],
                    "Confidence": pred["confidence"],
                    "Normal": pred["normal_probability"],
                    "Pneumonia": pred["pneumonia_probability"],
                    "Inference Seconds": elapsed,
                }
            )
        finally:
            if model is not None:
                del model
            tf.keras.backend.clear_session()
            gc.collect()

    return results
