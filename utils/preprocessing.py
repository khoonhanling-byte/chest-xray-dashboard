from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from config import IMAGE_SIZE


def _to_grayscale_uint8(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert an uploaded image to a 2-D uint8 grayscale array."""
    if isinstance(image, Image.Image):
        rgb = np.array(image.convert("RGB"))
    else:
        rgb = np.asarray(image)
        if rgb.ndim == 2:
            return rgb.astype(np.uint8)
        if rgb.ndim == 3 and rgb.shape[-1] == 1:
            return np.squeeze(rgb, axis=-1).astype(np.uint8)

    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError("Unsupported image format. Please upload a JPG, JPEG, or PNG chest X-ray.")

    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply the same CLAHE settings used in the FYP pipeline."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray.astype(np.uint8))


def preprocess_image(image: Image.Image | np.ndarray):
    """
    Resize -> grayscale -> CLAHE -> normalize -> duplicate to 3 channels.

    Returns
    -------
    original_resized : uint8 2-D image
    enhanced : uint8 2-D image
    model_input : float32 array, shape (1, 224, 224, 3), range [0, 1]
    """
    gray = _to_grayscale_uint8(image)
    original_resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    enhanced = apply_clahe(original_resized)

    normalized = enhanced.astype(np.float32) / 255.0
    rgb_3ch = np.stack([normalized] * 3, axis=-1)
    model_input = np.expand_dims(rgb_3ch, axis=0).astype(np.float32)
    return original_resized, enhanced, model_input


def image_metrics(gray: np.ndarray) -> dict[str, float]:
    """Compute the four image-quality measures used in the project."""
    gray = gray.astype(np.uint8)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    probs = hist / max(hist.sum(), 1.0)
    probs = probs[probs > 0]
    entropy = float(-(probs * np.log2(probs)).sum())

    return {
        "Mean intensity": float(np.mean(gray)),
        "Standard deviation": float(np.std(gray)),
        "Entropy": entropy,
        "Laplacian variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }
