"""Raw predictions and a separate presentation adapter. No HTTP service."""
import json
from pathlib import Path

import numpy as np

from artlens.data import config_hash
from artlens.prepare import digest
from artlens.preprocessing import CLASSES, CONFIG, ann_features, cnn_tensor


def presentation_scores(raw):
    """Normalize nonnegative sigmoid scores by their sum; not calibrated probabilities.

    All-zero numerical saturation falls back to uniform scores. Argmax ties always
    select the first class index. Raw outputs remain available to callers.
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape[-1:] != (5,) or not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
        raise ValueError("Expected finite five-class activations in [0,1]")
    total = raw.sum(axis=-1, keepdims=True)
    return np.divide(raw, total, out=np.full_like(raw, 0.2), where=total > 0)


def load_artifact(directory):
    import tensorflow as tf
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["classes"] != list(CLASSES) or metadata["preprocessing_hash"] != config_hash() or metadata["preprocessing"] != CONFIG:
        raise ValueError("Artifact class order/preprocessing is incompatible")
    if digest(directory / "model.keras") != metadata["model_sha256"]:
        raise ValueError("Model checksum mismatch")
    model = tf.keras.models.load_model(directory / "model.keras", compile=False)
    return model, metadata


PREPROCESSORS = {"simple_ann": ann_features, "cnn": cnn_tensor}


def raw_outputs(model, model_input):
    """Raw output activations for one preprocessed input (no batch dimension)."""
    return np.asarray(model(model_input[None], training=False))[0]


def predict_image(model, model_id, source):
    return result_from_raw(model_id, raw_outputs(model, PREPROCESSORS[model_id](source)))


def result_from_raw(model_id, raw):
    scores = presentation_scores(raw)
    index = int(np.argmax(raw))
    return {"model": model_id, "raw_activations": raw.tolist(), "class_index": index,
            "prediction": {"style": CLASSES[index], "confidence": float(scores[index])},
            "probabilities": [{"style": style, "probability": float(value)} for style, value in zip(CLASSES, scores)]}
