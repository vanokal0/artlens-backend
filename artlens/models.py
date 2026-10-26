"""Models: Keras architectures, TensorFlow runtime setup, inference and the active-version registry.

TensorFlow is imported lazily so importing this module stays cheap.
"""
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
from PIL import Image

from artlens.data import CLASSES, CONFIG, ROOT, ann_features, cnn_tensor, config_hash, digest

MODEL_IDS = ("simple_ann", "cnn")


# ---------------------------------------------------------------- architectures (standard serializable Keras layers)

def build_simple_ann(learning_rate=0.001, seed=42):
    """Paper section III, figures 3/4: one biased weighted layer with sigmoid outputs."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(2048,), dtype="float32"),
        tf.keras.layers.Dense(5, activation="sigmoid", use_bias=True,
                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                              bias_initializer="zeros"),
    ], name="simple_ann")
    # BCE, SGD and initialization are ArtLens conventions, not paper specifications.
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])
    return model


def build_cnn(learning_rate=0.001, seed=42):
    """Independent RGB CNN baseline trained from scratch."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    layers = [tf.keras.Input(shape=(128, 128, 3), dtype="float32")]
    for filters in (16, 32, 64):
        layers.extend([tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"),
                       tf.keras.layers.MaxPooling2D(2)])
    layers.extend([tf.keras.layers.GlobalAveragePooling2D(),
                   tf.keras.layers.Dense(64, activation="relu"),
                   tf.keras.layers.Dropout(0.30, seed=seed),
                   tf.keras.layers.Dense(5, activation="softmax")])
    model = tf.keras.Sequential(layers, name="cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])
    return model


def build_model(name, learning_rate=0.001, seed=42):
    builders = {"simple_ann": build_simple_ann, "cnn": build_cnn}
    if name not in builders:
        raise ValueError(f"Unknown model: {name}")
    return builders[name](learning_rate, seed)


# ---------------------------------------------------------------- runtime (call before model/tensor creation)

def configure(seed=42, threads=4):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return tf


def environment():
    import tensorflow as tf
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("tensorflow", "keras", "numpy", "Pillow", "matplotlib")},
            "devices": [str(d) for d in tf.config.get_visible_devices()],
            "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
            "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
            "oneDNN": os.environ.get("TF_ENABLE_ONEDNN_OPTS")}


# ---------------------------------------------------------------- inference (raw predictions + presentation adapter)

PREPROCESSORS = {"simple_ann": ann_features, "cnn": cnn_tensor}


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


# ---------------------------------------------------------------- active versions: validation, activation, persistence

BASELINES = {"simple_ann": ROOT / "artifacts/ann-full-seed42", "cnn": ROOT / "artifacts/cnn-full-seed42"}
STATE_FILE = Path(os.environ.get("ARTLENS_STATE_FILE", ROOT / "artifacts/active-models.json"))


def _blank_png():
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(stream, "PNG")
    stream.seek(0)
    return stream


def validate_artifact(directory, model_id):
    """Load a complete, non-smoke artifact and check one finite five-class prediction.

    The prediction also warms the model, so the first user request is not slower.
    """
    directory = Path(directory)
    if not (directory / "SUCCESS.json").is_file():
        raise ValueError(f"{directory.name} is not a completed training run")
    model, metadata = load_artifact(directory)
    if metadata["model_id"] != model_id:
        raise ValueError(f"{directory.name} contains {metadata['model_id']}, expected {model_id}")
    if metadata["training"].get("smoke") or metadata["training"].get("train_size") is not None:
        raise ValueError(f"{directory.name} was not trained on the full training partition")
    raw = raw_outputs(model, PREPROCESSORS[model_id](_blank_png()))
    if raw.shape != (5,) or not np.isfinite(raw).all():
        raise ValueError(f"{directory.name} produced invalid outputs")
    return model, describe(directory, metadata)


def describe(directory, metadata):
    training = metadata["training"]
    metrics = json.loads((directory / "validation-metrics.json").read_text(encoding="utf-8"))
    trained_at = datetime.fromtimestamp((directory / "SUCCESS.json").stat().st_mtime, timezone.utc)
    return {"version": directory.name, "ready": True,
            "baseline": directory.resolve() in {p.resolve() for p in BASELINES.values()},
            "trainedAt": trained_at.isoformat(timespec="seconds"),
            "selectedEpoch": metadata["selected_checkpoint_epoch"],
            "validationAccuracy": metrics["accuracy"],
            "settings": {"epochs": training["epochs"], "learning_rate": training["learning_rate"],
                         "batch_size": training["batch_size"], "random_seed": training["seed"],
                         "data_augmentation": bool(training.get("augment", False))}}


def _stored_path(directory):
    directory = Path(directory).resolve()
    try:
        return directory.relative_to(ROOT).as_posix()
    except ValueError:
        return str(directory)


def save_state(directories):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps({m: _stored_path(d) for m, d in directories.items()}, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(STATE_FILE)  # atomic: a crash never leaves a half-written state file


def load_active():
    """Return {model_id: (model, info, directory)} for the persisted active versions.

    Missing or invalid entries fall back to the full baseline artifact.
    """
    stored = json.loads(STATE_FILE.read_text(encoding="utf-8")) if STATE_FILE.is_file() else {}
    active = {}
    for model_id in MODEL_IDS:
        directory = Path(stored.get(model_id, BASELINES[model_id]))
        directory = directory if directory.is_absolute() else ROOT / directory
        try:
            model, info = validate_artifact(directory, model_id)
        except Exception as exc:
            if directory.resolve() == BASELINES[model_id].resolve():
                raise
            print(f"Active {model_id} version {directory} is unusable ({exc}); using baseline", flush=True)
            directory = BASELINES[model_id]
            model, info = validate_artifact(directory, model_id)
        active[model_id] = (model, info, directory)
    save_state({m: entry[2] for m, entry in active.items()})
    return active
