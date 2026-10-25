"""Active model versions: validation, activation and persistence across restarts."""
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from artlens.data import ROOT
from artlens.inference import PREPROCESSORS, load_artifact, raw_outputs

MODEL_IDS = ("simple_ann", "cnn")
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
