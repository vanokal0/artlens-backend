"""Training: batch datasets, evaluation, the offline training CLI and the optional training job manager.

CLI:  python -m artlens.training --model {simple_ann,cnn} --output DIR [...]
      --smoke forces at most 2 epochs and small balanced subsets.

The API's TrainingManager runs this same CLI as a separate cancellable process writing to
a new run directory, so active artifacts are never touched. Progress is read from the
CLI's Keras log and training.jsonl records.
"""
import argparse
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

import numpy as np

from artlens.data import (CLASSES, CONFIG, ROOT, SPLIT_SHA256, ann_features, cnn_tensor, config_hash, digest,
                          iter_preprocessed, load_manifest, select_samples)
from artlens.models import build_model, configure, environment, load_artifact


# ---------------------------------------------------------------- bounded deterministic batches (only ANN features cached)

def ann_matrix(samples, root, workers=1, cache=None):
    metadata = {"preprocessing_hash": config_hash(),
                "samples": [(s.filename, s.sha256, s.label) for s in samples]}
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    path = Path(cache) / (key + ".npy") if cache else None
    # Validate source bytes even for cache hits, so stale data cannot be concealed.
    if path and path.exists():
        for sample in samples:
            if digest(Path(root) / sample.filename) != sample.sha256:
                raise ValueError(f"Source changed: {sample.filename}")
        result = np.load(path, allow_pickle=False)
        if result.shape != (len(samples), 2048) or not np.isin(result, (0, 1)).all():
            raise ValueError("Invalid cached ANN features")
        return result.astype(np.float32)
    result = np.stack([value for _, value in iter_preprocessed(samples, root, workers=workers)])
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.save(stream, result.astype(np.uint8), allow_pickle=False)
        temporary.replace(path)
    return result


def make_dataset(samples, root, model, batch_size, seed, training, workers=1, features=None, augment=False):
    import tensorflow as tf
    if batch_size < 1 or not samples:
        raise ValueError("Positive batch size and nonempty samples required")
    if augment and (model != "cnn" or not training):
        raise ValueError("Augmentation is only supported for CNN training batches")
    expected = "train" if training else "validation"
    if any(s.split != expected for s in samples):
        raise ValueError(f"Dataset contains samples outside {expected}")
    epoch = 0

    def batches():
        nonlocal epoch
        indices = np.arange(len(samples))
        # Seeded per epoch: random horizontal flips of CNN training images only.
        flips = np.random.default_rng([seed, epoch, 1]) if augment else None
        if training:
            np.random.default_rng(seed + epoch).shuffle(indices)
            epoch += 1
        ordered = [samples[i] for i in indices]

        def batch(xs, labels):
            xs = np.stack(xs)
            if flips is not None:
                mask = flips.random(len(xs)) < 0.5
                xs[mask] = xs[mask][:, :, ::-1]
            return xs, np.eye(5, dtype=np.float32)[labels]

        if features is not None:
            for offset in range(0, len(indices), batch_size):
                selection = indices[offset:offset + batch_size]
                yield features[selection], np.eye(5, dtype=np.float32)[[samples[i].label for i in selection]]
        else:
            with closing(iter_preprocessed(ordered, root, model=model, workers=workers)) as results:
                xs, labels = [], []
                for sample, tensor in results:
                    xs.append(tensor)
                    labels.append(sample.label)
                    if len(xs) == batch_size:
                        yield batch(xs, labels)
                        xs, labels = [], []
                if xs:
                    yield batch(xs, labels)

    shape = (2048,) if model == "simple_ann" else (128, 128, 3)
    dataset = tf.data.Dataset.from_generator(batches, output_signature=(
        tf.TensorSpec((None,) + shape, tf.float32), tf.TensorSpec((None, 5), tf.float32)))
    dataset = dataset.apply(tf.data.experimental.assert_cardinality((len(samples) + batch_size - 1) // batch_size))
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.with_options(options).prefetch(1)


# ---------------------------------------------------------------- evaluation: five-class metrics, curves, latency

def classification_metrics(labels, raw):
    labels, raw = np.asarray(labels), np.asarray(raw)
    if raw.shape != (len(labels), 5) or not len(labels) or not np.isfinite(raw).all():
        raise ValueError("Expected nonempty finite N by 5 predictions")
    if not np.issubdtype(labels.dtype, np.integer) or ((labels < 0) | (labels > 4)).any():
        raise ValueError("Invalid class indices")
    predicted = np.argmax(raw, axis=1)
    matrix = np.zeros((5, 5), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    tp = np.diag(matrix)
    precision = np.divide(tp, matrix.sum(0), out=np.zeros(5), where=matrix.sum(0) > 0)
    recall = np.divide(tp, matrix.sum(1), out=np.zeros(5), where=matrix.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=precision + recall > 0)
    return {"classes": list(CLASSES), "accuracy": float(tp.sum() / len(labels)),
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()), "confusion_matrix": matrix.tolist(),
            "matrix_axes": "rows=true, columns=predicted", "zero_division": 0,
            "per_class": [{"style": c, "precision": float(p), "recall": float(r), "f1": float(f),
                           "support": int(n)} for c, p, r, f, n in zip(CLASSES, precision, recall, f1, matrix.sum(1))]}


def learning_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for axis, name in zip(axes, ("loss", "accuracy")):
        epochs = range(1, len(history[name]) + 1)
        axis.plot(epochs, history[name], label="train")
        axis.plot(epochs, history["val_" + name], label="validation")
        axis.set(xlabel="Epoch", ylabel=name, title="Smoke run" if len(history[name]) <= 2 else name)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def latency(model, model_id, samples, root):
    if not samples or any(s.split != "validation" for s in samples):
        raise ValueError("Phase 2A latency uses validation samples only")
    function = {"simple_ann": ann_features, "cnn": cnn_tensor}[model_id]
    warmup = function(Path(root) / samples[0].filename)[None]
    np.asarray(model(warmup, training=False))
    preprocess, infer = [], []
    for sample in samples:
        start = time.perf_counter()
        tensor = function(Path(root) / sample.filename)[None]
        preprocess.append(time.perf_counter() - start)
        start = time.perf_counter()
        np.asarray(model(tensor, training=False))  # materialize/synchronize output
        infer.append(time.perf_counter() - start)
    return {"images": len(samples), "batch_size": 1, "warmup_calls": 1,
            "preprocessing_seconds": preprocess, "inference_seconds": infer,
            "preprocessing_mean_ms": 1000 * float(np.mean(preprocess)),
            "inference_mean_ms": 1000 * float(np.mean(infer))}


# ---------------------------------------------------------------- offline training CLI

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Offline ArtLens training; --smoke forces at most 2 epochs and small balanced subsets.")
    parser.add_argument("--model", choices=("simple_ann", "cnn"), required=True)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--augment", action="store_true",
                        help="CNN only: seeded random horizontal flips of training images")
    args = parser.parse_args()
    if args.augment and args.model != "cnn":
        parser.error("--augment is only supported for the CNN; the research ANN is trained without augmentation")
    if args.batch_size < 1 or args.threads < 1 or not 1 <= args.workers <= 4 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Invalid batch size, threads, workers or learning rate")
    if args.epochs is None:
        args.epochs = 40 if args.model == "simple_ann" else 50
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.train_size = 100 if args.train_size is None else args.train_size
        args.validation_size = 50 if args.validation_size is None else args.validation_size
        if args.train_size > 200 or args.validation_size > 100:
            parser.error("Smoke limits: 200 train and 100 validation images")
    if args.output.exists():
        parser.error("Output must be new; refusing to overwrite an existing run")
    samples = load_manifest()
    train_samples = select_samples(samples, "train", args.train_size, args.seed)
    validation_samples = select_samples(samples, "validation", args.validation_size, args.seed)
    args.output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(args.output / "config.json", config)
    write_json(args.output / "samples.json", {"train": [asdict(s) for s in train_samples],
                                              "validation": [asdict(s) for s in validation_samples]})
    # ANN features are extracted by spawn workers before TensorFlow is imported.
    train_features = validation_features = None
    if args.model == "simple_ann":
        train_features = ann_matrix(train_samples, ROOT, args.workers, args.cache)
        validation_features = ann_matrix(validation_samples, ROOT, args.workers, args.cache)
    tf = configure(args.seed, args.threads)
    model = build_model(args.model, args.learning_rate, args.seed)
    before = [w.copy() for w in model.get_weights()]
    train = make_dataset(train_samples, ROOT, args.model, args.batch_size, args.seed, True, args.workers, train_features,
                         augment=args.augment)
    validation = make_dataset(validation_samples, ROOT, args.model, args.batch_size, args.seed, False, args.workers, validation_features)

    class StructuredLog(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            record = {"epoch": epoch + 1, **{k: float(v) for k, v in (logs or {}).items()}}
            with (args.output / "training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record), flush=True)

    callbacks = [tf.keras.callbacks.TerminateOnNaN(),
                 tf.keras.callbacks.ModelCheckpoint(args.output / "model.keras", monitor="val_loss", save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                 StructuredLog()]
    start = time.perf_counter()
    history = model.fit(train, validation_data=validation, epochs=args.epochs, callbacks=callbacks,
                        shuffle=False, verbose=2).history
    duration = time.perf_counter() - start
    if not all(np.isfinite(value).all() for value in history.values()):
        raise RuntimeError("Non-finite training history; run not successful")
    updated = any(not np.array_equal(a, b) for a, b in zip(before, model.get_weights()))
    if not updated:
        raise RuntimeError("Training did not update model weights")
    selected_epoch = int(np.argmin(history["val_loss"])) + 1
    # Explicitly load the best file, independent of callback restore behavior.
    best = tf.keras.models.load_model(args.output / "model.keras", compile=False)
    example = next(iter(validation))[0]
    np.testing.assert_allclose(np.asarray(model(example, training=False)), np.asarray(best(example, training=False)), rtol=1e-5, atol=1e-6)
    raw, labels = [], []
    for x, y in validation:
        raw.append(np.asarray(best(x, training=False)))
        labels.extend(np.argmax(y.numpy(), axis=1))
    raw = np.concatenate(raw)
    metrics = classification_metrics(labels, raw)
    metrics.update({"partition": "validation", "smoke_only": args.smoke, "images": len(labels),
                    "decision_rule": "argmax raw model outputs; first index breaks ties"})
    write_json(args.output / "history.json", history)
    write_json(args.output / "validation-metrics.json", metrics)
    write_json(args.output / "latency.json", latency(best, args.model, validation_samples[:10], ROOT))
    learning_curves(history, args.output / "learning-curves.png")
    source_hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted((ROOT / "artlens").rglob("*.py"))}
    metadata = {"schema_version": 1, "model_id": args.model, "classes": list(CLASSES),
                "architecture": json.loads(best.to_json()), "parameters": best.count_params(),
                "input_shape": list(best.input_shape), "input_dtype": "float32", "preprocessing": CONFIG,
                "preprocessing_version": CONFIG["version"], "preprocessing_hash": config_hash(),
                "split_sha256": SPLIT_SHA256, "dataset_manifest_sha256": digest(ROOT / "data/selected/manifest.csv"),
                "training": config, "optimizer": model.optimizer.get_config(), "loss": model.loss.get_config(),
                "seed": args.seed, "selected_checkpoint_epoch": selected_epoch, "environment": environment(),
                "model_sha256": digest(args.output / "model.keras"), "source_sha256": source_hashes,
                "training_seconds": duration, "weights_updated": updated, "reload_verified": True,
                "evaluation": {"partition": "validation", "file": "validation-metrics.json", "smoke_only": args.smoke},
                "presentation": "sigmoid/sum(sigmoid), uncalibrated" if args.model == "simple_ann" else "softmax"}
    write_json(args.output / "metadata.json", metadata)
    shutil.copyfile(ROOT / "data/prepared/splits.csv", args.output / "splits.csv")
    write_json(args.output / "preprocessing.json", CONFIG)
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    (args.output / "requirements.lock.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    loaded, _ = load_artifact(args.output)
    np.testing.assert_allclose(np.asarray(best(example, training=False)), np.asarray(loaded(example, training=False)), rtol=1e-5, atol=1e-6)
    write_json(args.output / "SUCCESS.json", {"reload_verified": True, "weights_updated": updated, "smoke_only": args.smoke})
    print(json.dumps({"output": str(args.output), "selected_epoch": selected_epoch, "smoke_only": args.smoke, "training_seconds": duration}))


# ---------------------------------------------------------------- optional training jobs for the API (one at a time)

RUNS_DIR = Path(os.environ.get("ARTLENS_RUNS_DIR", ROOT / "artifacts/runs"))
ANN_CACHE = ROOT / "data/cache/ann"
ACTIVE_STATES = {"running", "cancelling"}


class TrainingBusy(Exception):
    pass


class TrainingManager:
    def __init__(self, activate):
        """`activate(model_id, directory)` must validate the artifact and switch to it, or raise."""
        self._activate = activate
        self._lock = threading.Lock()
        self._job = None

    @property
    def busy(self):
        with self._lock:
            return self._job is not None and self._job["status"] in ACTIVE_STATES

    def start(self, settings):
        with self._lock:
            if self._job is not None and self._job["status"] in ACTIVE_STATES:
                raise TrainingBusy("A training job is already running")
            model = settings["model"]
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            training_id = f"{model}-{stamp}-{uuid.uuid4().hex[:6]}"
            output = RUNS_DIR / training_id
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-m", "artlens.training", "--model", model,
                       "--epochs", str(settings["epochs"]), "--learning-rate", repr(settings["learning_rate"]),
                       "--batch-size", str(settings["batch_size"]), "--seed", str(settings["random_seed"]),
                       "--threads", "4", "--output", str(output)]
            # Same worker/cache choices as the established baseline commands.
            command += ["--workers", "2", "--cache", str(ANN_CACHE)] if model == "simple_ann" else ["--workers", "1"]
            if settings["data_augmentation"]:
                command.append("--augment")
            log_path = RUNS_DIR / f"{training_id}.log"
            log = log_path.open("wb")
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, "PYTHONUNBUFFERED": "1"}, creationflags=flags)
            log.close()
            self._job = {"trainingId": training_id, "model": model, "settings": dict(settings),
                         "status": "running", "stage": "Loading dataset", "error": None,
                         "started": time.monotonic(), "finished": None, "process": process,
                         "output": output, "log": log_path, "cancel": False}
            threading.Thread(target=self._monitor, args=(self._job,), daemon=True).start()
        return self.status()

    def cancel(self):
        with self._lock:
            job = self._job
            if job is None or job["status"] not in ACTIVE_STATES:
                raise TrainingBusy("No training job is running")
            if job["process"].poll() is not None:
                # Training already finished; validation/activation cannot be interrupted safely.
                raise TrainingBusy("Training already finished and the new model is being activated")
            job["cancel"] = True
            job["status"] = "cancelling"
            job["stage"] = "Stopping training process"
        _kill_tree(job["process"])
        return self.status()

    def shutdown(self):
        with self._lock:
            job = self._job
        if job is not None and job["process"].poll() is None:
            job["cancel"] = True
            _kill_tree(job["process"], job["output"])
            job["process"].wait()
            shutil.rmtree(job["output"], ignore_errors=True)

    def status(self):
        with self._lock:
            job = dict(self._job) if self._job is not None else None
        if job is None:
            return {"status": "idle", "trainingId": None, "model": None, "settings": None, "stage": None,
                    "currentEpoch": 0, "totalEpochs": 0, "elapsedMs": 0, "metrics": None, "error": None}
        epochs = _epoch_records(job["output"])
        stage = job["stage"]
        if job["status"] == "running" and job["process"].poll() is None:
            started = _started_epoch(job["log"])
            if (job["output"] / "history.json").exists():
                stage = "Evaluating on validation set"
            elif started:
                stage = f"Training epoch {started} of {job['settings']['epochs']}"
            elif job["model"] == "simple_ann":
                stage = "Loading dataset and extracting 16 × 16 grid features"
        end = job["finished"] or time.monotonic()
        return {"status": job["status"], "trainingId": job["trainingId"], "model": job["model"],
                "settings": job["settings"], "stage": stage, "currentEpoch": len(epochs),
                "totalEpochs": job["settings"]["epochs"], "elapsedMs": round((end - job["started"]) * 1000),
                "metrics": epochs[-1] if epochs else None, "error": job["error"]}

    def _update(self, job, **fields):
        with self._lock:
            job.update(fields)

    def _monitor(self, job):
        code = job["process"].wait()
        if job["cancel"]:
            _kill_tree(job["process"], job["output"])
            shutil.rmtree(job["output"], ignore_errors=True)
            return self._update(job, status="cancelled", stage="Cancelled; previous model kept",
                                finished=time.monotonic())
        if code != 0:
            return self._update(job, status="failed", stage="Failed; previous model kept",
                                error=_failure_message(job["log"], code), finished=time.monotonic())
        self._update(job, stage="Validating and activating new model")
        try:
            self._activate(job["model"], job["output"])
        except Exception as exc:
            return self._update(job, status="failed", stage="Failed; previous model kept",
                                error=f"New model failed validation: {exc}", finished=time.monotonic())
        self._update(job, status="completed", stage="Completed; new model active", finished=time.monotonic())


def _kill_tree(process, output=None):
    """Stop the trainer and its spawned preprocessing workers."""
    if os.name != "nt":
        process.kill()
        return
    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    if output is None:
        return
    # The venv python.exe is a launcher; its real interpreter child can survive /T. Sweep any
    # process still referencing this run's unique output directory and kill it by PID.
    for _ in range(20):
        pids = _pids_referencing(output)
        if not pids:
            return
        for pid in pids:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        time.sleep(0.5)


def _pids_referencing(output):
    needle = str(Path(output).name).replace("'", "''")
    script = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              f"Where-Object {{ $_.CommandLine -like '*{needle}*' }} | "
              "ForEach-Object { $_.ProcessId }")
    result = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True)
    return [int(line) for line in result.stdout.split() if line.isdigit() and int(line) != os.getpid()]


def _epoch_records(output):
    path = Path(output) / "training.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # line still being written
            continue
        records.append({"epoch": record["epoch"], "loss": record.get("loss"), "accuracy": record.get("accuracy"),
                        "val_loss": record.get("val_loss"), "val_accuracy": record.get("val_accuracy")})
    return records


def _started_epoch(log):
    matches = re.findall(r"^Epoch (\d+)/\d+", _read(log), flags=re.MULTILINE)
    return int(matches[-1]) if matches else 0


def _failure_message(log, code):
    lines = [line.strip() for line in _read(log).splitlines() if line.strip()]
    errors = [line for line in lines if re.match(r"^[\w.]*(Error|Exception)\b|^[\w.]+\.py: error|^usage:", line)
              or ": error:" in line]
    return errors[-1] if errors else f"Training process exited with code {code}"


def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


if __name__ == "__main__":
    main()
