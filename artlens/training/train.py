"""Offline training CLI; --smoke forces at most 2 epochs and small balanced subsets."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np

from artlens.data import ROOT, SPLIT_SHA256, config_hash, load_manifest, select_samples
from artlens.prepare import digest
from artlens.preprocessing import CLASSES, CONFIG


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    from artlens.training.datasets import ann_matrix, make_dataset
    train_features = validation_features = None
    if args.model == "simple_ann":
        train_features = ann_matrix(train_samples, ROOT, args.workers, args.cache)
        validation_features = ann_matrix(validation_samples, ROOT, args.workers, args.cache)
    from artlens.runtime import configure, environment
    tf = configure(args.seed, args.threads)
    from artlens.models import build_model
    from artlens.evaluation import classification_metrics, learning_curves, latency
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
    from artlens.inference import load_artifact
    loaded, _ = load_artifact(args.output)
    np.testing.assert_allclose(np.asarray(best(example, training=False)), np.asarray(loaded(example, training=False)), rtol=1e-5, atol=1e-6)
    write_json(args.output / "SUCCESS.json", {"reload_verified": True, "weights_updated": updated, "smoke_only": args.smoke})
    print(json.dumps({"output": str(args.output), "selected_epoch": selected_epoch, "smoke_only": args.smoke, "training_seconds": duration}))


if __name__ == "__main__":
    main()
