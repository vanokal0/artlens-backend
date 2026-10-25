"""Five-class metrics, learning curves, and separately measured latencies."""
from pathlib import Path
import time

import numpy as np

from artlens.preprocessing import CLASSES, ann_features, cnn_tensor


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
