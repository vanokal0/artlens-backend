"""Bounded deterministic batch generators. Only the ANN feature matrix is cached."""
from contextlib import closing
import hashlib
import json
from pathlib import Path

import numpy as np

from artlens.data import config_hash
from artlens.parallel import iter_preprocessed
from artlens.prepare import digest


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
