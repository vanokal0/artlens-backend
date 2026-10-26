"""Images and dataset: classes, versioned preprocessing, the canonical split loader and
bounded parallel preprocessing. Kept TensorFlow-free because spawn workers import it."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SHA256 = "80556bc7841f318918b87e9105904031e822fd56d791fb11e5777cb671f01fce"

CLASSES = ("Baroque", "Impressionism", "Cubism", "Pop Art", "Ukiyo-e")
DIRECTORY_TO_CLASS = {name.replace(" ", "-"): name for name in CLASSES}
CONFIG = {
    "version": "artlens-preprocessing-v1",
    "classes": list(CLASSES),
    "decode": {"exif_transpose": True, "mode": "RGB", "alpha_background": "white",
               "animated_frame": 0, "icc_transform": False},
    "simple_ann": {"grid": [16, 16], "luminance": [0.299, 0.587, 0.114],
                   "grid_edges": "floor(i * dimension / 16)",
                   "tiny_image": "bilinear upscale each dimension to at least 16",
                   "quantization": "floor(mean + 0.5), clip to [0,255]",
                   "cell_order": "row-major", "bit_order": "MSB first", "shape": [2048]},
    "cnn": {"shape": [128, 128, 3], "resize": "bilinear stretch",
            "normalization": "float32 / 255", "augmentation": False},
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def config_hash():
    return hashlib.sha256(json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------- preprocessing (shared by training and inference)

def decode_rgb(source: str | Path | BinaryIO) -> Image.Image:
    """Decode frame zero, orient via EXIF, composite alpha on white; no ICC conversion."""
    with Image.open(source) as raw:
        raw.seek(0)
        image = ImageOps.exif_transpose(raw)
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = image.convert("RGBA")
            image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
        return image.convert("RGB")


def ann_features(source: str | Path | BinaryIO) -> np.ndarray:
    image = decode_rgb(source)
    width, height = image.size
    if min(width, height) < 16:
        image = image.resize((max(width, 16), max(height, 16)), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.uint8)
    height, width = rgb.shape[:2]
    ys = np.arange(17) * height // 16
    xs = np.arange(17) * width // 16
    # Convert one cell at a time: large paintings need no full float64 RGB copy.
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    means = np.array([(rgb[ys[y]:ys[y+1], xs[x]:xs[x+1]] @ weights).mean()
                      for y in range(16) for x in range(16)])
    values = np.clip(np.floor(means + 0.5), 0, 255).astype(np.uint8)
    return np.unpackbits(values, bitorder="big").astype(np.float32)


def cnn_tensor(source: str | Path | BinaryIO) -> np.ndarray:
    image = decode_rgb(source).resize((128, 128), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / np.float32(255)


# ---------------------------------------------------------------- canonical split (never repartitions)

@dataclass(frozen=True)
class Sample:
    filename: str
    label: int
    identity: str
    sha256: str
    split: str
    width: int
    height: int


def load_manifest(root=ROOT):
    root = Path(root)
    manifest = root / "data/prepared/splits.csv"
    if digest(manifest) != SPLIT_SHA256:
        raise ValueError("Canonical splits.csv fingerprint changed; do not silently repartition")
    if json.loads((root / "data/prepared/preprocessing.json").read_text(encoding="utf-8")) != CONFIG:
        raise ValueError("Preprocessing contract differs from Phase 1")
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen = {name: set() for name in ("filename", "sha256", "pixel_sha256")}
    samples = []
    for row in rows:
        label = int(row["class_id"])
        if not 0 <= label < 5 or CLASSES[label] != row["style"]:
            raise ValueError(f"Invalid class mapping: {row['filename']}")
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Invalid split")
        path = (root / row["filename"]).resolve()
        if (root / "data/selected").resolve() not in path.parents:
            raise ValueError("Path outside source dataset")
        if DIRECTORY_TO_CLASS.get(path.parent.name) != row["style"]:
            raise ValueError("Directory/class mismatch")
        for key in seen:
            if row[key] in seen[key]:
                raise ValueError(f"Repeated identity/leakage: {key}={row[key]}")
            seen[key].add(row[key])
        samples.append(Sample(row["filename"], label, row["pixel_sha256"], row["sha256"],
                              row["split"], int(row["width"]), int(row["height"])))
    return samples


def select_samples(samples, split, size=None, seed=42):
    """Deterministic class-balanced subset; size must be a multiple of five.

    No subset means the complete canonical partition, preserving manifest order.
    Test use is intentionally refused by this development loader.
    """
    if split not in {"train", "validation"}:
        raise ValueError("Phase 2A permits train/validation only")
    selected = [sample for sample in samples if sample.split == split]
    if size is None:
        return selected
    if size < 5 or size % 5:
        raise ValueError("Subset size must be a positive multiple of five")
    result = []
    for label in range(5):
        group = [s for s in selected if s.label == label]
        group.sort(key=lambda s: hashlib.sha256(f"{seed}:{s.identity}".encode()).hexdigest())
        if size // 5 > len(group):
            raise ValueError(f"Insufficient {CLASSES[label]} samples for balanced subset")
        result.extend(group[:size // 5])
    return sorted(result, key=lambda s: s.filename)


# ---------------------------------------------------------------- bounded Windows-spawn parallel preprocessing

def preprocess_worker(task):
    root, sample, model, verify_hash = task
    try:
        path = Path(root) / sample.filename
        if verify_hash and digest(path) != sample.sha256:
            raise ValueError("Source image checksum changed")
        function = {"simple_ann": ann_features, "cnn": cnn_tensor}[model]
        return sample, function(path)
    except Exception as exc:
        raise RuntimeError(f"Preprocessing failed for {sample.filename} ({sample.identity}): {exc}") from exc


def iter_preprocessed(samples, root, model="simple_ann", workers=1, max_in_flight=None, verify_hash=True):
    """Yield (Sample, array) in input order. At most 2*workers tasks by default.

    Paths/metadata go to workers, only compact processed arrays come back. Closing
    the generator cancels pending tasks and waits for active workers to finish.
    """
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4 on this machine")
    if model not in {"simple_ann", "cnn"}:
        raise ValueError("Unknown model")
    bound = max_in_flight if max_in_flight is not None else 2 * workers
    if bound < 1 or bound > 8:
        raise ValueError("max_in_flight must be between 1 and 8")
    tasks = ((str(root), sample, model, verify_hash) for sample in samples)
    if workers == 1:
        for task in tasks:
            yield preprocess_worker(task)
        return
    # Applied before spawn, so native libraries in children initialize one thread.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    pending = deque()
    try:
        for _ in range(bound):
            task = next(tasks, None)
            if task is None:
                break
            pending.append(executor.submit(preprocess_worker, task))
        while pending:
            result = pending.popleft().result()
            yield result
            task = next(tasks, None)
            if task is not None:
                pending.append(executor.submit(preprocess_worker, task))
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
