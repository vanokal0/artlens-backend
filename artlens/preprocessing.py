"""Versioned preprocessing shared by future training and inference callers."""
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps

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
