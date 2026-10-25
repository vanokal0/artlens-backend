"""Uncached ANN feature computation benchmark: python -m artlens.benchmark."""
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

from artlens.data import ROOT, SPLIT_SHA256, config_hash, load_manifest, select_samples
from artlens.parallel import iter_preprocessed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/parallel-benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    samples = select_samples(load_manifest(), "train", args.size, args.seed)
    expected = None
    timings = {w: [] for w in (1, 2, 4)}
    # Warm up every execution mode. Each measurement still starts its own pool.
    for workers in timings:
        list(iter_preprocessed(samples[:5], ROOT, workers=workers))
    for repeat in range(args.repeats):
        # Reverse order on alternate repeats to reduce a fixed cache/order bias.
        order = (1, 2, 4) if repeat % 2 == 0 else (4, 2, 1)
        for workers in order:
            start = time.perf_counter()
            result = list(iter_preprocessed(samples, ROOT, workers=workers))
            elapsed = time.perf_counter() - start
            if [sample for sample, _ in result] != samples:
                raise AssertionError("Result ordering/identity/label mismatch")
            features = np.stack([value for _, value in result])
            if expected is None:
                expected = features
            np.testing.assert_array_equal(expected, features)
            timings[workers].append(elapsed)
            print(json.dumps({"workers": workers, "repeat": repeat + 1, "seconds": elapsed}), flush=True)
    baseline = statistics.median(timings[1])
    report = {"images": len(samples), "seed": args.seed, "split": "train", "split_sha256": SPLIT_SHA256,
              "preprocessing_hash": config_hash(), "samples": [asdict(s) for s in samples],
              "pixel_count_min": min(s.width * s.height for s in samples),
              "pixel_count_max": max(s.width * s.height for s in samples),
              "pixel_count_total": sum(s.width * s.height for s in samples),
              "warmup_images_per_mode": 5, "cache": "no feature cache; OS filesystem cache uncontrolled",
              "timing_scope": "process start/shutdown + file hash + decode + feature extraction + result collection",
              "equivalence": "exact arrays, labels, identity and order for every run",
              "python": sys.version, "platform": platform.platform(), "cpu": platform.processor(),
              "numpy": importlib.metadata.version("numpy"), "Pillow": importlib.metadata.version("Pillow"),
              "max_in_flight": "2 * workers", "child_native_threads": 1,
              "results": [{"workers": w, "wall_seconds": values, "median_seconds": statistics.median(values),
                           "images_per_second": len(samples) / statistics.median(values),
                           "speedup": baseline / statistics.median(values)} for w, values in timings.items()]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
