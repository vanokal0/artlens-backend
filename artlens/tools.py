"""Offline dataset tools: python -m artlens.tools {prepare,verify,benchmark}.

prepare    read-only audit of data/selected and the reproducible train/validation/test split
verify     check split invariants and real-image preprocessing (no TensorFlow)
benchmark  uncached ANN feature extraction with 1/2/4 parallel workers
"""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import unicodedata

import numpy as np
from PIL import Image

from artlens.data import (CLASSES, CONFIG, DIRECTORY_TO_CLASS, ROOT, SPLIT_SHA256, ann_features, cnn_tensor,
                          config_hash, decode_rgb, digest, iter_preprocessed, load_manifest, select_samples)


# ---------------------------------------------------------------- prepare

def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def artist_key(value):
    value = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    return "" if value in {"", "nan", "unknown", "anonymous"} else value


def assign_splits(records, seed):
    """Deduplicated identities ranked by seeded SHA256, independently per class.

    Integer counts use largest remainders, ties train/validation/test order.
    Hash ranking is independent of CSV ordering and Python RNG implementation.
    """
    result = []
    for label in CLASSES:
        rows = [dict(row) for row in records if row["style"] == label]
        rows.sort(key=lambda r: (hashlib.sha256(f"{seed}:{r['pixel_sha256']}".encode()).hexdigest(), r["filename"]))
        n = len(rows)
        counts = [n * ratio // 100 for ratio in (70, 15, 15)]
        order = sorted(range(3), key=lambda i: (-(n * (70, 15, 15)[i] % 100), i))
        for i in order[:n - sum(counts)]:
            counts[i] += 1
        offset = 0
        for split, count in zip(("train", "validation", "test"), counts):
            for row in rows[offset:offset + count]:
                row["split"] = split
                result.append(row)
            offset += count
    return sorted(result, key=lambda row: row["filename"])


def prepare(root, output, seed=42):
    root, output = root.resolve(), output.resolve()
    selected = root / "data/selected"
    if output == selected or selected in output.parents:
        raise ValueError("Output must be outside the original dataset")
    manifest = selected / "manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if not {"filename", "style"}.issubset(columns or []):
            raise ValueError("Manifest requires filename and style")
        source = list(reader)
    valid, issues = [], []
    seen_rows, seen_paths = set(), set()
    referenced = set()
    for index, row in enumerate(source, 2):
        filename = row["filename"].replace("\\", "/")
        try:
            signature = tuple(row.get(column, "") for column in columns)
            if signature in seen_rows:
                raise ValueError("duplicate_record")
            seen_rows.add(signature)
            path = (root / filename).resolve()
            if selected not in path.parents:
                raise ValueError("path_outside_dataset")
            referenced.add(path)
            if path in seen_paths:
                raise ValueError("duplicate_path")
            seen_paths.add(path)
            style = DIRECTORY_TO_CLASS.get(row["style"], row["style"])
            if style not in CLASSES or DIRECTORY_TO_CLASS.get(path.parent.name) != style:
                raise ValueError("class_or_directory_mismatch")
            if not path.is_file():
                raise ValueError("missing_file")
            with Image.open(path) as image:
                fmt, mode = image.format, image.mode
                image.verify()
            rgb = decode_rgb(path)
            width, height = rgb.size
            pixel_hash = hashlib.sha256(f"RGB:{width}:{height}:".encode() + rgb.tobytes()).hexdigest()
            valid.append({"filename": path.relative_to(root).as_posix(), "style": style,
                          "class_id": CLASSES.index(style), "artist": row.get("artist", ""),
                          "artist_key": artist_key(row.get("artist", "")),
                          "artwork": row.get("artwork", ""), "source_url": row.get("source_url", ""),
                          "sha256": digest(path), "pixel_sha256": pixel_hash,
                          "width": width, "height": height, "format": fmt, "mode": mode,
                          "bytes": path.stat().st_size})
        except (ValueError, OSError, Image.DecompressionBombError) as exc:
            issues.append({"row": index, "filename": filename, "reason": str(exc)})
        if (index - 1) % 500 == 0:
            print(f"Audited {index - 1}/{len(source)}", flush=True)
    groups = defaultdict(list)
    for row in valid:
        groups[row["pixel_sha256"]].append(row)
    eligible, duplicate_groups, conflicts = [], [], []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda row: row["filename"])
        if len({row["style"] for row in group}) > 1:
            conflicts.append({"pixel_sha256": key, "files": [r["filename"] for r in group]})
            for row in group:
                issues.append({"row": "", "filename": row["filename"], "reason": "conflicting_labels"})
            continue
        eligible.append(group[0])
        if len(group) > 1:
            duplicate_groups.append({"retained": group[0]["filename"],
                                     "excluded": [r["filename"] for r in group[1:]]})
            for row in group[1:]:
                issues.append({"row": "", "filename": row["filename"], "reason": "duplicate_pixels"})
    assignments = assign_splits(eligible, seed)
    artist_sets = {s: {r["artist_key"] for r in assignments if r["split"] == s and r["artist_key"]}
                   for s in ("train", "validation", "test")}
    overlap = {f"{a}_{b}": sorted(artist_sets[a] & artist_sets[b])
               for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    actual_files = {p.resolve() for p in selected.rglob("*") if p.is_file() and p != manifest}
    report = {
        "manifest_sha256": digest(manifest), "manifest_columns": columns,
        "source_records": len(source), "source_class_counts": dict(Counter(r["style"] for r in source)),
        "validated_records": len(valid), "eligible_unique_images": len(eligible),
        "validated_class_counts": dict(Counter(r["style"] for r in valid)),
        "issue_counts": dict(Counter(r["reason"] for r in issues)),
        "duplicate_content_groups": duplicate_groups, "conflicting_content_groups": conflicts,
        "byte_duplicate_extra_records": sum(n - 1 for n in Counter(r["sha256"] for r in valid).values()),
        "unreferenced_files": sorted(p.relative_to(root).as_posix() for p in actual_files - referenced),
        "formats": dict(Counter(r["format"] for r in valid)), "modes": dict(Counter(r["mode"] for r in valid)),
        "dimensions": {axis: {"min": min((r[axis] for r in valid), default=0),
                               "max": max((r[axis] for r in valid), default=0)} for axis in ("width", "height")},
        "total_image_bytes": sum(r["bytes"] for r in valid),
        "seed": seed, "split_algorithm": "seeded-sha256-rank-v1-largest-remainders",
        "duplicate_policy": "retain lexicographically first path per decoded RGB identity; exclude conflicting labels",
        "split_counts": {s: dict(Counter(r["style"] for r in assignments if r["split"] == s)) for s in artist_sets},
        "artist_counts": {s: len(v) for s, v in artist_sets.items()},
        "artist_overlap_counts": {k: len(v) for k, v in overlap.items()}, "artist_overlap": overlap,
        "missing_artist_records": sum(not r["artist_key"] for r in assignments),
        "heldout_images_with_train_artist": {s: sum(r["artist_key"] in artist_sets["train"] for r in assignments if r["split"] == s)
                                              for s in ("validation", "test")},
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                        "numpy": importlib.metadata.version("numpy"), "Pillow": importlib.metadata.version("Pillow")},
    }
    if not eligible or set(r["style"] for r in eligible) != set(CLASSES):
        raise ValueError("Cannot create a five-class split: inspect source dataset")
    output.mkdir(parents=True, exist_ok=True)
    fields = list(valid[0])
    write_csv(output / "validated.csv", sorted(valid, key=lambda r: r["filename"]), fields)
    write_csv(output / "splits.csv", assignments, fields + ["split"])
    write_csv(output / "issues.csv", issues, ["row", "filename", "reason"])
    report["splits_sha256"] = digest(output / "splits.csv")
    for name, value in (("audit.json", report), ("preprocessing.json", CONFIG)):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("source_records", "validated_records", "eligible_unique_images", "issue_counts", "split_counts", "artist_overlap_counts", "splits_sha256")}, indent=2))


# ---------------------------------------------------------------- verify

def verify(root=ROOT):
    output = root / "data/prepared"
    report = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    with (output / "splits.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert digest(output / "splits.csv") == report["splits_sha256"]
    assert digest(root / "data/selected/manifest.csv") == report["manifest_sha256"]
    assert len({r["filename"] for r in rows}) == len(rows)
    assert len({r["pixel_sha256"] for r in rows}) == len(rows)
    assert len({r["sha256"] for r in rows}) == len(rows)
    assert assign_splits(rows, report["seed"]) == rows
    assert assign_splits(list(reversed(rows)), report["seed"]) == rows
    for row in rows:
        assert CLASSES[int(row["class_id"])] == row["style"]
    checked = 0
    for split in ("train", "validation", "test"):
        for style in CLASSES:
            subset = [r for r in rows if r["split"] == split and r["style"] == style]
            assert subset
            for row in subset[:2]:
                path = root / row["filename"]
                assert digest(path) == row["sha256"]
                for function, shape in ((ann_features, (2048,)), (cnn_tensor, (128, 128, 3))):
                    training = function(path)
                    inference = function(io.BytesIO(path.read_bytes()))
                    assert training.shape == shape and training.dtype == np.float32
                    assert np.isfinite(training).all()
                    assert ((training >= 0) & (training <= 1)).all()
                    np.testing.assert_array_equal(training, inference)
                checked += 1
    result = {"split_records_verified": len(rows), "real_images_preprocessed": checked,
              "path_vs_upload_bytes": "bit-identical", "split_reproduction": "passed",
              "ann_shape": [2048], "cnn_shape": [128, 128, 3], "finite_and_range_checks": "passed"}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------- benchmark

def benchmark(size=100, repeats=2, seed=42, output=Path("artifacts/parallel-benchmark.json")):
    samples = select_samples(load_manifest(), "train", size, seed)
    expected = None
    timings = {w: [] for w in (1, 2, 4)}
    # Warm up every execution mode. Each measurement still starts its own pool.
    for workers in timings:
        list(iter_preprocessed(samples[:5], ROOT, workers=workers))
    for repeat in range(repeats):
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
    report = {"images": len(samples), "seed": seed, "split": "train", "split_sha256": SPLIT_SHA256,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["results"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="audit data/selected and write data/prepared")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, default=Path("data/prepared"))
    p.add_argument("--seed", type=int, default=42)
    commands.add_parser("verify", help="verify data/prepared against the source dataset")
    b = commands.add_parser("benchmark", help="parallel ANN feature extraction benchmark")
    b.add_argument("--size", type=int, default=100)
    b.add_argument("--repeats", type=int, default=2)
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--output", type=Path, default=Path("artifacts/parallel-benchmark.json"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.output, args.seed)
    elif args.command == "verify":
        verify()
    else:
        if args.repeats < 1:
            parser.error("repeats must be positive")
        benchmark(args.size, args.repeats, args.seed, args.output)


if __name__ == "__main__":
    main()
