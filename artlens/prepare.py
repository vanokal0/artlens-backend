"""Read-only source audit and reproducible shared split: python -m artlens.prepare."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import unicodedata

from PIL import Image

from artlens.preprocessing import CLASSES, CONFIG, DIRECTORY_TO_CLASS, decode_rgb


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.root, args.output, args.seed)
