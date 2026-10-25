"""Read the canonical split. Never creates partitions or decodes held-out test data."""
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from artlens.prepare import digest
from artlens.preprocessing import CLASSES, CONFIG, DIRECTORY_TO_CLASS

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SHA256 = "80556bc7841f318918b87e9105904031e822fd56d791fb11e5777cb671f01fce"


def config_hash():
    return hashlib.sha256(json.dumps(CONFIG, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
