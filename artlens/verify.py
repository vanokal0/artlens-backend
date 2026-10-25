"""Verify split invariants and real-image preprocessing without TensorFlow."""
import csv
import io
import json
from pathlib import Path

import numpy as np

from artlens.prepare import assign_splits, digest
from artlens.preprocessing import CLASSES, ann_features, cnn_tensor


def main():
    root = Path(__file__).resolve().parents[1]
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


if __name__ == "__main__":
    main()
