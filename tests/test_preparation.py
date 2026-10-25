import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from artlens.prepare import assign_splits, prepare
from artlens.preprocessing import CLASSES, DIRECTORY_TO_CLASS, ann_features, cnn_tensor, decode_rgb


def png(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)
    return stream


class PreprocessingTests(unittest.TestCase):
    def test_known_luminance_and_bit_order(self):
        # Red: round(0.299 * 255) = 76 = 01001100.
        features = ann_features(png(Image.new("RGB", (32, 48), (255, 0, 0))))
        np.testing.assert_array_equal(features, np.tile([0, 1, 0, 0, 1, 1, 0, 0], 256))
        self.assertEqual(features.dtype, np.float32)

    def test_grid_order_and_unequal_cells(self):
        image = np.zeros((33, 35, 3), dtype=np.uint8)
        for y in range(16):
            for x in range(16):
                image[y * 33 // 16:(y + 1) * 33 // 16,
                      x * 35 // 16:(x + 1) * 35 // 16] = y * 16 + x
        bits = ann_features(png(Image.fromarray(image)))
        np.testing.assert_array_equal(np.packbits(bits.astype(np.uint8)), np.arange(256))

    def test_modes_tiny_images_and_repeatability(self):
        for mode, color in (("L", 128), ("RGBA", (255, 0, 0, 0)), ("RGB", (1, 2, 3))):
            image = Image.new(mode, (1, 3), color)
            for function, shape in ((ann_features, (2048,)), (cnn_tensor, (128, 128, 3))):
                a, b = function(png(image)), function(png(image))
                self.assertEqual(a.shape, shape)
                self.assertTrue(np.isfinite(a).all())
                self.assertTrue(((a >= 0) & (a <= 1)).all())
                np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(cnn_tensor(png(Image.new("RGBA", (2, 2), (0, 0, 0, 0)))), 1)

    def test_rgb_channels(self):
        value = cnn_tensor(png(Image.new("RGB", (20, 40), (255, 128, 0))))
        np.testing.assert_allclose(value[0, 0], [1, 128 / 255, 0])
        self.assertEqual(value.dtype, np.float32)

    def test_exif_orientation(self):
        image = Image.new("RGB", (20, 30))
        exif = Image.Exif()
        exif[274] = 6
        stream = io.BytesIO()
        image.save(stream, format="JPEG", exif=exif)
        stream.seek(0)
        self.assertEqual(decode_rgb(stream).size, (30, 20))

    def test_bad_image_rejected(self):
        with self.assertRaises(OSError):
            cnn_tensor(io.BytesIO(b"invalid image"))

    def test_class_mapping(self):
        self.assertEqual(CLASSES, ("Baroque", "Impressionism", "Cubism", "Pop Art", "Ukiyo-e"))
        self.assertEqual(DIRECTORY_TO_CLASS["Pop-Art"], "Pop Art")


class DatasetTests(unittest.TestCase):
    def test_split_ratios_and_order_independence(self):
        rows = [{"filename": f"{label}/{i}", "style": label, "pixel_sha256": f"{label}:{i}"}
                for label in CLASSES for i in range(1000)]
        split = assign_splits(rows, 42)
        self.assertEqual(split, assign_splits(list(reversed(rows)), 42))
        self.assertNotEqual(split, assign_splits(rows, 43))
        for label in CLASSES:
            for name, count in (("train", 700), ("validation", 150), ("test", 150)):
                self.assertEqual(sum(r["style"] == label and r["split"] == name for r in split), count)

    def test_audit_excludes_problems_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selected = root / "data/selected"
            rows = []
            for i, directory in enumerate(DIRECTORY_TO_CLASS):
                folder = selected / directory
                folder.mkdir(parents=True)
                for j in range(10):
                    path = folder / f"{j}.png"
                    Image.new("RGB", (16, 16), (i * 40, j * 20, 0)).save(path)
                    rows.append({"filename": path.relative_to(root).as_posix(), "style": directory})
            duplicate = selected / "Baroque/duplicate.png"
            duplicate.write_bytes((selected / "Baroque/0.png").read_bytes())
            rows.extend([{"filename": "data/selected/Baroque/duplicate.png", "style": "Baroque"},
                         dict(rows[0]), {"filename": "data/selected/Baroque/missing.png", "style": "Baroque"}])
            bad = selected / "Baroque/bad.png"
            bad.write_bytes(b"bad")
            rows.append({"filename": "data/selected/Baroque/bad.png", "style": "Baroque"})
            conflict = selected / "Cubism/conflict.png"
            conflict.write_bytes((selected / "Baroque/1.png").read_bytes())
            rows.append({"filename": "data/selected/Cubism/conflict.png", "style": "Cubism"})
            with (selected / "manifest.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["filename", "style"])
                writer.writeheader()
                writer.writerows(rows)
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(root, root / "prepared")
            report = json.loads((root / "prepared/audit.json").read_text())
            self.assertEqual(report["eligible_unique_images"], 49)
            self.assertEqual(report["issue_counts"]["duplicate_record"], 1)
            self.assertEqual(report["issue_counts"]["missing_file"], 1)
            self.assertEqual(report["issue_counts"]["duplicate_pixels"], 1)
            self.assertEqual(report["issue_counts"]["conflicting_labels"], 2)
            self.assertTrue(duplicate.exists())
            self.assertTrue(bad.exists())


if __name__ == "__main__":
    unittest.main()
