from collections import Counter
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from artlens.data import ROOT, Sample, ann_features, cnn_tensor, digest, iter_preprocessed, load_manifest, select_samples
from artlens.models import presentation_scores
from artlens.training import classification_metrics


class MathematicalTests(unittest.TestCase):
    def test_all_luminance_coefficients(self):
        for rgb, expected in (((255, 0, 0), 76), ((0, 255, 0), 150),
                              ((0, 0, 255), 29), ((10, 100, 200), 84)):
            buffer = io.BytesIO()
            Image.new("RGB", (16, 16), rgb).save(buffer, format="PNG")
            buffer.seek(0)
            bits = ann_features(buffer)
            np.testing.assert_array_equal(np.packbits(bits.astype(np.uint8)), np.full(256, expected))

    def test_arithmetic_mean_not_median_and_rounding(self):
        # Each 2x2 cell contains [0,0,0,255]: mean 63.75 rounds to 64 (01000000).
        pixels = np.zeros((32, 32, 3), dtype=np.uint8)
        pixels[1::2, 1::2] = 255
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        buffer.seek(0)
        bits = ann_features(buffer)
        np.testing.assert_array_equal(np.packbits(bits.astype(np.uint8)), np.full(256, 64))
        self.assertEqual(bits.shape, (2048,))

    def test_sigmoid_presentation_preserves_ranking(self):
        raw = np.array([[0.2, 0.8, 0.1, 0.4, 0.3], [0, 0, 0, 0, 0]])
        original = raw.copy()
        scores = presentation_scores(raw)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_allclose(scores.sum(1), 1)
        np.testing.assert_array_equal(np.argsort(raw), np.argsort(scores))
        np.testing.assert_array_equal(np.argmax(raw, axis=1), np.argmax(scores, axis=1))
        with self.assertRaises(ValueError):
            presentation_scores([0, 0, np.nan, 0, 0])

    def test_metrics_known_confusion(self):
        labels = np.array([0, 1, 2, 3, 4, 0])
        raw = np.eye(5)[[0, 1, 2, 3, 4, 1]]
        result = classification_metrics(labels, raw)
        self.assertAlmostEqual(result["accuracy"], 5 / 6)
        self.assertAlmostEqual(result["macro_precision"], 0.9)
        self.assertAlmostEqual(result["macro_recall"], 0.9)
        self.assertAlmostEqual(result["macro_f1"], (3 + 4 / 3) / 5)
        self.assertEqual(result["confusion_matrix"][0][1], 1)


class ManifestTests(unittest.TestCase):
    def test_real_train_validation_paths_equal_upload_bytes(self):
        rows = load_manifest()
        for split in ("train", "validation"):
            for sample in select_samples(rows, split, 5):
                path = ROOT / sample.filename
                for function in (ann_features, cnn_tensor):
                    expected = function(path)
                    actual = function(io.BytesIO(path.read_bytes()))
                    np.testing.assert_array_equal(expected, actual)
                    self.assertTrue(np.isfinite(actual).all())

    def test_canonical_identity_mapping_and_subsets(self):
        rows = load_manifest()
        self.assertEqual(Counter(s.split for s in rows), {"train": 3487, "validation": 748, "test": 747})
        self.assertEqual(len({s.identity for s in rows}), 4982)
        self.assertTrue(all(s.label == 3 for s in rows if "/Pop-Art/" in s.filename))
        subset = select_samples(rows, "train", 50)
        self.assertEqual(subset, select_samples(list(reversed(rows)), "train", 50))
        self.assertEqual(Counter(s.label for s in subset), dict.fromkeys(range(5), 10))
        self.assertTrue(all(s.split == "train" for s in subset))
        with self.assertRaises(ValueError):
            select_samples(rows, "test")


class ParallelTests(unittest.TestCase):
    def test_cache_hit_and_source_drift(self):
        from artlens.training import ann_matrix
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "image.png"
            Image.new("RGB", (16, 16), "red").save(path)
            sample = Sample(path.name, 0, "identity", digest(path), "train", 16, 16)
            a = ann_matrix([sample], root, cache=root / "cache")
            b = ann_matrix([sample], root, cache=root / "cache")
            np.testing.assert_array_equal(a, b)
            Image.new("RGB", (16, 16), "blue").save(path)
            with self.assertRaisesRegex(ValueError, "Source changed"):
                ann_matrix([sample], root, cache=root / "cache")

    def test_spawn_equivalence_order_and_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = []
            for i in range(11):
                path = root / f"{i}.png"
                Image.new("RGB", (17 + i, 31), (i * 20, 12, 90)).save(path)
                samples.append(Sample(path.name, i % 5, str(i), digest(path), "train", 17 + i, 31))
            sequential = list(iter_preprocessed(samples, root))
            for workers in (2, 4):
                parallel = list(iter_preprocessed(samples, root, workers=workers, max_in_flight=3))
                self.assertEqual([s for s, _ in parallel], samples)
                np.testing.assert_array_equal([x for _, x in sequential], [x for _, x in parallel])
                broken = [replace(samples[0], filename="missing.png"), *samples[1:]]
                with self.assertRaisesRegex(RuntimeError, "missing.png"):
                    list(iter_preprocessed(broken, root, workers=workers))
            # Early consumer termination exercises cancellation and executor shutdown.
            stream = iter_preprocessed(samples, root, workers=2)
            next(stream)
            stream.close()


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from artlens.models import configure
        cls.tf = configure(42, 2)

    def test_ann_single_sigmoid_layer_and_analytical_forward(self):
        from artlens.models import build_simple_ann as build
        model = build()
        self.assertEqual(len(model.layers), 1)
        self.assertEqual(model.count_params(), 10245)
        self.assertEqual(model.layers[0].activation.__name__, "sigmoid")
        weights = np.zeros((2048, 5), dtype=np.float32)
        weights[0] = [1, 2, 3, 4, 5]
        bias = np.array([-1, 0, 1, -2, -3], dtype=np.float32)
        model.set_weights([weights, bias])
        x = np.zeros((1, 2048), dtype=np.float32)
        x[0, 0] = 1
        expected = 1 / (1 + np.exp(-(weights[0] + bias)))
        np.testing.assert_allclose(model(x).numpy()[0], expected, rtol=1e-6)

    def test_models_gradient_and_serialization(self):
        from artlens.models import build_model
        for name, shape in (("simple_ann", (2048,)), ("cnn", (128, 128, 3))):
            model = build_model(name)
            x = np.random.default_rng(42).integers(0, 2, (5,) + shape).astype(np.float32)
            target = np.eye(5, dtype=np.float32)
            before = [w.copy() for w in model.get_weights()]
            values = model.train_on_batch(x, target)
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(before, model.get_weights())))
            predictions = np.asarray(model(x, training=False))
            self.assertEqual(predictions.shape, (5, 5))
            self.assertTrue(np.isfinite(predictions).all())
            if name == "cnn":
                self.assertEqual(model.count_params(), 28069)
                np.testing.assert_allclose(predictions.sum(1), 1, atol=1e-6)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "model.keras"
                model.save(path)
                loaded = self.tf.keras.models.load_model(path, compile=False)
                np.testing.assert_allclose(predictions, loaded(x, training=False), rtol=1e-5, atol=1e-6)

    def test_dataset_training_guard_and_validation_order(self):
        from artlens.training import make_dataset
        samples = select_samples(load_manifest(), "validation", 10)
        features = np.zeros((10, 2048), dtype=np.float32)
        features[:, 0] = np.arange(10)
        ds = make_dataset(samples, ROOT, "simple_ann", 3, 42, False, features=features)
        for _ in range(2):
            values = list(ds.as_numpy_iterator())
            np.testing.assert_array_equal(np.concatenate([x for x, _ in values])[:, 0], np.arange(10))
            np.testing.assert_array_equal(np.concatenate([y for _, y in values]).argmax(1), [s.label for s in samples])
        with self.assertRaises(ValueError):
            make_dataset(samples, ROOT, "simple_ann", 3, 42, True, features=features)


if __name__ == "__main__":
    unittest.main()
