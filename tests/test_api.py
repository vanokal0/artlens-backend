"""End-to-end API contract tests with real validation-partition paintings.

Starts the real uvicorn server in a subprocess (TensorFlow runtime settings are
process-wide, so the service is kept out of the unittest process). Uses the full
artifacts unless ARTLENS_ANN_ARTIFACT/ARTLENS_CNN_ARTIFACT are set.
"""
import math
import socket
import subprocess
import sys
import time
import unittest

import httpx

from artlens.data import CLASSES, ROOT, load_manifest, select_samples

STATUS_KEYS = {"status", "progress", "stage", "elapsedMs", "result"}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ApiContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        port = free_port()
        cls.server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "artlens.api:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cls.client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30)
        deadline = time.monotonic() + 180
        while True:
            try:
                cls.client.get("/api/analyses/startup-check")
                break
            except httpx.TransportError:
                if cls.server.poll() is not None or time.monotonic() > deadline:
                    cls.server.kill()
                    raise RuntimeError("API server did not start")
                time.sleep(0.5)
        # One real validation painting per class; test-partition images are never used.
        cls.samples = select_samples(load_manifest(), "validation", 5, seed=42)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.server.terminate()
        cls.server.wait(timeout=30)

    def analyse(self, sample, model, result_type="single"):
        path = ROOT / sample.filename
        form = {"model": model, "result_type": result_type, "epochs": "20", "learning_rate": "0.001",
                "batch_size": "32", "random_seed": "42", "data_augmentation": "false"}
        with path.open("rb") as stream:
            response = self.client.post("/api/analyses", data=form,
                                        files={"image": (path.name, stream, "image/jpeg")})
        self.assertIn(response.status_code, (200, 201, 202))
        analysis_id = response.json()["analysisId"]
        self.assertIsInstance(analysis_id, str)
        stages = []
        deadline = time.monotonic() + 120
        while True:
            status = self.client.get(f"/api/analyses/{analysis_id}").json()
            self.assertTrue(STATUS_KEYS <= status.keys())
            self.assertIn(status["status"], ("processing", "completed", "failed"))
            self.assertTrue(0 <= status["progress"] <= 100)
            self.assertIsInstance(status["elapsedMs"], int)
            stages.append(status["stage"])
            if status["status"] != "processing":
                break
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.25)  # same pause as the frontend
        self.assertEqual(status["status"], "completed", status.get("error"))
        self.assertEqual(status["progress"], 100)
        return status["result"], stages

    def check_model_result(self, result, model):
        self.assertEqual(set(result), {"model", "prediction", "probabilities"})
        self.assertEqual(result["model"], model)
        probabilities = result["probabilities"]
        self.assertEqual([p["style"] for p in probabilities], list(CLASSES))
        values = [p["probability"] for p in probabilities]
        self.assertTrue(all(math.isfinite(v) and 0 <= v <= 1 for v in values))
        self.assertAlmostEqual(sum(values), 1.0, places=5)
        best = max(probabilities, key=lambda p: p["probability"])
        self.assertEqual(result["prediction"], {"style": best["style"], "confidence": best["probability"]})

    def test_single_models(self):
        for model in ("simple_ann", "cnn"):
            for sample in self.samples:
                with self.subTest(model=model, image=sample.filename):
                    result, stages = self.analyse(sample, model, "probabilities")
                    self.check_model_result(result, model)
                    self.assertEqual(stages[-1], "Completed")

    def test_comparison(self):
        for sample in self.samples:
            with self.subTest(image=sample.filename):
                result, _ = self.analyse(sample, "comparison")
                self.assertEqual(set(result), {"model", "simple_ann", "cnn", "agreement"})
                self.assertEqual(result["model"], "comparison")
                self.check_model_result(result["simple_ann"], "simple_ann")
                self.check_model_result(result["cnn"], "cnn")
                self.assertEqual(result["agreement"], result["simple_ann"]["prediction"]["style"]
                                 == result["cnn"]["prediction"]["style"])

    def test_undecodable_image_fails_job(self):
        response = self.client.post("/api/analyses", data={"model": "cnn", "result_type": "single"},
                                    files={"image": ("broken.png", b"not an image", "image/png")})
        analysis_id = response.json()["analysisId"]
        for _ in range(200):
            status = self.client.get(f"/api/analyses/{analysis_id}").json()
            if status["status"] != "processing":
                break
            time.sleep(0.05)
        self.assertEqual(status["status"], "failed")
        self.assertIsNone(status["result"])
        self.assertEqual(status["error"], "The image could not be decoded.")

    def test_unknown_id_and_invalid_model(self):
        self.assertEqual(self.client.get("/api/analyses/missing").status_code, 404)
        response = self.client.post("/api/analyses", data={"model": "resnet"},
                                    files={"image": ("x.png", b"x", "image/png")})
        self.assertEqual(response.status_code, 422)

    def test_cors_preflight_for_vite(self):
        response = self.client.options("/api/analyses", headers={
            "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:5173")


if __name__ == "__main__":
    unittest.main()
