"""End-to-end optional retraining workflow against a real uvicorn server.

Uses a temporary state file and runs directory, so the real active versions in
artifacts/active-models.json are never changed. Runs short (2-epoch) trainings on
the full prepared dataset; takes a few minutes.
"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import httpx

from artlens.data import ROOT, load_manifest, select_samples

BASELINES = {"simple_ann": "ann-full-seed42", "cnn": "cnn-full-seed42"}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TrainingWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = Path(tempfile.mkdtemp(prefix="artlens-training-test-"))
        cls.env = {**os.environ, "ARTLENS_STATE_FILE": str(cls.temp / "active-models.json"),
                   "ARTLENS_RUNS_DIR": str(cls.temp / "runs")}
        cls.painting = ROOT / select_samples(load_manifest(), "validation", 5, seed=42)[2].filename
        cls.start_server()

    @classmethod
    def tearDownClass(cls):
        cls.stop_server()
        shutil.rmtree(cls.temp, ignore_errors=True)

    @classmethod
    def start_server(cls):
        port = free_port()
        cls.server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "artlens.api:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, env=cls.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cls.client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30)
        deadline = time.monotonic() + 180
        while True:
            try:
                cls.client.get("/api/training/status")
                return
            except httpx.TransportError:
                if cls.server.poll() is not None or time.monotonic() > deadline:
                    cls.server.kill()
                    raise RuntimeError("API server did not start")
                time.sleep(0.5)

    @classmethod
    def stop_server(cls):
        cls.client.close()
        cls.server.terminate()
        cls.server.wait(timeout=30)

    def status(self):
        response = self.client.get("/api/training/status")
        self.assertEqual(response.status_code, 200)
        return response.json()

    def active_versions(self):
        return {m: info["version"] for m, info in self.status()["activeModels"].items()}

    def analysis_status_code(self):
        with self.painting.open("rb") as stream:
            response = self.client.post("/api/analyses", data={"model": "comparison", "result_type": "single"},
                                        files={"image": (self.painting.name, stream, "image/jpeg")})
        if response.status_code != 202:
            return response.status_code, None
        analysis_id = response.json()["analysisId"]
        for _ in range(400):
            status = self.client.get(f"/api/analyses/{analysis_id}").json()
            if status["status"] != "processing":
                return response.status_code, status
            time.sleep(0.25)
        self.fail("Analysis did not finish")

    def wait_for(self, predicate, timeout=600):
        deadline = time.monotonic() + timeout
        seen = []
        while time.monotonic() < deadline:
            status = self.status()
            seen.append(status)
            if predicate(status):
                return status, seen
            time.sleep(0.5)
        self.fail(f"Timed out; last status {seen[-1]}")

    def training_children(self):
        """PIDs of training processes writing into this test's temporary runs directory."""
        script = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like "
                  f"'*artlens.training *{self.temp.name}*' }} | ForEach-Object {{ $_.ProcessId }}")
        output = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                                capture_output=True, text=True).stdout
        return [int(line) for line in output.split()]

    def test_1_baselines_ready_without_training(self):
        status = self.status()
        self.assertEqual(status["status"], "idle")
        self.assertEqual(self.active_versions(), BASELINES)
        self.assertTrue(all(info["ready"] and info["baseline"] for info in status["activeModels"].values()))
        code, analysis = self.analysis_status_code()
        self.assertEqual(code, 202)
        self.assertEqual(analysis["status"], "completed")

    def test_2_invalid_settings_rejected(self):
        for body in ({"model": "simple_ann", "data_augmentation": True}, {"model": "resnet"},
                     {"model": "cnn", "epochs": 0}, {"model": "cnn", "learning_rate": 0}):
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/api/training", json=body).status_code, 422)
        self.assertEqual(self.client.post("/api/training/cancel").status_code, 409)

    def test_3_successful_training_activates_new_ann(self):
        response = self.client.post("/api/training", json={
            "model": "simple_ann", "epochs": 2, "learning_rate": 0.001, "batch_size": 32,
            "random_seed": 7, "data_augmentation": False})
        self.assertEqual(response.status_code, 202)
        started = response.json()
        self.assertEqual((started["status"], started["model"], started["totalEpochs"]), ("running", "simple_ann", 2))
        # Only one operation at a time; analyses are paused while training.
        self.assertEqual(self.client.post("/api/training", json={"model": "cnn"}).status_code, 409)
        self.assertEqual(self.analysis_status_code()[0], 409)
        final, seen = self.wait_for(lambda s: s["status"] not in ("running", "cancelling"))
        self.assertEqual(final["status"], "completed", final["error"])
        self.assertEqual(final["currentEpoch"], 2)
        epochs = [s["currentEpoch"] for s in [started] + seen]
        self.assertEqual(epochs, sorted(epochs))
        self.assertTrue(all(s["totalEpochs"] == 2 for s in seen))
        self.assertTrue(any(s["stage"].startswith("Training epoch") for s in seen if s["stage"]))
        self.assertGreater(final["elapsedMs"], 0)
        self.assertIsNotNone(final["metrics"]["val_loss"])
        self.assertEqual(final["activeModels"]["simple_ann"]["version"], final["trainingId"])
        self.assertFalse(final["activeModels"]["simple_ann"]["baseline"])
        self.assertEqual(final["activeModels"]["simple_ann"]["settings"]["random_seed"], 7)
        self.assertEqual(final["activeModels"]["cnn"]["version"], BASELINES["cnn"])
        self.assertTrue((ROOT / "artifacts" / BASELINES["simple_ann"] / "SUCCESS.json").exists())
        code, analysis = self.analysis_status_code()
        self.assertEqual((code, analysis["status"]), (202, "completed"))
        type(self).trained_ann = final["trainingId"]

    def test_4_cancel_preserves_previous_cnn(self):
        before = self.active_versions()
        response = self.client.post("/api/training", json={"model": "cnn", "epochs": 3, "data_augmentation": True})
        self.assertEqual(response.status_code, 202)
        training_id = response.json()["trainingId"]
        self.wait_for(lambda s: (s["stage"] or "").startswith("Training epoch 1"), timeout=180)
        cancelling = self.client.post("/api/training/cancel")
        self.assertEqual(cancelling.status_code, 202)
        self.assertIn(cancelling.json()["status"], ("cancelling", "cancelled"))
        final, _ = self.wait_for(lambda s: s["status"] not in ("running", "cancelling"), timeout=60)
        self.assertEqual(final["status"], "cancelled")
        self.assertEqual(self.active_versions(), before)
        self.assertFalse((self.temp / "runs" / training_id).exists())
        self.assertEqual(self.training_children(), [])
        self.assertEqual(self.analysis_status_code()[1]["status"], "completed")

    def test_5_failure_preserves_previous_ann(self):
        before = self.active_versions()
        response = self.client.post("/api/training", json={"model": "simple_ann", "epochs": 200})
        self.assertEqual(response.status_code, 202)
        self.wait_for(lambda s: s["status"] == "running" and self.training_children(), timeout=60)
        for pid in self.training_children():  # simulate the training process crashing
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        final, _ = self.wait_for(lambda s: s["status"] not in ("running", "cancelling"), timeout=60)
        self.assertEqual(final["status"], "failed")
        self.assertTrue(final["error"])
        self.assertEqual(self.active_versions(), before)
        self.assertEqual(self.analysis_status_code()[1]["status"], "completed")

    def test_6_restart_restores_active_versions(self):
        expected = {"simple_ann": self.trained_ann, "cnn": BASELINES["cnn"]}
        self.stop_server()
        self.start_server()
        self.assertEqual(self.active_versions(), expected)
        self.assertEqual(self.status()["status"], "idle")
        # A smoke-test artifact is never accepted as an active version.
        smoke = self.temp / "smoke-cnn"
        shutil.copytree(ROOT / "artifacts" / BASELINES["cnn"], smoke)
        metadata = json.loads((smoke / "metadata.json").read_text(encoding="utf-8"))
        metadata["training"]["smoke"] = True
        (smoke / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        state = Path(self.env["ARTLENS_STATE_FILE"])
        stored = json.loads(state.read_text(encoding="utf-8"))
        state.write_text(json.dumps({**stored, "cnn": str(smoke)}), encoding="utf-8")
        self.stop_server()
        self.start_server()
        self.assertEqual(self.active_versions(), expected)


if __name__ == "__main__":
    unittest.main()
