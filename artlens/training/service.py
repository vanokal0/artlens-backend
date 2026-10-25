"""One optional training job at a time, run as a separate cancellable process.

The process is the unchanged offline CLI (`python -m artlens.training.train`)
writing to a new run directory, so the active artifacts are never touched.
Progress is read from the CLI's own Keras log and training.jsonl records.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

from artlens.data import ROOT

RUNS_DIR = Path(os.environ.get("ARTLENS_RUNS_DIR", ROOT / "artifacts/runs"))
ANN_CACHE = ROOT / "data/cache/ann"
ACTIVE_STATES = {"running", "cancelling"}


class TrainingBusy(Exception):
    pass


class TrainingManager:
    def __init__(self, activate):
        """`activate(model_id, directory)` must validate the artifact and switch to it, or raise."""
        self._activate = activate
        self._lock = threading.Lock()
        self._job = None

    @property
    def busy(self):
        with self._lock:
            return self._job is not None and self._job["status"] in ACTIVE_STATES

    def start(self, settings):
        with self._lock:
            if self._job is not None and self._job["status"] in ACTIVE_STATES:
                raise TrainingBusy("A training job is already running")
            model = settings["model"]
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            training_id = f"{model}-{stamp}-{uuid.uuid4().hex[:6]}"
            output = RUNS_DIR / training_id
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-m", "artlens.training.train", "--model", model,
                       "--epochs", str(settings["epochs"]), "--learning-rate", repr(settings["learning_rate"]),
                       "--batch-size", str(settings["batch_size"]), "--seed", str(settings["random_seed"]),
                       "--threads", "4", "--output", str(output)]
            # Same worker/cache choices as the established baseline commands.
            command += ["--workers", "2", "--cache", str(ANN_CACHE)] if model == "simple_ann" else ["--workers", "1"]
            if settings["data_augmentation"]:
                command.append("--augment")
            log_path = RUNS_DIR / f"{training_id}.log"
            log = log_path.open("wb")
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, "PYTHONUNBUFFERED": "1"}, creationflags=flags)
            log.close()
            self._job = {"trainingId": training_id, "model": model, "settings": dict(settings),
                         "status": "running", "stage": "Loading dataset", "error": None,
                         "started": time.monotonic(), "finished": None, "process": process,
                         "output": output, "log": log_path, "cancel": False}
            threading.Thread(target=self._monitor, args=(self._job,), daemon=True).start()
        return self.status()

    def cancel(self):
        with self._lock:
            job = self._job
            if job is None or job["status"] not in ACTIVE_STATES:
                raise TrainingBusy("No training job is running")
            if job["process"].poll() is not None:
                # Training already finished; validation/activation cannot be interrupted safely.
                raise TrainingBusy("Training already finished and the new model is being activated")
            job["cancel"] = True
            job["status"] = "cancelling"
            job["stage"] = "Stopping training process"
        _kill_tree(job["process"])
        return self.status()

    def shutdown(self):
        with self._lock:
            job = self._job
        if job is not None and job["process"].poll() is None:
            job["cancel"] = True
            _kill_tree(job["process"], job["output"])
            job["process"].wait()
            shutil.rmtree(job["output"], ignore_errors=True)

    def status(self):
        with self._lock:
            job = dict(self._job) if self._job is not None else None
        if job is None:
            return {"status": "idle", "trainingId": None, "model": None, "settings": None, "stage": None,
                    "currentEpoch": 0, "totalEpochs": 0, "elapsedMs": 0, "metrics": None, "error": None}
        epochs = _epoch_records(job["output"])
        stage = job["stage"]
        if job["status"] == "running" and job["process"].poll() is None:
            started = _started_epoch(job["log"])
            if (job["output"] / "history.json").exists():
                stage = "Evaluating on validation set"
            elif started:
                stage = f"Training epoch {started} of {job['settings']['epochs']}"
            elif job["model"] == "simple_ann":
                stage = "Loading dataset and extracting 16 × 16 grid features"
        end = job["finished"] or time.monotonic()
        return {"status": job["status"], "trainingId": job["trainingId"], "model": job["model"],
                "settings": job["settings"], "stage": stage, "currentEpoch": len(epochs),
                "totalEpochs": job["settings"]["epochs"], "elapsedMs": round((end - job["started"]) * 1000),
                "metrics": epochs[-1] if epochs else None, "error": job["error"]}

    def _update(self, job, **fields):
        with self._lock:
            job.update(fields)

    def _monitor(self, job):
        code = job["process"].wait()
        if job["cancel"]:
            _kill_tree(job["process"], job["output"])
            shutil.rmtree(job["output"], ignore_errors=True)
            return self._update(job, status="cancelled", stage="Cancelled; previous model kept",
                                finished=time.monotonic())
        if code != 0:
            return self._update(job, status="failed", stage="Failed; previous model kept",
                                error=_failure_message(job["log"], code), finished=time.monotonic())
        self._update(job, stage="Validating and activating new model")
        try:
            self._activate(job["model"], job["output"])
        except Exception as exc:
            return self._update(job, status="failed", stage="Failed; previous model kept",
                                error=f"New model failed validation: {exc}", finished=time.monotonic())
        self._update(job, status="completed", stage="Completed; new model active", finished=time.monotonic())


def _kill_tree(process, output=None):
    """Stop the trainer and its spawned preprocessing workers."""
    if os.name != "nt":
        process.kill()
        return
    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    if output is None:
        return
    # The venv python.exe is a launcher; its real interpreter child can survive /T. Sweep any
    # process still referencing this run's unique output directory and kill it by PID.
    for _ in range(20):
        pids = _pids_referencing(output)
        if not pids:
            return
        for pid in pids:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        time.sleep(0.5)


def _pids_referencing(output):
    needle = str(Path(output).name).replace("'", "''")
    script = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              f"Where-Object {{ $_.CommandLine -like '*{needle}*' }} | "
              "ForEach-Object { $_.ProcessId }")
    result = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True)
    return [int(line) for line in result.stdout.split() if line.isdigit() and int(line) != os.getpid()]


def _epoch_records(output):
    path = Path(output) / "training.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # line still being written
            continue
        records.append({"epoch": record["epoch"], "loss": record.get("loss"), "accuracy": record.get("accuracy"),
                        "val_loss": record.get("val_loss"), "val_accuracy": record.get("val_accuracy")})
    return records


def _started_epoch(log):
    matches = re.findall(r"^Epoch (\d+)/\d+", _read(log), flags=re.MULTILINE)
    return int(matches[-1]) if matches else 0


def _failure_message(log, code):
    lines = [line.strip() for line in _read(log).splitlines() if line.strip()]
    errors = [line for line in lines if re.match(r"^[\w.]*(Error|Exception)\b|^train\.py: error|^usage:", line)
              or ": error:" in line]
    return errors[-1] if errors else f"Training process exited with code {code}"


def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
