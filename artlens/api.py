"""ArtLens FastAPI service: run with `python -m uvicorn artlens.api:app --port 8000`.

The active Simple ANN and CNN versions (full baselines unless retrained) are
loaded once at startup. Each analysis runs on a single background worker
thread, so polling is never blocked. Training form fields sent with analyses are
ignored; optional retraining is a separate, explicit /api/training workflow.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import io
import logging
import os
import threading
import time
from typing import Literal
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, model_validator

from artlens.models import (PREPROCESSORS, configure, load_active, raw_outputs, result_from_raw, save_state,
                            validate_artifact)
from artlens.training import TrainingBusy, TrainingManager

log = logging.getLogger("artlens.api")

MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # same limit as the frontend
MAX_ACTIVE_JOBS = 20
JOB_TTL_SECONDS = 30 * 60
CORS_ORIGIN_REGEX = os.environ.get("ARTLENS_CORS_ORIGIN_REGEX", r"http://(localhost|127\.0\.0\.1)(:\d+)?")

STAGES = {
    "simple_ann": ("Extracting 16 × 16 intensity grid features", "Running Simple ANN"),
    "cnn": ("Preparing CNN input (128 × 128 RGB)", "Running CNN"),
}

models = {}  # model_id -> loaded Keras model (active version)
active = {}  # model_id -> {"info": ..., "directory": Path}
models_lock = threading.Lock()
jobs = {}
jobs_lock = threading.Lock()
executor = None


def activate(model_id, directory):
    """Validate a new artifact, then atomically switch to it and persist the choice."""
    model, info = validate_artifact(directory, model_id)
    with models_lock:
        directories = {m: entry["directory"] for m, entry in active.items()}
        directories[model_id] = directory
        save_state(directories)
        models[model_id] = model
        active[model_id] = {"info": info, "directory": directory}
    print(f"Activated {model_id} version {info['version']}", flush=True)


trainer = TrainingManager(activate)


@asynccontextmanager
async def lifespan(_app):
    global executor
    configure(seed=42, threads=4)
    for model_id, (model, info, directory) in load_active().items():
        models[model_id] = model
        active[model_id] = {"info": info, "directory": directory}
        print(f"Loaded {model_id} version {info['version']}", flush=True)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="artlens-analysis")
    try:
        yield
    finally:
        trainer.shutdown()
        executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="ArtLens API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origin_regex=CORS_ORIGIN_REGEX,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


def _update(job_id, **fields):
    with jobs_lock:
        jobs[job_id].update(fields)


def _model_result(job_id, model_id, data, progress, snapshot):
    """Preprocess with an independent byte stream, then run the preloaded model."""
    _update(job_id, stage=STAGES[model_id][0], progress=progress[0])
    model_input = PREPROCESSORS[model_id](io.BytesIO(data))
    _update(job_id, stage=STAGES[model_id][1], progress=progress[1])
    result = result_from_raw(model_id, raw_outputs(snapshot[model_id], model_input))
    return {key: result[key] for key in ("model", "prediction", "probabilities")}


def _run(job_id, mode, data):
    _update(job_id, stage="Validating image", progress=5)
    with models_lock:
        snapshot = dict(models)  # one consistent pair of active versions per analysis
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        return _fail(job_id, "The image could not be decoded.")
    try:
        if mode == "comparison":
            ann = _model_result(job_id, "simple_ann", data, (15, 40), snapshot)
            cnn = _model_result(job_id, "cnn", data, (50, 80), snapshot)
            _update(job_id, stage="Comparing predictions", progress=95)
            result = {"model": "comparison", "simple_ann": ann, "cnn": cnn,
                      "agreement": ann["prediction"]["style"] == cnn["prediction"]["style"]}
        else:
            result = _model_result(job_id, mode, data, (20, 60), snapshot)
    except Exception as exc:  # report any preprocessing/inference failure to the client
        log.exception("Analysis %s failed", job_id)
        return _fail(job_id, f"Analysis failed: {exc}")
    _update(job_id, status="completed", progress=100, stage="Completed",
            result=result, finished=time.monotonic())


def _fail(job_id, message):
    _update(job_id, status="failed", error=message, finished=time.monotonic())


def _purge_expired():
    now = time.monotonic()
    for job_id in [k for k, j in jobs.items() if j["finished"] and now - j["finished"] > JOB_TTL_SECONDS]:
        del jobs[job_id]


@app.post("/api/analyses", status_code=202)
async def create_analysis(
    image: UploadFile = File(...),
    model: Literal["simple_ann", "cnn", "comparison"] = Form("cnn"),
    result_type: Literal["single", "probabilities"] = Form("single"),
    # Training settings: accepted for frontend compatibility, ignored during inference.
    epochs: int | None = Form(None),
    learning_rate: float | None = Form(None),
    batch_size: int | None = Form(None),
    random_seed: int | None = Form(None),
    data_augmentation: bool | None = Form(None),
):
    if trainer.busy:
        raise HTTPException(409, "Model training is in progress; analysis is temporarily unavailable")
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "Empty image upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds 30 MB")
    job_id = uuid.uuid4().hex
    with jobs_lock:
        _purge_expired()
        if sum(j["status"] == "processing" for j in jobs.values()) >= MAX_ACTIVE_JOBS:
            raise HTTPException(503, "Too many analyses in progress")
        jobs[job_id] = {"status": "processing", "progress": 0, "stage": "Waiting for inference",
                        "started": time.monotonic(), "finished": None, "result": None, "error": None}
    executor.submit(_run, job_id, model, data)
    return {"analysisId": job_id}


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str):
    with jobs_lock:
        job = jobs.get(analysis_id)
        if job is None:
            raise HTTPException(404, "Analysis not found")
        job = dict(job)
    end = job["finished"] or time.monotonic()
    status = {"status": job["status"], "progress": job["progress"], "stage": job["stage"],
              "elapsedMs": round((end - job["started"]) * 1000), "result": job["result"]}
    if job["status"] == "failed":
        status["error"] = job["error"]
    return status


class TrainingRequest(BaseModel):
    model: Literal["simple_ann", "cnn"]
    epochs: int = Field(20, ge=1, le=200)
    learning_rate: float = Field(0.001, gt=0, le=1)
    batch_size: int = Field(32, ge=1, le=256)
    random_seed: int = Field(42, ge=0, le=2**31 - 1)
    data_augmentation: bool = False

    @model_validator(mode="after")
    def augmentation_only_for_cnn(self):
        if self.data_augmentation and self.model == "simple_ann":
            raise ValueError("data_augmentation is only supported for the CNN; "
                             "the research Simple ANN is always trained without augmentation")
        return self


def _training_status():
    with models_lock:
        models_info = {m: entry["info"] for m, entry in active.items()}
    return {**trainer.status(), "activeModels": models_info}


@app.post("/api/training", status_code=202)
def start_training(request: TrainingRequest):
    try:
        trainer.start(request.model_dump())
    except TrainingBusy as exc:
        raise HTTPException(409, str(exc))
    return _training_status()


@app.get("/api/training/status")
def training_status():
    return _training_status()


@app.post("/api/training/cancel", status_code=202)
def cancel_training():
    try:
        trainer.cancel()
    except TrainingBusy as exc:
        raise HTTPException(409, str(exc))
    return _training_status()
