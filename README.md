# ArtLens backend

ArtLens is a FastAPI service that classifies a painting as **Baroque, Impressionism,
Cubism, Pop Art or Ukiyo-e**. It can use the research-paper Simple ANN, a CNN, or both
models together (Comparison mode). Training a model again is optional and has to be
started explicitly. The frontend repository is `../lab1-frontend`.

Run all commands from this directory. The project uses a single environment:
`.venv`, Python 3.12, CPU only.

## Quick start

```powershell
# 1. Install dependencies (once). Use requirements.lock.txt for exact tested versions.
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Start the backend. The trained models in artifacts/ are loaded automatically.
& .\.venv\Scripts\python.exe -m uvicorn artlens.api:app --host 127.0.0.1 --port 8000
```

To connect the frontend, create `../lab1-frontend/.env.local` and run `npm run dev`:

```dotenv
VITE_USE_MOCK_API=false
VITE_API_BASE_URL=http://127.0.0.1:8000
```

CORS allows `http://localhost:<port>` and `http://127.0.0.1:<port>`. To allow other
origins, set the `ARTLENS_CORS_ORIGIN_REGEX` environment variable.

## API

| Method | URL | Purpose |
|---|---|---|
| `POST` | `/api/analyses` | Upload an image with `model` = `simple_ann`, `cnn` or `comparison`. Returns `202 {"analysisId"}` |
| `GET` | `/api/analyses/{analysisId}` | Returns `status`, `progress`, `stage`, `elapsedMs` and `result` (prediction plus all five class scores) |
| `POST` | `/api/training` | Retrains one model in the background. Returns `409` if a training run is already active |
| `GET` | `/api/training/status` | Shows the training state, epochs, elapsed time, metrics and active model versions |
| `POST` | `/api/training/cancel` | Cancels the training run; the previous model stays active |

The complete frontend contract, with real example responses, is
[`.github/docs/API_CONTRACT.md`](.github/docs/API_CONTRACT.md).

- Training settings sent together with an analysis request are ignored; analysis never
  retrains a model. Analysis jobs run one at a time on a background thread, so the
  status endpoints stay responsive while a job runs.
- Training runs `python -m artlens.training` in a separate process and writes to
  `artifacts/runs/`. The new model is validated before it is activated, and the active
  versions are saved in `artifacts/active-models.json`. Delete that file to go back to
  the baseline models. While training runs, new analyses return `409`. Data
  augmentation can only be used with the CNN.

## Project layout

```text
artlens/                 Python package (all code, five modules)
  api.py                 FastAPI app: the single server entry point
  data.py                classes, ANN/CNN preprocessing, dataset split loader, multiprocessing
  models.py              Simple ANN (research paper) and CNN, inference, active model versions
  training.py            training CLI, batches, metrics, background training jobs for the API
  tools.py               dataset audit/split (prepare), verify, parallel benchmark
tests/                   unit and end-to-end API tests
scripts/                 one-off WikiArt download tools (need pandas, requests, tqdm, kagglehub)
data/                    raw, selected and prepared dataset (Git-ignored)
artifacts/               trained models, training runs, active-models.json (Git-ignored)
.github/                 project documentation, reports and research material (see .github/README.md)
.tools/python312/        local Python 3.12 that .venv is built on (do not delete)
requirements.txt         direct dependencies; requirements.lock.txt has the tested versions
```

## Other commands

```powershell
# Tests (about 3 minutes; includes real training runs on the dataset)
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Rebuild the baseline models offline. Output directories must not exist yet;
# delete artifacts/active-models.json afterwards.
& .\.venv\Scripts\python.exe -m artlens.training --model simple_ann --epochs 40 --learning-rate 0.001 --batch-size 32 --seed 42 --workers 2 --threads 4 --cache data/cache/ann --output artifacts/ann-full-seed42
& .\.venv\Scripts\python.exe -m artlens.training --model cnn --epochs 50 --learning-rate 0.001 --batch-size 32 --seed 42 --workers 1 --threads 4 --output artifacts/cnn-full-seed42

# Dataset preparation and checks (already done; outputs go to data/prepared/)
& .\.venv\Scripts\python.exe -m artlens.tools prepare
& .\.venv\Scripts\python.exe -m artlens.tools verify
& .\.venv\Scripts\python.exe -m artlens.tools benchmark --size 100 --repeats 2
```
