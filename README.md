# Late Delivery Prediction Service

Predicts whether a placed order from the Olist e-commerce dataset will be
delivered late, using the model trained in `notebooks/` (Qafza MLOps
training, Task 2). This repo is the inference service — training stays in
the notebooks, this code loads what they produced and serves predictions
through a real API, containerized, tested, and monitored.

**Status: all 10 steps complete and verified**, not just written. Every
piece below has been run for real, against the actual trained model, not
assumed to work from the code alone.

## What this service does

Given the details of a placed order (items, price, payment, dates, customer
and seller location), it predicts whether that order will be delivered
late, returning a prediction, a probability, and the model version that
made the call.

## An important design decision: aggregated input, not raw orders

A real order arrives as raw pieces — a list of items, a list of payments, a
zip code — not as one aggregated row. Building a full aggregator that
replicates notebook 1's database joins in real time would be its own
sub-project. This service instead expects an order already in the
**aggregated** shape notebook 5 works with (item counts, total price,
distance, etc. already computed). See
`src/validation/schemas.py::OrderRequest` for the exact fields expected. A
true raw-order endpoint, if needed later, would sit in front of this
service as a separate aggregation step, not inside it.

## Prerequisites

- Python 3.13
- Docker + Docker Compose (for containerized runs)
- Your trained artifacts from Task 2, placed in `artifacts/`:
  `model.pkl`, `preprocessor.pkl`, `decision_threshold.pkl`,
  `top_states.pkl`, `top_seller_states.pkl`, `feature_list.txt`
  (and `logreg_scaler.pkl` too, only if logistic regression won in
  notebook 6)
- A `.env` file in the repo root (copy `.env.example` and fill in real
  values)

## Running it from zero (local, without Docker)

```bash
git clone https://github.com/aelfoudhail/MLOps-Qafza-2026.git
cd MLOps-Qafza-2026

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements-dev.txt
pre-commit install

cp .env.example .env
# edit .env with your real database values (not needed for the API itself,
# only for the training-side scripts)

# copy your trained artifacts into artifacts/ (see Prerequisites above)

# register the trained model with MLflow, this is what the live API
# actually loads from, not the raw .pkl directly
python -m src.register_model

pytest

uvicorn app.main:app --reload
```

Once it's running:
- Health check: `curl http://localhost:8000/health`
- Metrics: `curl http://localhost:8000/metrics`
- Interactive API docs: open `http://localhost:8000/docs`, try `/predict`
  directly from the browser, it comes pre-filled with a working example

## Running it with Docker Compose (recommended, matches production shape)

```bash
docker compose up --build
```

This brings up the database, the API, and persistent storage volumes for
artifacts, all with one command. The API container loads the model from
the MLflow registry (`mlruns/` and `mlflow.db`), and the smaller artifacts
(preprocessor, threshold, state lists) directly from `artifacts/`.

**A known limitation worth knowing**: the Dockerfile's `WORKDIR` is set to
match the exact absolute path MLflow's local file-based store bakes into
its registry records (`/Users/<you>/Documents/MLOps-Qafza-2026`). This is a
real workaround, not a clean fix — MLflow's file-based tracking store
records absolute paths, so the container's internal folder structure has
to mirror the machine that registered the model. A production-grade setup
would use a proper MLflow tracking server with remote artifact storage
(S3, Azure Blob, etc.) specifically to avoid this coupling. That's real,
heavier infrastructure, out of scope for this project's timeline.

## Project structure

```
app/            FastAPI application (routes only, no business logic)
config/         config.yaml, the single source of truth for paths/params
data/           raw/processed data (versioned with DVC, not git)
models/         placeholder for artifacts pulled from a registry
notebooks/      the six training notebooks from task 2
src/
  config.py                 loads config.yaml, resolves ${ENV_VAR} placeholders
  logging_config.py          sets up console + rotating file logging
  pipeline.py                 glues feature engineering + prediction + monitoring
  register_model.py           logs and registers the trained model with MLflow
  features/engineer.py        same transformations as notebook 5, as functions
  models/predictor.py         loads the registered model + fitted artifacts
  validation/
    schemas.py                 pydantic request/response schemas
    validate_training_data.py  Great Expectations checks on training data
  monitoring/
    metrics.py                  in-process request/prediction counters
    analyze_predictions.py      drift check against the task 2 baseline late rate
tests/          pytest suite: unit, schema/leakage, model, integration tests
artifacts/      preprocessor.pkl, decision_threshold.pkl, state lists, etc.
                (model.pkl itself is DVC-tracked, not committed to git)
mlruns/         MLflow's local tracking store (git-ignored, DVC-untracked)
.github/workflows/ci.yml   lint, format check, tests, then a conditional
                            Docker image build
.pre-commit-config.yaml    runs ruff + black automatically before every commit
Dockerfile      builds a slim image for the API service
docker-compose.yml   brings up db + api + persistent artifact storage
```

## Data versioning (DVC)

`model.pkl` and the other trained artifacts are versioned with DVC, not
committed directly to git — GitHub rejects any single file over 100MB, and
`model.pkl` is ~118MB. Each artifact has a small `.dvc` pointer file
(committed to git) that references the real file, stored in a local DVC
remote (`~/dvc-storage` on this machine).

```bash
dvc pull   # fetch the real files, given the .dvc pointers
dvc push   # after retraining, push the new versions
```

**Known limitation**: the DVC remote is a local folder on the original
developer's machine, not a shared cloud remote. A teammate cloning this
repo could see the `.dvc` pointer files but couldn't `dvc pull` the real
data without access to that same local folder. A real team setup would use
S3/GCS/Azure as the remote instead.

## Data validation (Great Expectations)

`src/validation/validate_training_data.py` runs 6 checks against the raw
training table (`order_id` uniqueness, `customer_state` format, non-negative
prices, etc.) before it's trusted for anything downstream.

```bash
python -m src.validation.validate_training_data
```

This validates **training data** in batch. Live API requests are validated
separately and in real time by the pydantic schemas in
`src/validation/schemas.py` — a deliberate split, since Great Expectations
is built for batch dataframe validation, not single-row real-time checks.

## Experiment tracking and model registry (MLflow)

```bash
python -m src.register_model
mlflow ui --backend-store-uri sqlite:///mlflow.db   # view at localhost:5000
```

The trained model is logged with its real hyperparameters and the decision
threshold, then registered with a version number and a `champion` alias
(the modern replacement for MLflow's deprecated Staging/Production
"stages" — the underlying registry mechanism, just renamed).

**The live service loads the model from this registry**
(`models:/late_delivery_classifier@champion`), not from a local `.pkl`
path directly — verified to produce identical predictions either way.

## Testing

```bash
pytest -v
```

Four categories, 22 tests total, all self-contained (no dependency on your
real trained model, since tests should never require production secrets or
data to exist on whoever's machine runs them):

- **Unit tests** (`test_features.py`) — the date engineering and category
  bucketing logic, verified line-by-line against the real notebook cells
- **Schema/leakage tests** (`test_schemas.py`) — request validation, plus an
  explicit check that no post-delivery field (review score, delivered date)
  can ever exist on the request schema
- **Model tests** (`test_predictor.py`) — a throwaway fake model built in
  memory proves the prediction logic, threshold handling, and error cases
  work correctly
- **Integration tests** (`test_api_integration.py`) — real HTTP requests
  through the full FastAPI app, using a throwaway model + registry built
  fresh for the test session (`tests/conftest.py`), so these run identically
  whether on your laptop or a bare CI runner with none of your real files

## CI/CD

Every push to `main` triggers `.github/workflows/ci.yml`:
1. Lint with `ruff`
2. Check formatting with `black`
3. Run the full test suite
4. If all of the above pass, build the Docker image

Pre-commit hooks (`.pre-commit-config.yaml`) run `ruff` and `black`
automatically on every local commit, catching formatting/lint issues before
they're even pushed. Install once per clone:
```bash
pre-commit install
```

**Known limitation**: the CI image-build step uses placeholder (empty)
artifact files, since the real trained artifacts aren't available on a
fresh GitHub runner (the DVC remote is local-only). This verifies the
Dockerfile's structure builds correctly, not that the resulting image
contains a working model. A real deployment pipeline would pull real
artifacts from a shared, CI-reachable DVC remote or MLflow server.

## Monitoring

- `GET /metrics` — request count, error rate, average latency, and the
  late/on-time distribution of predictions served this process
- Every prediction is appended to `logs/predictions.jsonl` (timestamp,
  order_id, prediction, probability, model version) — once real delivery
  outcomes are known later, these logged predictions can be joined against
  actual results to measure real-world accuracy, not just the offline
  test-set number
- `python -m src.monitoring.analyze_predictions` checks the observed late
  rate against the 6.6% baseline from task 2's test set, flagging drift
  beyond a configurable threshold (`config.yaml` → `monitoring:`)

**What I'd alert on in a real production deployment:**
- `error_rate` above 5% sustained over a few minutes — usually bad client
  input or an artifact/model failing to load, not the model itself
- `avg_latency_ms` above ~500ms — this model is fast, a slowdown usually
  points to something upstream, not the prediction logic
- late rate drifting more than 5 points from the 6.6% baseline, sustained
  over a real sample (100+ predictions, not just the first few) — a small
  sample swinging by chance isn't the same as a genuine shift
- `/health` reporting `model_loaded: false`, ever — this should page
  immediately, it means the service is serving zero real predictions

## Why each config value exists

- `artifacts.dir` / `*_file` keys — nothing in `src/` hardcodes a path
- `features.numeric_features` / `categorical_features` — must match
  notebook 5 exactly; kept in config so there's one place to check they
  still agree, not duplicated across code and notebook
- `features.high_risk_months` — the `[11, 2, 3]` flag from notebook 4/5's
  findings, editable without a code change
- `mlflow.model_name` / `model_alias` — so the registry lookup isn't
  hardcoded inside `predictor.py`
- `monitoring.baseline_late_rate` / `drift_threshold_points` — the real
  task 2 test-set late rate, and how much deviation counts as worth a
  warning
- `logging.*` — level/file/rotation settings, changeable per environment
  without touching code