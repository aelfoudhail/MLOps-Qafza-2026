# Late Delivery Prediction Service

Predicts whether a placed order from the Olist e-commerce dataset will be
delivered late, using the model trained in `notebooks/` (Qafza MLOps
training, Task 2). This repo is the inference service only — training
stays in the notebooks, this code just loads what they produced and serves
predictions.

## Prerequisites

- Python 3.13
- Your trained artifacts from Task 2, placed in `artifacts/`:
  `model.pkl`, `preprocessor.pkl`, `decision_threshold.pkl`,
  `top_states.pkl`, `top_seller_states.pkl`, `feature_list.txt`
  (and `logreg_scaler.pkl` too, only if logistic regression won in
  notebook 6)
- A `.env` file in the repo root (copy `.env.example` and fill in real
  values, see below)

## Running it from zero

```bash
git clone https://github.com/aelfoudhail/MLOps-Qafza-2026.git
cd MLOps-Qafza-2026

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements-dev.txt

cp .env.example .env
# then edit .env with your real database values

# copy your trained artifacts into artifacts/ (see Prerequisites above)

pytest

uvicorn app.main:app --reload
```

Once it's running:
- Health check: `curl http://localhost:8000/health`
- Interactive API docs: open `http://localhost:8000/docs` in a browser

## An important design decision

A real order arrives as raw pieces — a list of items, a list of payments, a
zip code — not as one aggregated row. Building a full aggregator that
replicates notebook 1's database joins in real time would be its own
sub-project. This service instead expects an order already in the
*aggregated* shape notebook 5 works with (item counts, total price,
distance, etc. already computed). See
`src/validation/schemas.py::OrderRequest` for the exact fields expected.
A true raw-order endpoint, if needed later, would sit in front of this
service as a separate aggregation step, not inside it.

## Project structure

```
app/            FastAPI application (routes only, no business logic)
config/         config.yaml, the single source of truth for paths/params
data/           raw/processed data (versioned with DVC later, not git)
models/         placeholder for artifacts pulled from a registry (later)
notebooks/      the six training notebooks from task 2
src/
  config.py             loads config.yaml, resolves ${ENV_VAR} placeholders
  features/engineer.py   same transformations as notebook 5, as functions
  models/predictor.py    loads fitted artifacts, runs inference
  validation/schemas.py  pydantic request/response schemas
tests/          pytest unit tests, self-contained, don't need real artifacts
artifacts/      where model.pkl, preprocessor.pkl, etc. get placed
```