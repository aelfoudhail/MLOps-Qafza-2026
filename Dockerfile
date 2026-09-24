FROM python:3.13-slim

WORKDIR /Users/elfoudhailahmed/Documents/MLOps-Qafza-2026

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY src/ ./src/
COPY config/ ./config/
COPY artifacts/preprocessor.pkl artifacts/decision_threshold.pkl artifacts/top_states.pkl artifacts/top_seller_states.pkl artifacts/feature_list.txt ./artifacts/
COPY mlruns/ ./mlruns/
COPY mlflow.db .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]