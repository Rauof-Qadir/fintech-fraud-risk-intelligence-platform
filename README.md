# FinTech Fraud Risk Intelligence Platform

An end-to-end machine learning project for transaction fraud risk analysis. It combines data quality checks, temporal evaluation, an ML inference API, batch predictions, and monitoring dashboards.

> **Research/demo project:** This project uses a synthetic PaySim-style transaction dataset. Its results do not establish real-world banking performance.

## Features

- Transaction data quality checks and temporal dataset splitting.
- Machine learning model training and evaluation.
- Validation-based operating threshold selection.
- FastAPI endpoints for health checks and fraud-risk predictions.
- Batch CSV prediction support.
- Streamlit dashboard for predictions and monitoring.
- Temporal drift and model evaluation reports.
- Unit tests and Docker Compose deployment.

## Architecture

```text
Transaction Data
       |
       v
Data Quality & Temporal Splitting
       |
       v
Feature Engineering & Model Evaluation
       |
       v
Trained Model + Operating Policy
       |
       +------> FastAPI
       |           |
       |           +--> Single Prediction
       |           +--> Batch Prediction
       |
       +------> Streamlit Dashboard

Monitoring Reports --> Dashboard
```

## Technology Stack

- Python, pandas, NumPy
- scikit-learn and joblib
- FastAPI and Uvicorn
- Streamlit
- pytest
- Docker and Docker Compose

## Run Locally

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the unit tests:

```powershell
$env:PYTHONPATH="."
python -m pytest tests/unit -v
```

Start the API:

```powershell
uvicorn src.api.main:app --reload
```

API documentation:

- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

Start the dashboard in a second terminal:

```powershell
streamlit run dashboard/app.py
```

Dashboard: http://localhost:8501

## Run with Docker

From the project root:

```powershell
docker compose build
docker compose up -d
```

Open these URLs:

- Dashboard: http://localhost:8501
- API documentation: http://localhost:8000/docs
- API health: http://localhost:8000/health

Stop the services:

```powershell
docker compose down
```

## Project Structure

```text
configs/       Configuration files
dashboard/     Streamlit dashboards
docs/          Model card and documentation
models/        Trained model artifact
reports/       Evaluation and monitoring reports
scripts/       Data preparation and evaluation scripts
src/api/       FastAPI application
src/data/      Ingestion, quality, and splitting
src/features/  Feature engineering
src/models/    Model training
src/inference/ Batch inference
src/monitoring/Monitoring metrics
tests/unit/    Unit tests
```

## Evaluation and Responsible Use

The project uses temporal data splits to evaluate performance on later transactions. Its operating threshold is selected using validation data rather than the held-out test set.

Fraud detection involves a trade-off between missed fraud and false alerts. The included reports document evaluation metrics, threshold analysis, and monitoring observations.

The dataset is synthetic, the cost assumptions are illustrative, and model scores should not automatically be interpreted as calibrated probabilities. Temporal drift indicators are descriptive, not proof of model failure.

This project is intended for learning, experimentation, and portfolio demonstration—not direct deployment in a real financial institution.
