
from pathlib import Path
import logging
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import io

import pandas as pd
from fastapi import File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from src.inference.batch import predict_batch

from src.features.engineering import build_features


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("fraud_risk_api")


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "baseline_sgd.joblib"
POLICY_PATH = ROOT / "reports" / "operating_policy.json"

app = FastAPI(
    title="FinTech Fraud Risk Intelligence API",
    description="Fraud risk scoring for transaction data.",
    version="1.0.0",
)


# Load the saved model and threshold once when the API module starts.
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not POLICY_PATH.exists():
    raise FileNotFoundError(f"Operating policy not found: {POLICY_PATH}")

import json

model = joblib.load(MODEL_PATH)

with open(POLICY_PATH, "r", encoding="utf-8") as file:
    policy = json.load(file)

THRESHOLD = float(policy["threshold"])


logger.info(
    "API initialized | model=baseline_sgd | threshold=%.2f",
    THRESHOLD,
)


class TransactionInput(BaseModel):
    step: int = Field(gt=0, description="Positive simulation time step")
    type: str = Field(
        description="CASH_IN, CASH_OUT, DEBIT, PAYMENT, or TRANSFER"
    )
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)


class FraudPrediction(BaseModel):
    fraud_score: float
    threshold: float
    decision: str
    model_name: str


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=FraudPrediction)
def predict_fraud(transaction: TransactionInput):
    allowed_types = {
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    }

    if transaction.type not in allowed_types:
        raise HTTPException(
            status_code=422,
            detail=f"type must be one of: {sorted(allowed_types)}",
        )

    transaction_df = pd.DataFrame([transaction.model_dump()])

    try:
        features = build_features(transaction_df)
        start_time = time.perf_counter()
        fraud_score = float(model.predict_proba(features)[0, 1])
    except (ValueError, TypeError) as exc:
        logger.warning("Prediction input rejected: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    decision = (
        "REVIEW"
        if fraud_score >= THRESHOLD
        else "NOT_FLAGGED"
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Prediction completed | decision=%s | latency_ms=%.2f",
        decision,
        elapsed_ms,
    )

    return FraudPrediction(
        fraud_score=fraud_score,
        threshold=THRESHOLD,
        decision=decision,
        model_name="baseline_sgd",
    )


@app.post("/predict/batch")
async def predict_batch_endpoint(
    file: UploadFile = File(...),
):
    """Upload a CSV and download its fraud predictions."""

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    max_file_bytes = 5 * 1024 * 1024  # 5 MB
    content = await file.read(max_file_bytes + 1)

    if len(content) > max_file_bytes:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 5 MB upload limit.",
        )

    try:
        transactions = pd.read_csv(io.BytesIO(content))

        results = predict_batch(
            transactions=transactions,
            model=model,
            threshold=THRESHOLD,
        )

        output = io.BytesIO()
        results.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    'attachment; filename="fraud_predictions.csv"'
            },
        )

    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read the CSV. Check its format and encoding.",
        ) from exc
