
import numpy as np
import pandas as pd

from src.features.engineering import build_features


REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]

ALLOWED_TRANSACTION_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
}

MAX_BATCH_ROWS = 5_000


def predict_batch(
    transactions: pd.DataFrame,
    model,
    threshold: float,
) -> pd.DataFrame:
    """
    Score multiple transactions and return their original data
    plus fraud scores and review decisions.
    """

    if not isinstance(transactions, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    if transactions.empty:
        raise ValueError("The uploaded dataset is empty.")

    if len(transactions) > MAX_BATCH_ROWS:
        raise ValueError(
            f"Maximum batch size is {MAX_BATCH_ROWS:,} rows."
        )

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in transactions.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    data = transactions.copy()

    # Validate transaction types before feature engineering.
    invalid_types = (
        set(data["type"].dropna().unique())
        - ALLOWED_TRANSACTION_TYPES
    )

    if data["type"].isna().any() or invalid_types:
        raise ValueError("Invalid or missing transaction type.")

    # Reuse the project's existing feature engineering logic.
    # Extra columns, including isFraud, are not used as model inputs.
    features = build_features(data)

    scores = model.predict_proba(features)[:, 1]

    if not np.isfinite(scores).all():
        raise ValueError("Model produced invalid prediction scores.")

    results = transactions.copy()
    results["fraud_score"] = scores
    results["threshold"] = threshold
    results["decision"] = np.where(
        scores >= threshold,
        "REVIEW",
        "NOT_FLAGGED",
    )

    return results
