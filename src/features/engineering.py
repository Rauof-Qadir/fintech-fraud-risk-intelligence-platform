
import numpy as np
import pandas as pd


RAW_FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]

TARGET_COLUMN = "isFraud"


def build_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build candidate features using transaction-time inputs.

    Assumption:
    Input balances are available before the decision.
    The function does not use labels, post-transaction
    balances, or raw account identifiers.
    """
    required = set(RAW_FEATURE_COLUMNS)
    missing = sorted(required - set(transactions.columns))

    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}"
        )

    # Select only permitted input columns.
    features = transactions[
        RAW_FEATURE_COLUMNS
    ].copy()

    # Convert numerical inputs explicitly.
    numeric_columns = [
        "step",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
    ]

    for column in numeric_columns:
        features[column] = pd.to_numeric(
            features[column], errors="raise"
        )

    numeric_values = features[numeric_columns].to_numpy(
        dtype=float
    )

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Numeric feature inputs contain missing "
            "or non-finite values."
        )

    if (features["amount"] < 0).any():
        raise ValueError(
            "Transaction amounts must be non-negative."
        )

    if (features["step"] <= 0).any():
        raise ValueError(
            "Transaction steps must be positive."
        )

    if (
        (features["oldbalanceOrg"] < 0).any()
        or (features["oldbalanceDest"] < 0).any()
    ):
        raise ValueError(
            "Pre-transaction balances must be non-negative."
        )

    allowed_types = {
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    }

    if (
        features["type"].isna().any()
        or not features["type"].isin(allowed_types).all()
    ):
        raise ValueError(
            "Unknown or missing transaction type."
        )

    # Relative transaction size. The +1 avoids division
    # by zero; the value is a candidate feature, not a
    # calibrated probability.
    features["amount_to_origin_balance_ratio"] = (
        features["amount"]
        / (features["oldbalanceOrg"] + 1.0)
    )

    features["amount_exceeds_origin_balance"] = (
        features["amount"] > features["oldbalanceOrg"]
    ).astype("int8")

    features["origin_balance_is_zero"] = (
        features["oldbalanceOrg"] == 0
    ).astype("int8")

    features["destination_balance_is_zero"] = (
        features["oldbalanceDest"] == 0
    ).astype("int8")

    return features
