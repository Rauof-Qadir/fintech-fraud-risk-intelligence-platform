
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
}

ALLOWED_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
}

NUMERIC_COLUMNS = {
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
}

BALANCE_COLUMNS = {
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
}


def audit_dataframe(df: pd.DataFrame) -> dict:
    """Audit one transaction batch without modifying it."""

    missing_columns = sorted(
        REQUIRED_COLUMNS - set(df.columns)
    )

    missing_values = {
        column: int(count)
        for column, count in df.isna().sum().items()
        if count > 0
    }

    numeric = {}
    invalid_numeric_values = {}

    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            continue

        original = df[column]
        converted = pd.to_numeric(
            original, errors="coerce"
        )

        invalid = (
            original.notna() & converted.isna()
        )

        if pd.api.types.is_numeric_dtype(converted):
            invalid |= (
                converted.notna()
                & ~np.isfinite(converted)
            )

        invalid_numeric_values[column] = int(
            invalid.sum()
        )

        numeric[column] = converted

    amount = numeric.get(
        "amount", pd.Series(index=df.index, dtype=float)
    )

    invalid_amount_rows = int(
        (
            amount.isna()
            | ~np.isfinite(amount)
            | (amount < 0)
        ).sum()
    )

    invalid_type = (
        ~df["type"].isin(ALLOWED_TYPES)
        | df["type"].isna()
        if "type" in df.columns
        else pd.Series(False, index=df.index)
    )

    invalid_labels = {}

    for column in ["isFraud", "isFlaggedFraud"]:
        if column not in numeric:
            continue

        labels = numeric[column]

        invalid_labels[column] = int(
            (
                labels.isna()
                | ~labels.isin([0, 1])
            ).sum()
        )

    negative_balance_rows = pd.Series(
        False, index=df.index
    )

    for column in BALANCE_COLUMNS:
        if column in numeric:
            values = numeric[column]
            negative_balance_rows |= values.lt(0).fillna(False)

    step = numeric.get(
        "step", pd.Series(index=df.index, dtype=float)
    )

    invalid_step_rows = int(
        (step.isna() | ~np.isfinite(step) | (step <= 0)).sum()
    )

    return {
        "rows_inspected": int(len(df)),
        "columns_found": int(len(df.columns)),
        "missing_required_columns": missing_columns,
        "missing_values": missing_values,
        "duplicate_rows_in_batch": int(
            df.duplicated().sum()
        ),
        "invalid_numeric_values": invalid_numeric_values,
        "invalid_amount_rows": invalid_amount_rows,
        "invalid_transaction_type_rows": int(
            invalid_type.sum()
        ),
        "invalid_labels": invalid_labels,
        "invalid_fraud_label_rows": invalid_labels.get(
        "isFraud", 0
        ),
        "negative_balance_rows": int(
            negative_balance_rows.sum()
        ),
        "invalid_step_rows": invalid_step_rows,
    }
