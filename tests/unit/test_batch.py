
import numpy as np
import pandas as pd
import pytest

from src.inference.batch import predict_batch


class FakeModel:
    """Deterministic model for testing batch inference."""

    def predict_proba(self, features):
        scores = np.array([0.95, 0.20])[:len(features)]

        return np.column_stack([
            1 - scores,
            scores,
        ])


@pytest.fixture
def transactions():
    return pd.DataFrame({
        "step": [10, 11],
        "type": ["TRANSFER", "PAYMENT"],
        "amount": [5000.0, 200.0],
        "oldbalanceOrg": [6000.0, 1000.0],
        "oldbalanceDest": [100.0, 50.0],
        "isFraud": [1, 0],
    })


def test_batch_returns_scores_and_decisions(transactions):
    result = predict_batch(
        transactions,
        FakeModel(),
        threshold=0.87,
    )

    assert len(result) == 2
    assert result["fraud_score"].tolist() == [0.95, 0.20]
    assert result["decision"].tolist() == [
        "REVIEW",
        "NOT_FLAGGED",
    ]


def test_original_columns_are_preserved(transactions):
    result = predict_batch(
        transactions,
        FakeModel(),
        threshold=0.87,
    )

    assert "isFraud" in result.columns
    assert "amount" in result.columns
    assert result["isFraud"].tolist() == [1, 0]


def test_missing_required_column_is_rejected(transactions):
    transactions = transactions.drop(columns=["amount"])

    with pytest.raises(ValueError, match="Missing required columns"):
        predict_batch(transactions, FakeModel(), 0.87)


def test_empty_dataframe_is_rejected(transactions):
    empty = transactions.iloc[0:0]

    with pytest.raises(ValueError, match="empty"):
        predict_batch(empty, FakeModel(), 0.87)


def test_invalid_threshold_is_rejected(transactions):
    with pytest.raises(ValueError, match="Threshold"):
        predict_batch(transactions, FakeModel(), 1.5)


def test_invalid_transaction_type_is_rejected(transactions):
    transactions.loc[0, "type"] = "UNKNOWN"

    with pytest.raises(ValueError, match="transaction type"):
        predict_batch(transactions, FakeModel(), 0.87)
