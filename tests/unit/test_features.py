
import pandas as pd
import pytest

from src.features.engineering import build_features


def sample_transactions():
    return pd.DataFrame({
        "step": [1, 2, 3],
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
        "amount": [500.0, 1200.0, 0.0],
        "oldbalanceOrg": [1000.0, 1000.0, 0.0],
        "oldbalanceDest": [0.0, 500.0, 0.0],
        "newbalanceOrig": [500.0, 0.0, 0.0],
        "newbalanceDest": [500.0, 1700.0, 0.0],
        "isFraud": [0, 1, 0],
        "nameOrig": ["A", "B", "C"],
        "nameDest": ["X", "Y", "Z"],
    })


def test_expected_features_are_created():
    result = build_features(sample_transactions())

    assert "amount_to_origin_balance_ratio" in result
    assert "amount_exceeds_origin_balance" in result
    assert "origin_balance_is_zero" in result
    assert "destination_balance_is_zero" in result


def test_target_and_post_transaction_columns_are_excluded():
    result = build_features(sample_transactions())

    assert "isFraud" not in result.columns
    assert "newbalanceOrig" not in result.columns
    assert "newbalanceDest" not in result.columns
    assert "nameOrig" not in result.columns
    assert "nameDest" not in result.columns


def test_relative_amount_feature_is_calculated():
    result = build_features(sample_transactions())

    assert result.loc[0, "amount_to_origin_balance_ratio"] == pytest.approx(
        500 / 1001
    )


def test_zero_origin_balance_is_handled():
    result = build_features(sample_transactions())

    assert result.loc[2, "origin_balance_is_zero"] == 1
    assert result.loc[2, "amount_to_origin_balance_ratio"] == 0


def test_negative_amount_is_rejected():
    data = sample_transactions()
    data.loc[0, "amount"] = -100

    with pytest.raises(ValueError, match="non-negative"):
        build_features(data)


def test_missing_required_feature_is_rejected():
    data = sample_transactions().drop(
        columns=["oldbalanceDest"]
    )

    with pytest.raises(ValueError, match="Missing required"):
        build_features(data)
