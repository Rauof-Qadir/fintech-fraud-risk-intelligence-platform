
import pandas as pd

from src.data.quality import audit_dataframe


def make_transactions():
    return pd.DataFrame({
        "step": [1, 2, 3],
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
        "amount": [100, 500, 200],
        "nameOrig": ["A", "B", "C"],
        "oldbalanceOrg": [1000, 800, 500],
        "newbalanceOrig": [900, 300, 300],
        "nameDest": ["X", "Y", "Z"],
        "oldbalanceDest": [0, 0, 100],
        "newbalanceDest": [100, 500, 300],
        "isFraud": [0, 1, 0],
        "isFlaggedFraud": [0, 0, 0],
    })


def test_valid_data_has_no_quality_issues():
    df = make_transactions()

    report = audit_dataframe(df)

    assert report["rows_inspected"] == 3
    assert report["missing_required_columns"] == []
    assert report["invalid_amount_rows"] == 0
    assert report["invalid_transaction_type_rows"] == 0
    assert report["invalid_fraud_label_rows"] == 0


def test_negative_amount_is_detected():
    df = make_transactions()
    df.loc[0, "amount"] = -100

    report = audit_dataframe(df)

    assert report["invalid_amount_rows"] == 1


def test_invalid_transaction_type_is_detected():
    df = make_transactions()
    df.loc[0, "type"] = "UNKNOWN"

    report = audit_dataframe(df)

    assert report["invalid_transaction_type_rows"] == 1


def test_invalid_fraud_label_is_detected():
    df = make_transactions()
    df.loc[0, "isFraud"] = 5

    report = audit_dataframe(df)

    assert report["invalid_fraud_label_rows"] == 1


def test_missing_required_column_is_detected():
    df = make_transactions().drop(columns=["amount"])

    report = audit_dataframe(df)

    assert "amount" in report["missing_required_columns"]


def test_invalid_flagged_fraud_label_is_detected():
    df = make_transactions()
    df.loc[0, "isFlaggedFraud"] = 2

    report = audit_dataframe(df)

    assert report["invalid_labels"]["isFlaggedFraud"] == 1


def test_non_numeric_amount_is_detected():
    df = make_transactions()
    df["amount"] = df["amount"].astype(object)
    df.loc[0, "amount"] = "unknown"

    report = audit_dataframe(df)

    assert report["invalid_numeric_values"]["amount"] == 1
    assert report["invalid_amount_rows"] == 1


def test_negative_balance_is_reported():
    df = make_transactions()
    df.loc[0, "oldbalanceOrg"] = -10

    report = audit_dataframe(df)

    assert report["negative_balance_rows"] == 1


def test_invalid_step_is_detected():
    df = make_transactions()
    df.loc[0, "step"] = 0

    report = audit_dataframe(df)

    assert report["invalid_step_rows"] == 1
