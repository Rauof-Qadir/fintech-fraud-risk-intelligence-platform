
import json

import pandas as pd
import pytest

from src.data.ingestion import audit_csv


def sample_data():
    return pd.DataFrame({
        "step": [1, 2, 3],
        "type": ["PAYMENT", "TRANSFER", "CASH_OUT"],
        "amount": [100, -50, 200],
        "nameOrig": ["A", "B", "C"],
        "oldbalanceOrg": [1000, 800, 500],
        "newbalanceOrig": [900, 750, 300],
        "nameDest": ["X", "Y", "Z"],
        "oldbalanceDest": [0, 0, 100],
        "newbalanceDest": [100, 50, 300],
        "isFraud": [0, 1, 0],
        "isFlaggedFraud": [0, 0, 0],
    })


def test_csv_is_audited_across_chunks(tmp_path):
    csv_path = tmp_path / "transactions.csv"
    report_path = tmp_path / "report.json"

    sample_data().to_csv(csv_path, index=False)

    report = audit_csv(
        csv_path,
        report_path,
        chunk_size=2,
    )

    assert report["status"] == "COMPLETED"
    assert report["rows_inspected"] == 3
    assert report["chunks_processed"] == 2
    assert report["invalid_amount_rows"] == 1
    assert report_path.exists()

    saved_report = json.loads(
        report_path.read_text(encoding="utf-8")
    )
    assert saved_report["rows_inspected"] == 3


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit_csv(
            tmp_path / "missing.csv",
            tmp_path / "report.json",
        )


def test_zero_chunk_size_is_rejected(tmp_path):
    csv_path = tmp_path / "transactions.csv"
    sample_data().to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        audit_csv(
            csv_path,
            tmp_path / "report.json",
            chunk_size=0,
        )
