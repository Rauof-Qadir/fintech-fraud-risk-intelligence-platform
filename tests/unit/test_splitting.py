
import pandas as pd
import pytest

from src.data.splitting import split_csv_by_time


def test_temporal_splits_are_correct(tmp_path):
    source = tmp_path / "transactions.csv"
    output = tmp_path / "splits"

    df = pd.DataFrame({
        "step": [1, 520, 521, 631, 632, 743, 744],
        "amount": [10, 20, 30, 40, 50, 60, 70],
        "isFraud": [0, 1, 0, 1, 0, 1, 0],
    })

    df.to_csv(source, index=False)

    result = split_csv_by_time(
        source, output, chunk_size=2
    )

    assert result["train"]["rows"] == 2
    assert result["validation"]["rows"] == 2
    assert result["test"]["rows"] == 2

    assert result["train"]["fraud"] == 1
    assert result["validation"]["fraud"] == 1
    assert result["test"]["fraud"] == 1

    assert result["excluded"]["rows"] == 1

    train = pd.read_csv(output / "train.csv")
    assert train["step"].tolist() == [1, 520]


def test_invalid_labels_are_rejected(tmp_path):
    source = tmp_path / "transactions.csv"

    pd.DataFrame({
        "step": [1],
        "isFraud": [5],
    }).to_csv(source, index=False)

    with pytest.raises(ValueError, match="Invalid fraud labels"):
        split_csv_by_time(
            source, tmp_path / "splits"
        )


def test_missing_source_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        split_csv_by_time(
            tmp_path / "missing.csv",
            tmp_path / "splits",
        )
