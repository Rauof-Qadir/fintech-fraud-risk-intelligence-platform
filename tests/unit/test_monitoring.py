
import pytest

from src.monitoring.metrics import build_monitoring_summary


def test_monitoring_without_labels():
    result = build_monitoring_summary(
        scores=[0.1, 0.8, 0.9],
        threshold=0.8,
    )

    assert result["rows"] == 3
    assert result["alerts"] == 2
    assert result["alert_rate"] == pytest.approx(2 / 3)
    assert "recall" not in result


def test_monitoring_with_labels():
    result = build_monitoring_summary(
        scores=[0.1, 0.8, 0.9, 0.2],
        threshold=0.8,
        y_true=[0, 1, 1, 0],
    )

    assert result["actual_fraud_cases"] == 2
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["average_precision"] is not None


def test_rejects_invalid_scores():
    with pytest.raises(ValueError):
        build_monitoring_summary(
            scores=[0.1, float("nan")],
            threshold=0.5,
        )


def test_rejects_mismatched_labels():
    with pytest.raises(ValueError):
        build_monitoring_summary(
            scores=[0.1, 0.9],
            threshold=0.5,
            y_true=[1],
        )
