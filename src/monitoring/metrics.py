
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
)


def build_monitoring_summary(
    scores,
    threshold,
    y_true=None,
):
    scores = np.asarray(scores, dtype=float)

    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("scores must be a non-empty 1D array.")

    if not np.isfinite(scores).all():
        raise ValueError("scores contain missing or non-finite values.")

    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("scores must be between 0 and 1.")

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    predictions = (scores >= threshold).astype(int)
    alerts = int(predictions.sum())

    summary = {
        "rows": int(len(scores)),
        "threshold": float(threshold),
        "alerts": alerts,
        "alert_rate": float(alerts / len(scores)),
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_p90": float(np.quantile(scores, 0.90)),
        "score_p95": float(np.quantile(scores, 0.95)),
        "score_p99": float(np.quantile(scores, 0.99)),
    }

    if y_true is not None:
        y_true = np.asarray(y_true)

        if y_true.ndim != 1 or len(y_true) != len(scores):
            raise ValueError(
                "y_true must be 1D and match the score count."
            )

        if not np.isin(y_true, [0, 1]).all():
            raise ValueError("y_true must contain only 0 and 1.")

        y_true = y_true.astype(int)

        if len(np.unique(y_true)) < 2:
            summary["average_precision"] = None
        else:
            summary["average_precision"] = float(
                average_precision_score(y_true, scores)
            )

        summary["actual_fraud_cases"] = int(y_true.sum())
        summary["actual_fraud_rate"] = float(y_true.mean())
        summary["precision"] = float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        )
        summary["recall"] = float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        )

    return summary
