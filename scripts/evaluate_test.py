
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.features.engineering import (
    build_features,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "baseline_sgd.joblib"
TEST_PATH = ROOT / "data" / "processed" / "splits" / "test.csv"
POLICY_PATH = ROOT / "reports" / "operating_policy.json"
REPORT_PATH = ROOT / "reports" / "final_test_metrics.json"

FP_COST = 5
FN_COST = 150


def main():
    for path in [MODEL_PATH, TEST_PATH, POLICY_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

    if policy.get("status") != "FEASIBLE_POLICY_FOUND":
        raise ValueError("No feasible validation policy is available.")

    threshold = float(policy["threshold"])

    # Load only the selected features and target.
    df = pd.read_csv(
        TEST_PATH,
        usecols=RAW_FEATURE_COLUMNS + [TARGET_COLUMN],
    )

    y = pd.to_numeric(
        df[TARGET_COLUMN], errors="raise"
    ).to_numpy(dtype=int)

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Test target must contain only 0 and 1.")

    X = build_features(df)

    model = joblib.load(MODEL_PATH)
    scores = model.predict_proba(X)[:, 1]

    # Fixed threshold selected on validation, not test.
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    alerts = tp + fp
    actual_fraud = int(y.sum())

    metrics = {
        "evaluation": "held_out_temporal_test",
        "model": "SGDClassifier",
        "threshold": threshold,
        "test_rows": int(len(y)),
        "actual_fraud_cases": actual_fraud,
        "actual_fraud_rate": float(y.mean()),
        "average_precision": float(
            average_precision_score(y, scores)
        ),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "alerts": int(alerts),
        "alert_rate": float(alerts / len(y)),
        "estimated_business_cost": int(
            FP_COST * fp + FN_COST * fn
        ),
        "false_positive_cost_assumption": FP_COST,
        "false_negative_cost_assumption": FN_COST,
        "costs_are_illustrative": True,
        "max_alert_rate": float(policy["max_alert_rate"]),
        "min_recall": float(policy["min_recall"]),
        "alert_rate_constraint_met": bool(
            alerts / len(y) <= policy["max_alert_rate"]
        ),
        "recall_constraint_met": bool(
            (tp / actual_fraud if actual_fraud else 0)
            >= policy["min_recall"]
        ),
        "threshold_tuned_on_test": False,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\n=== FINAL HELD-OUT TEST RESULTS ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
