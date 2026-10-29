from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.engineering import (
    build_features,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "baseline_sgd.joblib"
VALIDATION_PATH = (
    ROOT / "data" / "processed" / "splits" / "validation.csv"
)
REPORT_DIR = ROOT / "reports"

FP_COST = 5
FN_COST = 150


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation data not found: {VALIDATION_PATH}"
        )

    print("Loading validation data...")
    columns = RAW_FEATURE_COLUMNS + [TARGET_COLUMN]
    df = pd.read_csv(VALIDATION_PATH, usecols=columns)

    y = pd.to_numeric(
        df[TARGET_COLUMN], errors="raise"
    ).to_numpy(dtype=int)

    X = build_features(df)

    print("Loading saved model...")
    model = joblib.load(MODEL_PATH)

    print("Calculating fraud probabilities...")
    scores = model.predict_proba(X)[:, 1]

    # Test data is deliberately not loaded or used.
    thresholds = np.unique(
        np.concatenate([
            np.linspace(0.01, 0.99, 99),
            np.quantile(scores, np.linspace(0, 1, 101)),
            np.array([0.5]),
        ])
    )

    total = len(y)
    actual_fraud = int(y.sum())
    results = []

    for threshold in thresholds:
        predictions = scores >= threshold

        tp = int(np.sum(predictions & (y == 1)))
        fp = int(np.sum(predictions & (y == 0)))
        fn = int(np.sum(~predictions & (y == 1)))
        tn = int(np.sum(~predictions & (y == 0)))

        alerts = tp + fp

        precision = tp / alerts if alerts else 0.0
        recall = tp / actual_fraud if actual_fraud else 0.0

        estimated_cost = FP_COST * fp + FN_COST * fn

        results.append({
            "threshold": float(threshold),
            "alerts": alerts,
            "alert_rate": alerts / total,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": precision,
            "recall": recall,
            "estimated_business_cost": estimated_cost,
        })

    report = pd.DataFrame(results).sort_values("threshold")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = REPORT_DIR / "threshold_analysis.csv"
    report.to_csv(csv_path, index=False)

    # Best threshold under our illustrative cost assumptions.
    best = report.sort_values(
        ["estimated_business_cost", "threshold"],
        ascending=[True, False],
    ).iloc[0]

    print("\n=== BEST THRESHOLD BY ESTIMATED COST ===")
    print(best.to_string())

    print("\n=== SELECTED THRESHOLD COMPARISON ===")
    selected = report[
        report["threshold"].isin([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ]
    print(selected.to_string(index=False))

    print(f"\nValidation fraud cases: {actual_fraud:,}")
    print(f"Threshold report saved: {csv_path}")


if __name__ == "__main__":
    main()