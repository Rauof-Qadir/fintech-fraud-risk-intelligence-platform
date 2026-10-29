
from pathlib import Path
import json

import joblib
import pandas as pd

from src.features.engineering import build_features, TARGET_COLUMN
from src.monitoring.metrics import build_monitoring_summary

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "splits" / "validation.csv"
MODEL_PATH = ROOT / "models" / "baseline_sgd.joblib"
POLICY_PATH = ROOT / "reports" / "operating_policy.json"
REPORT_PATH = ROOT / "reports" / "validation_monitoring_report.json"


def main():
    for path in [DATA_PATH, MODEL_PATH, POLICY_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading validation data...")
    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    y_true = pd.to_numeric(df[TARGET_COLUMN], errors="raise")

    if not y_true.isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    print("Building features...")
    X = build_features(df)

    print("Loading model and operating policy...")
    model = joblib.load(MODEL_PATH)

    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        policy = json.load(f)

    threshold = float(policy["threshold"])

    print("Generating model scores...")
    scores = model.predict_proba(X)[:, 1]

    summary = build_monitoring_summary(
        scores=scores,
        threshold=threshold,
        y_true=y_true.to_numpy(),
    )

    summary["dataset"] = "validation"
    summary["model_file"] = str(MODEL_PATH.relative_to(ROOT))
    summary["policy_file"] = str(POLICY_PATH.relative_to(ROOT))
    summary["threshold_source"] = "validation_selected_operating_policy"
    summary["threshold_is_provisional"] = True
    summary["scores_are_calibrated_probabilities"] = False

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== VALIDATION MONITORING SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
