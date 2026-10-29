
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "reports" / "threshold_analysis.csv"
OUTPUT_PATH = ROOT / "reports" / "operating_policy.json"

MAX_ALERT_RATE = 0.03
MIN_RECALL = 0.90


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Threshold report not found: {INPUT_PATH}"
        )

    results = pd.read_csv(INPUT_PATH)

    required = {
        "threshold",
        "alert_rate",
        "recall",
        "precision",
        "estimated_business_cost",
        "alerts",
        "false_positives",
        "false_negatives",
    }

    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing report columns: {sorted(missing)}")

    feasible = results[
        (results["alert_rate"] <= MAX_ALERT_RATE)
        & (results["recall"] >= MIN_RECALL)
    ].copy()

    if feasible.empty:
        policy = {
            "status": "NO_FEASIBLE_POLICY",
            "reason": (
                "No tested threshold satisfies both alert-rate "
                "and minimum-recall requirements."
            ),
            "max_alert_rate": MAX_ALERT_RATE,
            "min_recall": MIN_RECALL,
            "selection_data": "validation",
        }
    else:
        best = feasible.sort_values(
            ["estimated_business_cost", "threshold"],
            ascending=[True, False],
        ).iloc[0]

        policy = {
            "status": "FEASIBLE_POLICY_FOUND",
            "selection_data": "validation",
            "threshold": float(best["threshold"]),
            "max_alert_rate": MAX_ALERT_RATE,
            "min_recall": MIN_RECALL,
            "actual_alert_rate": float(best["alert_rate"]),
            "recall": float(best["recall"]),
            "precision": float(best["precision"]),
            "alerts": int(best["alerts"]),
            "false_positives": int(best["false_positives"]),
            "false_negatives": int(best["false_negatives"]),
            "estimated_business_cost": float(
                best["estimated_business_cost"]
            ),
            "false_positive_cost_assumption": 5,
            "false_negative_cost_assumption": 150,
            "costs_are_illustrative": True,
            "threshold_is_provisional": True,
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(policy, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(policy, indent=2))
    print(f"\nPolicy report saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
