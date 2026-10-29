
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.features.engineering import (
    build_features,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
REPORT_PATH = ROOT / "reports" / "temporal_drift_report.json"

CHUNK_SIZE = 100_000
SAMPLE_FRACTION = 0.10
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_origin_balance_ratio",
]
CATEGORICAL_FEATURES = ["type"]


def load_sample(path: Path) -> tuple[pd.DataFrame, dict]:
    columns = RAW_FEATURE_COLUMNS + [TARGET_COLUMN]

    samples = []
    total_rows = 0
    total_fraud = 0

    for chunk in pd.read_csv(path, usecols=columns, chunksize=CHUNK_SIZE):
        total_rows += len(chunk)

        labels = pd.to_numeric(chunk[TARGET_COLUMN], errors="raise")
        if not labels.isin([0, 1]).all():
            raise ValueError(f"Invalid target labels in {path}")

        total_fraud += int(labels.sum())

        sample = chunk.sample(
            frac=SAMPLE_FRACTION,
            random_state=RANDOM_STATE,
        )
        if not sample.empty:
            samples.append(sample)

    if not samples:
        raise ValueError(f"No rows found in {path}")

    sampled_df = pd.concat(samples, ignore_index=True)

    features = build_features(sampled_df)
    metadata = {
        "rows": total_rows,
        "fraud_cases": total_fraud,
        "fraud_rate": total_fraud / total_rows,
        "sampled_rows": len(features),
    }

    return features, metadata


def psi(reference, comparison, bins=10):
    """Approximate Population Stability Index using reference quantiles."""
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 3:
        return None

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0]
    cmp_counts = np.histogram(comparison, bins=edges)[0]

    ref_pct = np.clip(ref_counts / len(reference), 1e-6, None)
    cmp_pct = np.clip(cmp_counts / len(comparison), 1e-6, None)

    return float(np.sum((cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)))


def main():
    split_data = {}
    metadata = {}

    for split in ["train", "validation", "test"]:
        path = SPLIT_DIR / f"{split}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Split not found: {path}")

        print(f"Sampling {split} data...")
        split_data[split], metadata[split] = load_sample(path)

    report = {
        "purpose": "descriptive_temporal_drift_analysis",
        "sampling_fraction_per_chunk": SAMPLE_FRACTION,
        "random_state": RANDOM_STATE,
        "splits": metadata,
        "numeric_feature_psi_vs_train": {},
        "categorical_type_proportions": {},
        "interpretation_note": (
            "PSI is a descriptive screening statistic, not proof of "
            "causation or model degradation. Results depend on binning "
            "and sampling. Test results must not be used to tune the "
            "current model or threshold."
        ),
    }

    train = split_data["train"]

    for split in ["validation", "test"]:
        comparison = split_data[split]

        report["numeric_feature_psi_vs_train"][split] = {}

        for feature in NUMERIC_FEATURES:
            report["numeric_feature_psi_vs_train"][split][feature] = psi(
                train[feature],
                comparison[feature],
            )

    for split, features in split_data.items():
        report["categorical_type_proportions"][split] = (
            features["type"]
            .value_counts(normalize=True)
            .sort_index()
            .to_dict()
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== TEMPORAL DRIFT SUMMARY ===")
    print(json.dumps(report, indent=2))
    print(f"\nReport saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
