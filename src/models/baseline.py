
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.engineering import build_features, RAW_FEATURE_COLUMNS, TARGET_COLUMN


ROOT = Path(__file__).resolve().parents[2]

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_origin_balance_ratio",
    "amount_exceeds_origin_balance",
    "origin_balance_is_zero",
    "destination_balance_is_zero",
]
CATEGORICAL_FEATURES = ["type"]


def load_split(path: Path):
    # Read only columns needed for model training.
    columns = RAW_FEATURE_COLUMNS + [TARGET_COLUMN]
    df = pd.read_csv(path, usecols=columns)

    y = pd.to_numeric(df[TARGET_COLUMN], errors="raise").astype(int)
    X = build_features(df)

    return X, y


def train_baseline():
    train_path = ROOT / "data" / "processed" / "splits" / "train.csv"
    validation_path = (
        ROOT / "data" / "processed" / "splits" / "validation.csv"
    )

    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError(
            "Train/validation splits missing. Create temporal splits first."
        )

    print("Loading training data...")
    X_train, y_train = load_split(train_path)

    print(f"Training rows: {len(X_train):,}")
    print(f"Training fraud cases: {int(y_train.sum()):,}")

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=1.0,
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    max_iter=30,
                    tol=1e-3,
                    random_state=42,
            )
            ),
        ]
    )

    print("Training baseline model...")
    model.fit(X_train, y_train)

    del X_train, y_train

    print("Loading validation data...")
    X_val, y_val = load_split(validation_path)

    print("Generating validation probabilities...")
    probabilities = model.predict_proba(X_val)[:, 1]

    # This is a preliminary threshold, not a production policy.
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": "SGDClassifier",
        "purpose": "baseline",
        "threshold": 0.5,
        "validation_rows": int(len(y_val)),
        "validation_fraud_cases": int(y_val.sum()),
        "average_precision": float(
            average_precision_score(y_val, probabilities)
        ),
        "precision": float(precision_score(
            y_val, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_val, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_val, predictions, zero_division=0
        )),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "training_rows": int(model.named_steps["classifier"].t_.__int__())
        if False else None,
    }

    models_dir = ROOT / "models"
    reports_dir = ROOT / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "baseline_sgd.joblib")

    with open(
        reports_dir / "baseline_validation_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\n=== VALIDATION RESULTS ===")
    print(json.dumps(metrics, indent=2))
    print("\nModel saved: models/baseline_sgd.joblib")
    print("Metrics saved: reports/baseline_validation_metrics.json")


if __name__ == "__main__":
    train_baseline()
