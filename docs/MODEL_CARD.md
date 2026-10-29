
# Model Card: FinTech Fraud Risk Intelligence Platform

## 1. Model Overview

- Model name: Baseline SGD Fraud Classifier
- Model type: Binary classification
- Framework: scikit-learn
- Primary use: Transaction risk screening
- Output: Fraud score and review decision
- Decision threshold: 0.87
- Threshold source: Validation-selected operating policy

This is a portfolio research prototype using a synthetic
PaySim-style transaction dataset.

It is not a production-validated banking fraud system.

## 2. Intended Use

The model is designed to:
- Demonstrate transaction fraud-risk scoring.
- Prioritize transactions for human review.
- Support experimentation with monitoring and evaluation.
- Demonstrate an end-to-end ML engineering workflow.

The model must not be used as the sole basis for blocking
real customer transactions or making consequential decisions.

## 3. Dataset

The dataset contains approximately 6.36 million synthetic
transactions and includes transaction type, amount, balances,
time step, and fraud labels.

The dataset is synthetic. Results do not establish performance
on real banking customers or live payment networks.

## 4. Features

The model uses information intended to be available before
the transaction decision:

- step
- type
- amount
- oldbalanceOrg
- oldbalanceDest
- Engineered amount-to-origin-balance ratio
- Engineered balance and amount indicators

Customer identifiers, the target label, and post-transaction
balances are excluded from model input.

The availability and timing of these fields must be verified
before using a different dataset or live transaction feed.

## 5. Training and Evaluation

- Model: Standardized numeric features, one-hot transaction
  type encoding, and SGDClassifier.
- Class imbalance: class_weight="balanced".
- Data splitting: chronological train, validation, and test.
- Operating threshold: selected using validation data.
- Test evaluation: performed at the fixed selected threshold.

The validation and test periods have different fraud prevalence.
This makes temporal evaluation and ongoing monitoring important.

## 6. Held-Out Test Results

Recorded results for the current model version:

- Average precision: 0.7494
- Precision: 0.2970
- Recall: 0.9457
- F1 score: 0.4521
- True positives: 1,184
- False positives: 2,802
- False negatives: 68
- Test alert rate: 4.46%

These results describe performance on this particular
synthetic held-out test split only.

They are not guarantees of future performance.

## 7. Operating Policy

The threshold was selected on validation data to target:
- Maximum alert rate: 3%
- Minimum recall: 90%

Validation results:
- Alert rate: 2.75%
- Recall: 90.34%

Held-out test results:
- Alert rate: 4.46%
- Recall: 94.57%

The test alert rate exceeded the intended 3% target.
This demonstrates that validation constraints may not
generalize under distribution or prevalence shift.

The threshold has not been retuned on the held-out test set.

## 8. Score Interpretation

The model's output is a decision score from the classifier.
It must not be interpreted as a calibrated real-world
probability of fraud.

A score at or above 0.87 triggers REVIEW under the current
operating policy. REVIEW means further investigation may be
needed; it does not establish that fraud occurred.

## 9. Limitations and Risks

- Synthetic data may not represent real fraud behaviour.
- Fraud prevalence changes across time periods.
- False positives may create investigation workload.
- False negatives may allow fraudulent transactions through.
- The illustrative cost assumptions are not verified bank costs.
- No live banking integration or production validation exists.
- Score calibration has not been established.
- Monitoring on labelled production data is not yet available.

## 10. Monitoring Plan

Monitor:
- Input schema errors and missing-value rates.
- Transaction type and numeric feature distributions.
- Fraud prevalence when trustworthy labels become available.
- Alert rate and score distribution.
- Precision and recall when labels mature.
- API latency and error rates.

Investigate material changes before deciding whether
retraining or threshold review is appropriate.

## 11. Human Oversight

Use REVIEW as a screening signal, not a final fraud verdict.
Consequential actions should follow an independently validated
policy with appropriate human oversight.

## 12. Version and Status

- Version: 1.0
- Status: Research / portfolio prototype
- Dataset: Synthetic PaySim-style data
- Threshold: 0.87
- Model artifact: models/baseline_sgd.joblib
- Evaluation report: reports/final_test_metrics.json
- Monitoring report: reports/validation_monitoring_report.json
