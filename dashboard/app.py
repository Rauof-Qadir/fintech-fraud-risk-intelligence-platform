
from pathlib import Path
import json

import pandas as pd
import requests
import streamlit as st

import os

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]

st.set_page_config(
    page_title="FinTech Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f4fbf5, #edf7ef);
    }
    [data-testid="stSidebar"] {
        background: #103d2b;
    }
    [data-testid="stSidebar"] * {
        color: white;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #d9e8dd;
        padding: 16px;
        border-radius: 12px;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(filename):
    path = REPORTS / filename
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return None


def find_value(data, *keys, default=None):
    """Return the first matching key from a report."""
    if not isinstance(data, dict):
        return default

    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return default


def show_metric(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def format_percent(value):
    if value is None:
        return "N/A"
    value = float(value)
    # Report rates are generally ratios, e.g. 0.0275.
    return f"{value * 100:.2f}%"


def extract_psi(data, path=""):
    """Find PSI values in nested drift report JSON."""
    found = []

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path} / {key}" if path else key

            if "psi" in str(key).lower():
                if isinstance(value, (int, float)):
                    found.append(
                        {"Feature / comparison": current_path,
                         "PSI": float(value)}
                    )
                else:
                    found.extend(extract_psi(value, current_path))
            elif isinstance(value, (dict, list)):
                found.extend(extract_psi(value, current_path))

    elif isinstance(data, list):
        for index, value in enumerate(data):
            found.extend(extract_psi(value, f"{path}[{index}]"))

    return found


# Load saved reports
policy = load_json("operating_policy.json")
test_report = load_json("final_test_metrics.json")
validation_report = load_json("validation_monitoring_report.json")
drift_report = load_json("temporal_drift_report.json")

# Sidebar
with st.sidebar:
    st.title("🛡️ FinTech Risk")
    st.caption("Fraud Risk Intelligence Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Single Prediction",
            "Batch CSV Analysis",
            "Model Monitoring",
        ],
    )

    st.divider()
    st.caption("Portfolio / research prototype")
    st.caption("Synthetic PaySim-style data")


# Header and API status
st.title("FinTech Fraud Risk Intelligence")
st.caption(
    "Transaction scoring, batch analysis and model monitoring "
    "in one workspace."
)

health, health_error = api_health()

if health and health.get("status") == "ok":
    st.success("Prediction API is connected.")
else:
    st.warning(
        "Prediction API is unavailable. Start FastAPI to use "
        "single prediction and batch analysis."
    )
    if health_error:
        st.caption(health_error)


if page == "Overview":
    st.subheader("Platform Overview")

    threshold = find_value(
        policy, "threshold", "selected_threshold",
        default=0.87
    )
    test_ap = find_value(
        test_report, "average_precision", "pr_auc", "ap"
    )
    test_recall = find_value(test_report, "recall")
    test_precision = find_value(test_report, "precision")
    test_alert_rate = find_value(test_report, "alert_rate")

    if test_alert_rate is None and test_report:
        alerts = find_value(test_report, "alerts")
        rows = find_value(test_report, "rows", "test_rows")
        if alerts is not None and rows:
            test_alert_rate = alerts / rows

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        show_metric("Decision Threshold", f"{float(threshold):.3f}")
    with c2:
        show_metric(
            "Test Average Precision",
            f"{float(test_ap):.4f}" if test_ap is not None else "N/A",
        )
    with c3:
        show_metric(
            "Test Recall",
            format_percent(test_recall),
        )
    with c4:
        show_metric(
            "Test Alert Rate",
            format_percent(test_alert_rate),
        )

    st.divider()
    st.markdown("### Navigate the platform")

    left, right = st.columns(2)
    with left:
        st.info(
            "**Single Prediction**\n\n"
            "Score one transaction and inspect the model decision."
        )
        st.info(
            "**Batch CSV Analysis**\n\n"
            "Score a CSV file and download the prediction results."
        )
    with right:
        st.info(
            "**Model Monitoring**\n\n"
            "Compare validation and held-out test performance."
        )
        st.warning(
            "**Important:** These results come from synthetic data. "
            "They do not establish real-world banking performance."
        )

    if test_precision is not None:
        st.caption(
            f"Held-out test precision: {float(test_precision):.4f}"
        )


elif page == "Single Prediction":
    st.subheader("Single Transaction Prediction")
    st.write(
        "Enter the transaction's available pre-decision fields. "
        "The API will return a score and review decision."
    )

    with st.form("single_prediction_form"):
        step = st.number_input(
            "Transaction step",
            min_value=1,
            value=100,
            step=1,
        )
        txn_type = st.selectbox(
            "Transaction type",
            ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"],
        )
        amount = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=1000.0,
            step=100.0,
        )
        origin_balance = st.number_input(
            "Origin balance before transaction",
            min_value=0.0,
            value=2000.0,
            step=100.0,
        )
        destination_balance = st.number_input(
            "Destination balance before transaction",
            min_value=0.0,
            value=1000.0,
            step=100.0,
        )

        submitted = st.form_submit_button(
            "Analyze Transaction",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        payload = {
            "step": int(step),
            "type": txn_type,
            "amount": float(amount),
            "oldbalanceOrg": float(origin_balance),
            "oldbalanceDest": float(destination_balance),
        }

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=15,
            )

            if response.ok:
                result = response.json()
                score = result.get("fraud_score")
                decision = result.get("decision", "Unknown")
                threshold_value = result.get("threshold")

                a, b, c = st.columns(3)
                with a:
                    show_metric(
                        "Fraud Score",
                        f"{float(score):.4f}"
                        if score is not None else "N/A",
                    )
                with b:
                    show_metric("Decision", decision)
                with c:
                    show_metric(
                        "Threshold",
                        f"{float(threshold_value):.3f}"
                        if threshold_value is not None else "N/A",
                    )

                if decision == "REVIEW":
                    st.warning(
                        "This transaction meets the model's review "
                        "threshold. It is not proof of fraud."
                    )
                else:
                    st.success(
                        "The transaction is below the current review "
                        "threshold. This does not guarantee safety."
                    )

                with st.expander("View API response"):
                    st.json(result)
            else:
                st.error(f"API error ({response.status_code})")
                st.code(response.text[:2000])

        except requests.RequestException as exc:
            st.error(f"Could not reach the prediction API: {exc}")


elif page == "Batch CSV Analysis":
    st.subheader("Batch CSV Analysis")
    st.write(
        "Upload a CSV with the required transaction columns. "
        "The API returns a downloadable CSV with fraud scores."
    )

    st.code(", ".join(REQUIRED_COLUMNS))

    uploaded_file = st.file_uploader(
        "Upload transaction CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("File is larger than the 5 MB upload limit.")
        else:
            try:
                preview_df = pd.read_csv(uploaded_file)

                missing = [
                    column for column in REQUIRED_COLUMNS
                    if column not in preview_df.columns
                ]

                if missing:
                    st.error(f"Missing required columns: {missing}")
                elif preview_df.empty:
                    st.error("The uploaded CSV is empty.")
                elif len(preview_df) > 5000:
                    st.error(
                        "Maximum batch size is 5,000 rows. "
                        "Please upload a smaller file."
                    )
                else:
                    st.caption(
                        f"Rows: {len(preview_df):,} | "
                        f"Columns: {len(preview_df.columns)}"
                    )
                    st.dataframe(
                        preview_df.head(10),
                        use_container_width=True,
                    )

                    if st.button(
                        "Run Batch Predictions",
                        type="primary",
                    ):
                        uploaded_file.seek(0)

                        try:
                            response = requests.post(
                                f"{API_URL}/predict/batch",
                                files={
                                    "file": (
                                        uploaded_file.name,
                                        uploaded_file.getvalue(),
                                        "text/csv",
                                    )
                                },
                                timeout=120,
                            )

                            if response.ok:
                                from io import BytesIO

                                results = pd.read_csv(
                                    BytesIO(response.content)
                                )

                                if "decision" in results.columns:
                                    alerts = int(
                                        (results["decision"] == "REVIEW").sum()
                                    )
                                else:
                                    alerts = 0

                                m1, m2, m3 = st.columns(3)
                                with m1:
                                    show_metric(
                                        "Transactions",
                                        f"{len(results):,}",
                                    )
                                with m2:
                                    show_metric("Review Alerts", f"{alerts:,}")
                                with m3:
                                    show_metric(
                                        "Alert Rate",
                                        format_percent(
                                            alerts / len(results)
                                            if len(results) else 0
                                        ),
                                    )

                                if "decision" in results.columns:
                                    st.markdown("#### Decision distribution")
                                    counts = (
                                        results["decision"]
                                        .value_counts()
                                        .rename_axis("Decision")
                                    )
                                    st.bar_chart(counts)

                                st.markdown("#### Prediction results")
                                st.dataframe(
                                    results.head(100),
                                    use_container_width=True,
                                )

                                st.download_button(
                                    "Download predictions CSV",
                                    data=results.to_csv(index=False).encode(
                                        "utf-8"
                                    ),
                                    file_name="fraud_predictions.csv",
                                    mime="text/csv",
                                    type="primary",
                                )
                            else:
                                st.error(
                                    f"Batch API error "
                                    f"({response.status_code})"
                                )
                                st.code(response.text[:2000])

                        except (requests.RequestException, ValueError) as exc:
                            st.error(f"Batch analysis failed: {exc}")

            except (ValueError, UnicodeDecodeError) as exc:
                st.error(f"Could not read CSV: {exc}")


elif page == "Model Monitoring":
    st.subheader("Model Monitoring")
    st.caption(
        "Saved evaluation reports — these are not live production "
        "monitoring unless new reports are generated."
    )

    missing_reports = []
    report_map = {
        "Operating policy": policy,
        "Held-out test metrics": test_report,
        "Validation monitoring": validation_report,
        "Temporal drift": drift_report,
    }

    for name, report in report_map.items():
        if report is None:
            missing_reports.append(name)

    if missing_reports:
        st.warning(
            "Some reports are missing or unreadable: "
            + ", ".join(missing_reports)
        )

    if policy:
        st.markdown("### Selected operating policy")

        threshold = find_value(
            policy, "threshold", "selected_threshold"
        )
        target_alert = find_value(
            policy, "max_alert_rate", "target_alert_rate"
        )
        target_recall = find_value(
            policy, "min_recall", "target_recall"
        )
        status = find_value(policy, "status", "policy_status")

        p1, p2, p3 = st.columns(3)
        with p1:
            show_metric(
                "Threshold",
                f"{float(threshold):.3f}"
                if threshold is not None else "N/A",
            )
        with p2:
            show_metric(
                "Target Alert Rate",
                format_percent(target_alert),
            )
        with p3:
            show_metric(
                "Minimum Recall",
                format_percent(target_recall),
            )

        if status:
            st.caption(f"Policy status: {status}")

    st.markdown("### Validation vs held-out test")

    def metric_value(report, *keys):
        value = find_value(report, *keys)
        return value if value is not None else None

    validation_rows = find_value(
        validation_report, "rows", "n_rows", "validation_rows"
    )
    test_rows = find_value(
        test_report, "rows", "n_rows", "test_rows"
    )

    validation_alert_rate = find_value(
        validation_report, "alert_rate"
    )
    if validation_alert_rate is None and validation_report:
        alerts = find_value(validation_report, "alerts", "alert_count")
        if alerts is not None and validation_rows:
            validation_alert_rate = alerts / validation_rows

    actual_test_alert_rate = find_value(test_report, "alert_rate")
    if actual_test_alert_rate is None and test_report:
        alerts = find_value(test_report, "alerts", "alert_count")
        if alerts is not None and test_rows:
            actual_test_alert_rate = alerts / test_rows

    rows = [
        {
            "Metric": "Rows",
            "Validation": validation_rows,
            "Held-out test": test_rows,
        },
        {
            "Metric": "Average precision",
            "Validation": metric_value(
                validation_report, "average_precision", "pr_auc", "ap"
            ),
            "Held-out test": metric_value(
                test_report, "average_precision", "pr_auc", "ap"
            ),
        },
        {
            "Metric": "Precision",
            "Validation": metric_value(validation_report, "precision"),
            "Held-out test": metric_value(test_report, "precision"),
        },
        {
            "Metric": "Recall",
            "Validation": metric_value(validation_report, "recall"),
            "Held-out test": metric_value(test_report, "recall"),
        },
        {
            "Metric": "F1",
            "Validation": metric_value(validation_report, "f1", "f1_score"),
            "Held-out test": metric_value(test_report, "f1", "f1_score"),
        },
        {
            "Metric": "Alert rate",
            "Validation": validation_alert_rate,
            "Held-out test": actual_test_alert_rate,
        },
        {
            "Metric": "False positives",
            "Validation": metric_value(validation_report, "fp", "false_positives"),
            "Held-out test": metric_value(test_report, "fp", "false_positives"),
        },
        {
            "Metric": "False negatives",
            "Validation": metric_value(validation_report, "fn", "false_negatives"),
            "Held-out test": metric_value(test_report, "fn", "false_negatives"),
        },
    ]

    comparison = pd.DataFrame(rows)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    chart_data = pd.DataFrame(
        {
            "Validation": {
                "Alert rate": validation_alert_rate,
                "Recall": metric_value(validation_report, "recall"),
            },
            "Held-out test": {
                "Alert rate": actual_test_alert_rate,
                "Recall": metric_value(test_report, "recall"),
            },
        }
    ).dropna(how="all")

    if not chart_data.empty:
        st.markdown("#### Alert rate and recall comparison")
        st.bar_chart(chart_data)

    # Compare held-out test performance with policy targets.
    target_alert = find_value(
        policy, "max_alert_rate", "target_alert_rate"
    ) if policy else None
    target_recall = find_value(
        policy, "min_recall", "target_recall"
    ) if policy else None
    test_recall = metric_value(test_report, "recall")

    if (
        target_alert is not None
        and actual_test_alert_rate is not None
        and actual_test_alert_rate > target_alert
    ):
        st.warning(
            "Held-out test alert rate exceeds the saved policy target. "
            "Validation constraints did not fully generalize to test."
        )

    if (
        target_recall is not None
        and test_recall is not None
        and test_recall < target_recall
    ):
        st.warning(
            "Held-out test recall is below the saved policy target."
        )

    st.markdown("### Temporal drift")

    if drift_report:
        psi_rows = extract_psi(drift_report)

        if psi_rows:
            psi_df = pd.DataFrame(psi_rows).drop_duplicates()
            st.dataframe(
                psi_df.sort_values("PSI", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "PSI is a descriptive drift indicator, not proof that "
                "model performance has degraded. Time/step drift may be "
                "expected when comparing different time windows."
            )
        else:
            st.info(
                "No numeric PSI fields were detected in the drift report. "
                "Check the report structure if drift values should be shown."
            )
    else:
        st.info("Temporal drift report is not available.")

    st.markdown("### Validation score distribution")

    if validation_report:
        score_fields = {
            "Mean": find_value(validation_report, "score_mean"),
            "Median": find_value(validation_report, "score_median"),
            "P90": find_value(validation_report, "score_p90"),
            "P95": find_value(validation_report, "score_p95"),
            "P99": find_value(validation_report, "score_p99"),
        }
        score_fields = {
            key: float(value)
            for key, value in score_fields.items()
            if value is not None
        }

        if score_fields:
            st.bar_chart(
                pd.DataFrame.from_dict(
                    score_fields, orient="index", columns=["Score"]
                )
            )
        else:
            st.info("Score distribution fields are not in this report.")

    st.divider()
    st.caption(
        "Research prototype only. The dataset is synthetic, costs are "
        "illustrative, and scores are not calibrated real-world fraud "
        "probabilities. Decisions require human review."
    )
