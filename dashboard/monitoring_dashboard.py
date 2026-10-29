
import json
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

POLICY_PATH = REPORTS / "operating_policy.json"
TEST_PATH = REPORTS / "final_test_metrics.json"
MONITORING_PATH = REPORTS / "validation_monitoring_report.json"
DRIFT_PATH = REPORTS / "temporal_drift_report.json"


st.set_page_config(
    page_title="Fraud Model Monitoring",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0fdf4, #ffffff);
    }

    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
    }

    .hero {
        background: linear-gradient(120deg, #064e3b, #047857);
        color: white;
        padding: 1.4rem;
        border-radius: 16px;
        margin-bottom: 1.4rem;
    }

    .hero h1 {
        color: white;
    }

    .hero p {
        color: #d1fae5;
        margin-bottom: 0;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #d1fae5;
        border-radius: 12px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(path):
    """Load a JSON report, returning None if unavailable or invalid."""
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def get_value(data, *keys, default=None):
    """Safely find the first available key in a report."""
    if not isinstance(data, dict):
        return default

    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return default


def show_report_status(name, path, data):
    if data is None:
        st.warning(f"{name} unavailable: {path.name}")
    else:
        st.caption(f"{name}: loaded from `{path.relative_to(ROOT)}`")


st.markdown(
    """
    <div class="hero">
        <h1>📊 Fraud Model Monitoring</h1>
        <p>Performance, operational policy, and data drift review</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Historical report viewer only. These reports are not a live production "
    "feed. Metrics reflect the saved evaluation snapshots."
)

policy = load_json(POLICY_PATH)
test = load_json(TEST_PATH)
monitoring = load_json(MONITORING_PATH)
drift = load_json(DRIFT_PATH)

with st.expander("Report availability", expanded=False):
    show_report_status("Operating policy", POLICY_PATH, policy)
    show_report_status("Held-out test", TEST_PATH, test)
    show_report_status("Validation monitoring", MONITORING_PATH, monitoring)
    show_report_status("Temporal drift", DRIFT_PATH, drift)


# --------------------------------------------------
# OPERATING POLICY
# --------------------------------------------------

st.subheader("1. Operating policy")

threshold = get_value(
    policy,
    "threshold",
    "selected_threshold",
    default=0.87,
)
target_alert_rate = get_value(
    policy,
    "max_alert_rate",
    default=0.03,
)
target_recall = get_value(
    policy,
    "min_recall",
    default=0.90,
)

if isinstance(target_alert_rate, (int, float)) and target_alert_rate > 1:
    target_alert_rate /= 100

if isinstance(target_recall, (int, float)) and target_recall > 1:
    target_recall /= 100

p1, p2, p3 = st.columns(3)
p1.metric("Selected threshold", f"{threshold:.3f}")
p2.metric("Target alert rate", f"{target_alert_rate:.1%}")
p3.metric("Target minimum recall", f"{target_recall:.1%}")

st.caption(
    "Threshold is the validation-selected operating point. "
    "It should not be changed based on held-out test results."
)


# --------------------------------------------------
# VALIDATION VS TEST
# --------------------------------------------------

st.subheader("2. Validation vs held-out test")

validation_alert = get_value(
    monitoring,
    "alert_rate",
    default=None,
)
validation_recall = get_value(
    monitoring,
    "recall",
    default=None,
)
validation_precision = get_value(
    monitoring,
    "precision",
    default=None,
)
validation_ap = get_value(
    monitoring,
    "average_precision",
    "avg_precision",
    "ap",
    default=None,
)

test_alert = get_value(test, "alert_rate", default=None)
test_recall = get_value(test, "recall", default=None)
test_precision = get_value(test, "precision", default=None)
test_ap = get_value(
    test,
    "average_precision",
    "avg_precision",
    "ap",
    default=None,
)

# If older reports do not store alert_rate, derive it from counts.
if test_alert is None:
    alerts = get_value(test, "alerts", default=None)
    rows = get_value(test, "rows", "test_rows", default=None)
    if alerts is not None and rows:
        test_alert = alerts / rows

if validation_alert is None:
    alerts = get_value(monitoring, "alerts", default=None)
    rows = get_value(monitoring, "rows", default=None)
    if alerts is not None and rows:
        validation_alert = alerts / rows

metrics = pd.DataFrame(
    [
        {
            "Dataset": "Validation",
            "Alert rate": validation_alert,
            "Precision": validation_precision,
            "Recall": validation_recall,
            "Average precision": validation_ap,
        },
        {
            "Dataset": "Held-out test",
            "Alert rate": test_alert,
            "Precision": test_precision,
            "Recall": test_recall,
            "Average precision": test_ap,
        },
    ]
)

st.dataframe(
    metrics,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Alert rate": st.column_config.NumberColumn(
            format="percent",
        ),
        "Precision": st.column_config.NumberColumn(
            format="percent",
        ),
        "Recall": st.column_config.NumberColumn(
            format="percent",
        ),
        "Average precision": st.column_config.NumberColumn(
            format="percent",
        ),
    },
)

if validation_alert is not None and test_alert is not None:
    c1, c2 = st.columns(2)

    with c1:
        st.write("Alert rate by evaluation period")
        chart_data = pd.DataFrame(
            {
                "Alert rate": [validation_alert, test_alert],
            },
            index=["Validation", "Held-out test"],
        )
        st.bar_chart(chart_data)

    with c2:
        st.write("Recall by evaluation period")
        if validation_recall is not None and test_recall is not None:
            recall_data = pd.DataFrame(
                {
                    "Recall": [validation_recall, test_recall],
                },
                index=["Validation", "Held-out test"],
            )
            st.bar_chart(recall_data)

    if test_alert > target_alert_rate:
        st.error(
            f"Test alert rate ({test_alert:.2%}) exceeds the configured "
            f"target ({target_alert_rate:.2%})."
        )
    else:
        st.success("Held-out test alert rate is within the configured target.")

if test_recall is not None:
    if test_recall >= target_recall:
        st.success("Held-out test recall meets the configured target.")
    else:
        st.warning("Held-out test recall is below the configured target.")


# --------------------------------------------------
# DATA DRIFT
# --------------------------------------------------

st.subheader("3. Temporal drift")

if drift is None:
    st.warning(
        "Drift report not found. Run scripts/analyze_temporal_drift.py "
        "to generate it."
    )
else:
    st.write(
        "PSI (Population Stability Index) measures how feature distributions "
        "differ between periods. Interpret it alongside context and data quality."
    )

    st.caption(
        "The time-step feature is expected to change across chronological "
        "splits. High PSI for time alone does not prove model degradation."
    )

    def collect_psi(report):
        """Support common report layouts without assuming one schema."""
        found = []

        def walk(value, path=""):
            if isinstance(value, dict):
                for key, item in value.items():
                    key_path = f"{path}.{key}" if path else key
                    normalized = key.lower().replace("-", "_")

                    if "psi" in normalized and isinstance(item, (int, float)):
                        found.append(
                            {"Metric": key_path, "PSI": float(item)}
                        )
                    else:
                        walk(item, key_path)

            elif isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, f"{path}[{index}]")

        walk(report)
        return found

    psi_rows = collect_psi(drift)

    if psi_rows:
        psi_df = pd.DataFrame(psi_rows).drop_duplicates()
        st.dataframe(psi_df, use_container_width=True, hide_index=True)

        non_time = psi_df[
            ~psi_df["Metric"].str.lower().str.contains(
                r"step|time",
                regex=True,
            )
        ]

        if not non_time.empty:
            highest = non_time.loc[non_time["PSI"].idxmax()]
            st.metric(
                "Largest non-time PSI",
                f"{highest['PSI']:.4f}",
            )
            st.caption(f"Feature/report field: {highest['Metric']}")
    else:
        st.warning(
            "The report's PSI field names were not recognized automatically. "
            "Inspect reports/temporal_drift_report.json to map its schema."
        )


# --------------------------------------------------
# SCORE DISTRIBUTION
# --------------------------------------------------

st.subheader("4. Validation score distribution")

if monitoring is not None:
    score_fields = {
        "Mean": ["score_mean"],
        "Median": ["score_median"],
        "90th percentile": ["score_p90"],
        "95th percentile": ["score_p95"],
        "99th percentile": ["score_p99"],
    }

    score_values = {
        label: get_value(monitoring, *keys, default=None)
        for label, keys in score_fields.items()
    }

    available = {
        key: value
        for key, value in score_values.items()
        if isinstance(value, (int, float))
    }

    if available:
        st.bar_chart(
            pd.DataFrame(
                {"Score": available}
            )
        )
    else:
        st.warning("Score distribution metrics were not found in the report.")
else:
    st.warning("Validation monitoring report is unavailable.")


# --------------------------------------------------
# LIMITATIONS
# --------------------------------------------------

st.subheader("5. Model limitations")

st.markdown(
    """
    - The dataset is synthetic, not real banking data.
    - Fraud scores are not calibrated real-world probabilities.
    - The 3% alert-rate target did not generalize to the held-out test period.
    - Feature drift and fraud prevalence changes need separate investigation.
    - REVIEW is a screening recommendation, not proof of fraud.
    - No live transaction stream or continuous production labels are connected.
    """
)

st.caption(
    "Project status: research/portfolio prototype. "
    "Do not use as the sole basis for real customer transaction decisions."
)
