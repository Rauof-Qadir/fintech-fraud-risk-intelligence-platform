from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_BATCH_ROWS = 5_000

REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
]
TRANSACTION_TYPES = ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"]

st.set_page_config(
    page_title="FinSight | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Visual system — calm finance dashboard, strong hierarchy, responsive layout
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #14251d;
        --muted: #718078;
        --green: #176b4d;
        --green-dark: #104832;
        --green-soft: #e8f4ed;
        --line: #e4ebe6;
        --surface: #ffffff;
        --canvas: #f5f8f6;
        --amber: #a85b13;
        --red: #b42332;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 82% 0%, rgba(209, 235, 218, .45), transparent 32%),
            var(--canvas);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(245, 248, 246, .88);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #123f2e 0%, #0d3023 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }
    [data-testid="stSidebar"] * {
        color: #f4fbf6;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 10px;
        padding: .45rem .55rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(255,255,255,.09);
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,.16);
        margin: 1.15rem 0;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #b9d3c4;
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--ink);
        letter-spacing: -.035em;
    }
    h1 { font-weight: 800 !important; }
    h2, h3 { font-weight: 750 !important; }
    p, li, label { color: #35473d; }

    .brand-mark {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: .1rem 0 .2rem 0;
    }
    .brand-icon {
        width: 42px;
        height: 42px;
        display: grid;
        place-items: center;
        background: rgba(255,255,255,.12);
        border: 1px solid rgba(255,255,255,.2);
        border-radius: 14px;
        font-size: 23px;
    }
    .brand-name {
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 1.2rem;
        letter-spacing: -.04em;
        color: white;
    }
    .brand-subtitle {
        color: #b9d3c4;
        font-size: .76rem;
        margin-top: -2px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(118deg, #123f2e 0%, #176b4d 63%, #27835f 100%);
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 24px;
        padding: 28px 30px;
        color: #fff;
        margin-bottom: 22px;
        box-shadow: 0 14px 35px rgba(18,63,46,.12);
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -85px;
        top: -125px;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,.14);
        box-shadow: 0 0 0 28px rgba(255,255,255,.035),
                    0 0 0 58px rgba(255,255,255,.025);
    }
    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: .15em;
        font-weight: 700;
        font-size: .72rem;
        color: #bfe6cf;
        margin-bottom: 9px;
    }
    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.7rem, 3vw, 2.5rem);
        line-height: 1.12;
        font-weight: 800;
        letter-spacing: -.055em;
        color: #fff;
        max-width: 760px;
        margin: 0;
    }
    .hero-copy {
        color: #d8eee0;
        font-size: .96rem;
        max-width: 700px;
        line-height: 1.65;
        margin-top: 12px;
        margin-bottom: 0;
    }
    .hero-chip {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        margin-top: 18px;
        padding: 7px 11px;
        background: rgba(255,255,255,.12);
        border: 1px solid rgba(255,255,255,.2);
        border-radius: 999px;
        color: #f1fff5;
        font-size: .76rem;
        font-weight: 600;
    }

    .section-heading {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 1.2rem 0 .75rem 0;
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: -.035em;
    }
    .section-heading span {
        width: 7px;
        height: 24px;
        background: #2b8a62;
        border-radius: 99px;
        display: inline-block;
    }
    .muted-copy {
        color: var(--muted);
        font-size: .88rem;
        line-height: 1.6;
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 17px 18px;
        box-shadow: 0 3px 12px rgba(22, 54, 37, .025);
        min-height: 122px;
    }
    div[data-testid="stMetricLabel"] p {
        color: #718078 !important;
        font-size: .79rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        letter-spacing: -.04em;
    }
    div[data-testid="stMetricDelta"] {
        font-size: .75rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 17px;
    }
    [data-testid="stForm"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 20px;
    }
    [data-testid="stFileUploader"] section {
        background: #fbfdfb;
        border: 1px dashed #a8cbb5;
        border-radius: 15px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: 999px;
        padding: 7px 11px;
        font-size: .78rem;
        font-weight: 700;
        border: 1px solid #c9e7d4;
        background: #eaf7ef;
        color: #176b4d;
    }
    .status-dot {
        height: 7px;
        width: 7px;
        border-radius: 50%;
        background: #1f9d61;
        display: inline-block;
    }
    .status-pill.offline {
        background: #fff3e9;
        border-color: #f3d2b7;
        color: #98500f;
    }
    .status-pill.offline .status-dot { background: #d97706; }

    .feature-card {
        height: 100%;
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 17px;
        padding: 19px 20px;
        box-shadow: 0 3px 12px rgba(22,54,37,.025);
    }
    .feature-icon {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        background: var(--green-soft);
        border-radius: 13px;
        font-size: 20px;
        margin-bottom: 13px;
    }
    .feature-title {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 1rem;
        margin-bottom: 6px;
    }
    .feature-copy {
        color: var(--muted);
        font-size: .86rem;
        line-height: 1.55;
    }
    .notice {
        background: #fff8ed;
        border: 1px solid #f4dfbc;
        border-radius: 14px;
        padding: 14px 16px;
        color: #80501a;
        font-size: .86rem;
        line-height: 1.55;
    }
    .footer-note {
        color: #7c8981;
        font-size: .76rem;
        line-height: 1.6;
        border-top: 1px solid var(--line);
        padding-top: 16px;
        margin-top: 30px;
    }

    .stButton > button, .stFormSubmitButton > button,
    [data-testid="stDownloadButton"] > button {
        border-radius: 11px;
        min-height: 43px;
        font-weight: 700;
        transition: all .18s ease;
    }
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] {
        background: #176b4d;
        border: 1px solid #176b4d;
    }
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button[kind="primary"]:hover {
        background: #104832;
        border-color: #104832;
        transform: translateY(-1px);
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 10px;
    }
    [data-testid="stAlert"] {
        border-radius: 12px;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    @media (max-width: 800px) {
        .block-container { padding-top: 1rem; padding-left: 1rem; padding-right: 1rem; }
        .hero { padding: 23px 20px; border-radius: 18px; }
        div[data-testid="stMetric"] { padding: 13px; min-height: 105px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def load_json(filename: str):
    path = REPORTS_DIR / filename
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return None


def find_value(data, *keys, default=None):
    """Return the first non-null matching key from a report."""
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def as_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(value, digits=2):
    number = as_float(value)
    return "N/A" if number is None else f"{number * 100:.{digits}f}%"


def number(value, digits=3):
    number_value = as_float(value)
    return "N/A" if number_value is None else f"{number_value:,.{digits}f}"


def count_text(value):
    try:
        return f"{int(value):,}" if value is not None else "N/A"
    except (TypeError, ValueError, OverflowError):
        return "N/A"


def metric(label, value, help_text=None, delta=None):
    st.metric(label, value, delta=delta, help=help_text)


def section(title: str, subtitle: str | None = None):
    st.markdown(
        f'<div class="section-heading"><span></span>{title}</div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.markdown(f'<div class="muted-copy">{subtitle}</div>', unsafe_allow_html=True)


def api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


def extract_psi(data, path=""):
    """Find numeric PSI values in a nested drift-report structure."""
    found = []
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path} / {key}" if path else str(key)
            if "psi" in str(key).lower():
                if isinstance(value, (int, float)):
                    found.append({"Feature / comparison": current_path, "PSI": float(value)})
                elif isinstance(value, (dict, list)):
                    found.extend(extract_psi(value, current_path))
            elif isinstance(value, (dict, list)):
                found.extend(extract_psi(value, current_path))
    elif isinstance(data, list):
        for index, value in enumerate(data):
            found.extend(extract_psi(value, f"{path}[{index}]"))
    return found


def render_hero(kicker, title, copy, chip=None):
    chip_html = f'<div class="hero-chip">✦ &nbsp;{chip}</div>' if chip else ""
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">{kicker}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-copy">{copy}</p>
            {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <div class="footer-note">
          <b>FinSight · Research prototype</b><br>
          Built with Streamlit and FastAPI. Evaluation uses synthetic PaySim-style data;
          illustrative costs and model scores must not be treated as real-world banking
          outcomes or calibrated fraud probabilities. Human review remains essential.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Load saved artifacts and check service
# ---------------------------------------------------------------------------
policy = load_json("operating_policy.json")
test_report = load_json("final_test_metrics.json")
validation_report = load_json("validation_monitoring_report.json")
drift_report = load_json("temporal_drift_report.json")
health, health_error = api_health()
api_online = bool(health and health.get("status") == "ok")

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand-mark">
          <div class="brand-icon">🛡️</div>
          <div>
            <div class="brand-name">FinSight</div>
            <div class="brand-subtitle">Fraud intelligence</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("WORKSPACE")
    page = st.radio(
        "Navigate",
        ["Executive Overview", "Transaction Scoring", "Batch Analysis", "Model Monitoring"],
        label_visibility="collapsed",
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("SYSTEM STATUS")
    status_class = "" if api_online else "offline"
    status_label = "Prediction API online" if api_online else "Prediction API offline"
    st.markdown(
        f'<div class="status-pill {status_class}"><span class="status-dot"></span>{status_label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("ENVIRONMENT")
    st.caption("Synthetic PaySim-style dataset")
    st.caption("Portfolio / research prototype")
    with st.expander("Connection details"):
        st.code(API_URL, language="text")
        if not api_online and health_error:
            st.caption(health_error)

# ---------------------------------------------------------------------------
# Page 1 — Executive overview
# ---------------------------------------------------------------------------
if page == "Executive Overview":
    render_hero(
        "RISK OPERATIONS WORKSPACE",
        "Fraud signals. Clear decisions.",
        "Explore transaction risk scores, review suspicious activity, and understand how "
        "the model behaves across validation and held-out test data.",
        "Decision support · Human-in-the-loop",
    )

    if api_online:
        st.success("Prediction service is connected and responding.", icon="🟢")
    else:
        st.warning(
            "The prediction service is currently unreachable. You can still inspect saved "
            "evaluation reports; scoring features need the API to be online.",
            icon="⚠️",
        )

    threshold = find_value(policy, "threshold", "selected_threshold", default=0.87)
    test_ap = find_value(test_report, "average_precision", "pr_auc", "ap")
    test_recall = find_value(test_report, "recall")
    test_precision = find_value(test_report, "precision")
    test_alert_rate = find_value(test_report, "alert_rate")
    test_rows = find_value(test_report, "rows", "n_rows", "test_rows")

    if test_alert_rate is None and test_report:
        alerts = find_value(test_report, "alerts", "alert_count")
        if alerts is not None and test_rows:
            test_alert_rate = alerts / test_rows

    section("Model snapshot", "A compact view of the saved held-out test evaluation.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("Operating threshold", number(threshold), "The saved score cutoff used for review decisions.")
    with c2:
        metric("Test average precision", number(test_ap, 4), "Precision-recall area metric; useful for imbalanced classification.")
    with c3:
        metric("Test recall", pct(test_recall), "Share of known fraud cases detected in the held-out test set.")
    with c4:
        metric("Test alert rate", pct(test_alert_rate), "Share of transactions that would be sent for review.")

    st.markdown("")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        section("Your workflow", "Choose a workspace based on the task you need to complete.")
        st.markdown(
            """
            <div class="feature-card">
              <div class="feature-icon">🔎</div>
              <div class="feature-title">Transaction scoring</div>
              <div class="feature-copy">Evaluate one transaction, inspect its risk score, and see whether it crosses the review threshold.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            """
            <div class="feature-card">
              <div class="feature-icon">📄</div>
              <div class="feature-title">Batch analysis</div>
              <div class="feature-copy">Upload a small CSV, preview the data, run predictions, and export the scored results.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        section("Model monitoring", "Compare evaluation windows before relying on an operating policy.")
        st.markdown(
            """
            <div class="feature-card">
              <div class="feature-icon">📈</div>
              <div class="feature-title">Performance comparison</div>
              <div class="feature-copy">Review precision, recall, average precision, alert rate, and error counts across validation and test data.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            """
            <div class="notice">
              <b>Important context</b><br>
              This is a research prototype trained and evaluated with synthetic data.
              Good offline metrics do not guarantee performance in a real financial institution.
            </div>
            """,
            unsafe_allow_html=True,
        )

    section("Evaluation at a glance")
    if test_precision is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            metric("Test precision", number(test_precision, 4), "Share of flagged transactions that were fraud in the test set.")
        with c2:
            metric("Test rows", count_text(test_rows), "Number of held-out transactions used for this report.")
        with c3:
            metric("Policy status", str(find_value(policy, "status", "policy_status", default="Report unavailable")), "Status stored in the operating-policy report.")
    else:
        st.info("Some snapshot metrics are unavailable. Open Model Monitoring to inspect report availability.")

    render_footer()

# ---------------------------------------------------------------------------
# Page 2 — Single transaction scoring
# ---------------------------------------------------------------------------
elif page == "Transaction Scoring":
    render_hero(
        "TRANSACTION REVIEW",
        "Score a transaction",
        "Enter the fields available before the decision. The model returns a score and a "
        "review recommendation — not a definitive fraud verdict.",
        "Single transaction · Interactive analysis",
    )

    if not api_online:
        st.error("The prediction API is offline. Start the API service, then refresh this page.")
    else:
        left, right = st.columns([1.1, .9], gap="large")
        with left:
            section("Transaction details", "Use non-negative amounts and select the transaction type.")
            with st.form("single_prediction_form", border=False):
                step = st.number_input("Transaction step", min_value=1, value=100, step=1,
                                       help="Time-step index from the source dataset.")
                txn_type = st.selectbox("Transaction type", TRANSACTION_TYPES)
                amount = st.number_input("Transaction amount", min_value=0.0, value=1000.0,
                                         step=100.0, format="%.2f")
                origin_balance = st.number_input("Origin balance before transaction",
                                                 min_value=0.0, value=2000.0, step=100.0,
                                                 format="%.2f")
                destination_balance = st.number_input("Destination balance before transaction",
                                                      min_value=0.0, value=1000.0, step=100.0,
                                                      format="%.2f")
                submitted = st.form_submit_button("Analyze transaction  →", type="primary",
                                                  use_container_width=True)

        with right:
            section("Before you score", "A few reminders for responsible interpretation.")
            st.markdown(
                """
                <div class="feature-card">
                  <div class="feature-icon">🧭</div>
                  <div class="feature-title">What the score means</div>
                  <div class="feature-copy">A model output used to rank risk. It is not a calibrated real-world fraud probability.</div>
                  <hr style="border-color:#e4ebe6;margin:16px 0">
                  <div class="feature-title">What the decision means</div>
                  <div class="feature-copy"><b>REVIEW</b> means the score crossed the configured threshold. It does not prove misconduct.</div>
                  <hr style="border-color:#e4ebe6;margin:16px 0">
                  <div class="feature-title">What to do next</div>
                  <div class="feature-copy">Use supporting evidence and a human review process before taking action.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if submitted:
            payload = {
                "step": int(step),
                "type": txn_type,
                "amount": float(amount),
                "oldbalanceOrg": float(origin_balance),
                "oldbalanceDest": float(destination_balance),
            }
            with st.spinner("Scoring transaction…"):
                try:
                    response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
                    if response.ok:
                        result = response.json()
                        score = as_float(result.get("fraud_score"))
                        decision = str(result.get("decision", "Unknown")).upper()
                        threshold_value = as_float(result.get("threshold"))

                        st.markdown("")
                        section("Analysis result", "The result below reflects the current model and configured threshold.")
                        a, b, c = st.columns(3)
                        with a:
                            metric("Model score", number(score, 5), "Raw model score. It should not be interpreted as a calibrated probability.")
                        with b:
                            metric("Recommended action", decision, "The API's threshold-based decision.")
                        with c:
                            metric("Review threshold", number(threshold_value), "Score cutoff used by the API.")

                        if decision == "REVIEW":
                            st.warning("⚠️ This transaction crossed the review threshold. Investigate using independent evidence; this is not proof of fraud.")
                        else:
                            st.success("✓ This transaction is below the review threshold. That does not guarantee the transaction is safe.")

                        with st.expander("View complete API response"):
                            st.json(result)
                    else:
                        st.error(f"Prediction API returned HTTP {response.status_code}.")
                        st.code(response.text[:2000])
                except requests.RequestException as exc:
                    st.error(f"Could not reach the prediction API: {exc}")

    render_footer()

# ---------------------------------------------------------------------------
# Page 3 — Batch CSV analysis
# ---------------------------------------------------------------------------
elif page == "Batch Analysis":
    render_hero(
        "BATCH OPERATIONS",
        "Analyze a CSV in one pass",
        "Upload a small transaction file, validate its structure, preview the records, and "
        "download the scored output when the run finishes.",
        "CSV upload · Up to 5,000 rows",
    )

    section("1 · Prepare your file", "The file must contain the five fields below. Extra columns are allowed.")
    st.code(", ".join(REQUIRED_COLUMNS), language="text")
    st.markdown(
        '<div class="muted-copy">Upload limit: 5 MB · Batch limit: 5,000 rows · CSV format only</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Choose transaction CSV", type=["csv"], label_visibility="visible")

    if uploaded_file is None:
        st.markdown("")
        st.markdown(
            """
            <div class="feature-card">
              <div class="feature-icon">⬆️</div>
              <div class="feature-title">Ready when you are</div>
              <div class="feature-copy">Choose a CSV file to begin. You'll see a preview and any missing-column errors before predictions run.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif uploaded_file.size > MAX_UPLOAD_BYTES:
        st.error("This file is larger than the 5 MB upload limit. Please choose a smaller CSV.")
    else:
        try:
            preview_df = pd.read_csv(uploaded_file)
            missing = [column for column in REQUIRED_COLUMNS if column not in preview_df.columns]
            section("2 · Validate and preview")
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            elif preview_df.empty:
                st.error("The uploaded CSV has no transaction rows.")
            elif len(preview_df) > MAX_BATCH_ROWS:
                st.error(f"This file has {len(preview_df):,} rows. The maximum is {MAX_BATCH_ROWS:,}.")
            else:
                a, b, c = st.columns(3)
                with a:
                    metric("Rows detected", count_text(len(preview_df)))
                with b:
                    metric("Columns detected", count_text(len(preview_df.columns)))
                with c:
                    metric("Required fields", f"{len(REQUIRED_COLUMNS)}/{len(REQUIRED_COLUMNS)}", "All required columns are present.")
                st.dataframe(preview_df.head(8), use_container_width=True, hide_index=True)
                with st.expander("View column types and missing values"):
                    quality = pd.DataFrame({
                        "Column": preview_df.columns,
                        "Data type": [str(dtype) for dtype in preview_df.dtypes],
                        "Missing values": [int(value) for value in preview_df.isna().sum().values],
                    })
                    st.dataframe(quality, use_container_width=True, hide_index=True)

                section("3 · Run analysis")
                if not api_online:
                    st.warning("The API is offline. Start the prediction service before running this batch.")
                elif st.button("Run batch predictions  →", type="primary", use_container_width=True):
                    uploaded_file.seek(0)
                    with st.spinner(f"Scoring {len(preview_df):,} transactions…"):
                        try:
                            response = requests.post(
                                f"{API_URL}/predict/batch",
                                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                                timeout=120,
                            )
                            if response.ok:
                                results = pd.read_csv(BytesIO(response.content))
                                if results.empty:
                                    st.warning("The API returned an empty result file.")
                                else:
                                    alerts = int((results["decision"] == "REVIEW").sum()) if "decision" in results.columns else 0
                                    alert_rate = alerts / len(results) if len(results) else 0
                                    st.markdown("")
                                    section("Batch results", "Review the summary, inspect the scored rows, and export the complete output.")
                                    m1, m2, m3 = st.columns(3)
                                    with m1:
                                        metric("Transactions scored", count_text(len(results)))
                                    with m2:
                                        metric("Review alerts", count_text(alerts))
                                    with m3:
                                        metric("Alert rate", pct(alert_rate))
                                    if "decision" in results.columns:
                                        decision_counts = results["decision"].value_counts().rename_axis("Decision").to_frame("Transactions")
                                        chart_col, detail_col = st.columns([.85, 1.15], gap="large")
                                        with chart_col:
                                            section("Decision mix")
                                            st.bar_chart(decision_counts)
                                        with detail_col:
                                            section("Scored transaction preview")
                                            st.dataframe(results.head(15), use_container_width=True, hide_index=True)
                                    else:
                                        st.dataframe(results.head(50), use_container_width=True, hide_index=True)
                                    st.download_button(
                                        "Download full predictions CSV  ↓",
                                        data=results.to_csv(index=False).encode("utf-8"),
                                        file_name="finsight_fraud_predictions.csv",
                                        mime="text/csv",
                                        type="primary",
                                        use_container_width=True,
                                    )
                            else:
                                st.error(f"Batch API returned HTTP {response.status_code}.")
                                st.code(response.text[:2000])
                        except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
                            st.error(f"Batch analysis failed: {exc}")
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            st.error(f"Could not read this CSV: {exc}")

    render_footer()

# ---------------------------------------------------------------------------
# Page 4 — Model monitoring
# ---------------------------------------------------------------------------
elif page == "Model Monitoring":
    render_hero(
        "MODEL GOVERNANCE",
        "Monitor performance, not just scores",
        "Compare saved validation and held-out test reports, review the operating policy, "
        "and inspect temporal drift indicators.",
        "Saved reports · Offline evaluation",
    )
    st.info("These are saved evaluation artifacts, not live production monitoring. Refresh reports from your evaluation pipeline to update them.")

    report_map = {
        "Operating policy": policy,
        "Held-out test metrics": test_report,
        "Validation monitoring": validation_report,
        "Temporal drift": drift_report,
    }
    missing_reports = [name for name, report in report_map.items() if report is None]
    if missing_reports:
        st.warning("Some reports are missing or unreadable: " + ", ".join(missing_reports))

    section("Operating policy", "Current policy settings saved in reports/operating_policy.json.")
    if policy:
        threshold = find_value(policy, "threshold", "selected_threshold")
        target_alert = find_value(policy, "max_alert_rate", "target_alert_rate")
        target_recall = find_value(policy, "min_recall", "target_recall")
        status = find_value(policy, "status", "policy_status", default="Not specified")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            metric("Selected threshold", number(threshold))
        with p2:
            metric("Target alert rate", pct(target_alert))
        with p3:
            metric("Minimum recall target", pct(target_recall))
        with p4:
            metric("Policy status", str(status))
    else:
        st.info("Operating-policy report is unavailable.")

    def report_metric(report, *keys):
        return as_float(find_value(report, *keys))

    validation_rows = find_value(validation_report, "rows", "n_rows", "validation_rows")
    test_rows = find_value(test_report, "rows", "n_rows", "test_rows")

    validation_alert_rate = report_metric(validation_report, "alert_rate")
    if validation_alert_rate is None and validation_report:
        alerts = find_value(validation_report, "alerts", "alert_count")
        if alerts is not None and validation_rows:
            validation_alert_rate = alerts / validation_rows

    test_alert_rate = report_metric(test_report, "alert_rate")
    if test_alert_rate is None and test_report:
        alerts = find_value(test_report, "alerts", "alert_count")
        if alerts is not None and test_rows:
            test_alert_rate = alerts / test_rows

    validation_recall = report_metric(validation_report, "recall")
    test_recall = report_metric(test_report, "recall")
    validation_ap = report_metric(validation_report, "average_precision", "pr_auc", "ap")
    test_ap = report_metric(test_report, "average_precision", "pr_auc", "ap")

    section("Validation vs held-out test", "A time-separated test helps reveal whether validation performance generalizes.")
    comparison = pd.DataFrame([
        {"Metric": "Rows", "Validation": validation_rows, "Held-out test": test_rows},
        {"Metric": "Average precision", "Validation": validation_ap, "Held-out test": test_ap},
        {"Metric": "Precision", "Validation": report_metric(validation_report, "precision"), "Held-out test": report_metric(test_report, "precision")},
        {"Metric": "Recall", "Validation": validation_recall, "Held-out test": test_recall},
        {"Metric": "F1 score", "Validation": report_metric(validation_report, "f1", "f1_score"), "Held-out test": report_metric(test_report, "f1", "f1_score")},
        {"Metric": "Alert rate", "Validation": validation_alert_rate, "Held-out test": test_alert_rate},
        {"Metric": "False positives", "Validation": find_value(validation_report, "fp", "false_positives"), "Held-out test": find_value(test_report, "fp", "false_positives")},
        {"Metric": "False negatives", "Validation": find_value(validation_report, "fn", "false_negatives"), "Held-out test": find_value(test_report, "fn", "false_negatives")},
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    chart_data = pd.DataFrame({
        "Validation": {"Alert rate": validation_alert_rate, "Recall": validation_recall},
        "Held-out test": {"Alert rate": test_alert_rate, "Recall": test_recall},
    }).dropna(how="all")
    if not chart_data.empty:
        section("Operational trade-off", "Alert rate and recall are shown together for comparison; neither metric should be optimized in isolation.")
        st.bar_chart(chart_data)

    target_alert = find_value(policy, "max_alert_rate", "target_alert_rate") if policy else None
    target_recall = find_value(policy, "min_recall", "target_recall") if policy else None
    if target_alert is not None and test_alert_rate is not None and test_alert_rate > as_float(target_alert, float("inf")):
        st.warning("The held-out test alert rate exceeds the saved policy target. Validation constraints did not fully generalize to test.")
    if target_recall is not None and test_recall is not None and test_recall < as_float(target_recall, float("-inf")):
        st.warning("Held-out test recall is below the saved policy target.")

    section("Temporal drift", "Population Stability Index (PSI) values, when present in the saved report.")
    if drift_report:
        psi_rows = extract_psi(drift_report)
        if psi_rows:
            psi_df = pd.DataFrame(psi_rows).drop_duplicates()
            st.dataframe(psi_df.sort_values("PSI", ascending=False), use_container_width=True, hide_index=True)
            st.caption("PSI is a descriptive drift indicator, not proof of model degradation. Changes across time windows may be expected.")
        else:
            st.info("No numeric PSI fields were detected in the temporal drift report.")
    else:
        st.info("Temporal drift report is not available.")

    section("Validation score distribution", "Summary statistics for model scores, if stored in the validation report.")
    if validation_report:
        score_fields = {
            "Mean": find_value(validation_report, "score_mean"),
            "Median": find_value(validation_report, "score_median"),
            "P90": find_value(validation_report, "score_p90"),
            "P95": find_value(validation_report, "score_p95"),
            "P99": find_value(validation_report, "score_p99"),
        }
        score_fields = {key: as_float(value) for key, value in score_fields.items() if as_float(value) is not None}
        if score_fields:
            st.bar_chart(pd.DataFrame.from_dict(score_fields, orient="index", columns=["Score"]))
        else:
            st.info("Score distribution fields are not included in the validation report.")
    else:
        st.info("Validation monitoring report is not available.")

    with st.expander("Inspect report files"):
        for label, report in report_map.items():
            st.markdown(f"**{label}**")
            if report is not None:
                st.json(report)
            else:
                st.caption("Not available")

    render_footer()
