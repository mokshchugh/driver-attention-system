from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analytics.metrics import build_driver_metrics
from analytics.queries import get_event_timeline
# MODIFIED: auth now lives in db.accounts; bcrypt moved there too
from db.accounts import create_account, get_account_by_email, verify_password
from db.drivers import create_driver_with_email, get_driver, get_driver_by_email

import os

IS_CLOUD = os.getenv("STREAMLIT_CLOUD", "false") == "true"
start_calibration = None
if not IS_CLOUD:
    from services.calibration import start_calibration


EVENT_LABELS = {
    "PHONE_DISTRACTION_EVENT": "Phone distractions",
    "DROWSINESS_EVENT":        "Drowsiness events",
    "MICROSLEEP_EVENT":        "Microsleep events",
    "GAZE_AWAY_EVENT":         "Gaze-away events",
}


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────

def init_session_state():
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("current_user_email", None)
    st.session_state.setdefault("current_user_name", None)
    st.session_state.setdefault("current_driver_id", None)


# ─────────────────────────────────────────────
# Auth helpers  (MODIFIED: all backed by Postgres now)
# ─────────────────────────────────────────────

def _resolve_driver(email: str, name: str) -> int | None:
    """Return existing driver_id for this email, or create one."""
    driver = get_driver_by_email(email)
    if driver is None:
        driver = create_driver_with_email(name, email)
    return driver["driver_id"] if driver else None


def do_login(email: str, password: str) -> str | None:
    """Returns error string on failure, None on success."""
    account = get_account_by_email(email)
    if account is None:
        return "No account found with that email."
    if not verify_password(password, account["password_hash"]):
        return "Incorrect password."

    st.session_state["authenticated"]      = True
    st.session_state["current_user_email"] = account["email"]
    st.session_state["current_user_name"]  = account["name"]
    st.session_state["current_driver_id"]  = account["driver_id"]
    return None


def do_signup(name: str, email: str, password: str) -> str | None:
    """Returns error string on failure, None on success."""
    if not name.strip():
        return "Name is required."
    if not email.strip() or "@" not in email:
        return "A valid email is required."
    if len(password) < 6:
        return "Password must be at least 6 characters."
    if get_account_by_email(email) is not None:
        return "An account with that email already exists."

    driver_id = _resolve_driver(email, name.strip())
    if driver_id is None:
        return "Failed to create driver profile. Please try again."

    account = create_account(name.strip(), email, password, driver_id)
    if account is None:
        return "Failed to create account. Please try again."

    st.session_state["authenticated"]      = True
    st.session_state["current_user_email"] = email
    st.session_state["current_user_name"]  = name.strip()
    st.session_state["current_driver_id"]  = driver_id
    return None


def do_logout():
    st.session_state["authenticated"]      = False
    st.session_state["current_user_email"] = None
    st.session_state["current_user_name"]  = None
    st.session_state["current_driver_id"]  = None


# ─────────────────────────────────────────────
# Login / Signup UI  (unchanged)
# ─────────────────────────────────────────────

def render_auth_page():
    st.title("Driver Attention Dashboard")
    tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

    with tab_login:
        st.subheader("Welcome back")
        with st.form("login_form"):
            email    = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", use_container_width=True)
        if submitted:
            err = do_login(email.strip().lower(), password)
            if err:
                st.error(err)
            else:
                st.rerun()

    with tab_signup:
        st.subheader("Create an account")
        with st.form("signup_form"):
            name         = st.text_input("Full name")
            email_su     = st.text_input("Email", placeholder="you@example.com")
            password_su  = st.text_input("Password", type="password")
            password_su2 = st.text_input("Confirm password", type="password")
            submitted_su = st.form_submit_button("Create account", use_container_width=True)
        if submitted_su:
            if password_su != password_su2:
                st.error("Passwords do not match.")
            else:
                err = do_signup(name, email_su.strip().lower(), password_su)
                if err:
                    st.error(err)
                else:
                    st.success("Account created! Loading dashboard…")
                    st.rerun()


# ─────────────────────────────────────────────
# Dashboard  (unchanged)
# ─────────────────────────────────────────────

def main():
    init_session_state()

    st.set_page_config(
        page_title="Driver Attention Dashboard",
        layout="centered" if not st.session_state["authenticated"] else "wide",
    )

    if not st.session_state["authenticated"]:
        render_auth_page()
        return

    st.title("Driver Attention Dashboard")

    with st.sidebar:
        st.markdown(f"Logged in as **{st.session_state['current_user_name']}**")
        if st.button("Log out", use_container_width=True):
            do_logout()
            st.rerun()
        st.divider()

    driver_id = st.session_state["current_driver_id"]

    if driver_id is None:
        st.error("No driver profile is linked to your account. Please contact support.")
        return

    driver = get_driver(driver_id)
    if driver is None:
        st.error("Driver profile could not be loaded from the database.")
        return

    render_calibration_sidebar(driver_id)
    render_driver_baseline(driver)

    metrics     = build_driver_metrics(driver_id)
    timeline_df = get_event_timeline(driver_id)

    render_current_risk(metrics["latest_risk"])
    render_risk_chart(metrics["risk_timeseries"])
    render_event_timeline(timeline_df)
    render_session_stats(metrics["event_counts"], metrics["sessions"], metrics["max_risk"])

def render_calibration_sidebar(driver_id: int):
    st.sidebar.header("Calibration")
    if IS_CLOUD:
        st.sidebar.caption("⚠️ Calibration must be run locally using `python src/main.py`.")
        return
    if st.sidebar.button("Start calibration", use_container_width=True):
        with st.spinner("Capturing baseline data from the camera..."):
            try:
                result = start_calibration(driver_id)
                st.sidebar.success(
                    f"Calibration saved "
                    f"(EAR={result['baseline_ear']:.3f}, Yaw={result['baseline_yaw']:.2f})"
                )
                st.rerun()
            except Exception as exc:
                st.sidebar.error(str(exc))


# ─────────────────────────────────────────────
# Render helpers  (all unchanged)
# ─────────────────────────────────────────────

def render_driver_baseline(driver):
    col1, col2, col3 = st.columns(3)
    col1.metric("Driver", driver["name"])
    col2.metric(
        "Baseline EAR",
        f"{driver['baseline_ear']:.3f}" if driver["baseline_ear"] is not None else "Not calibrated",
    )
    col3.metric(
        "Baseline Yaw",
        f"{driver['baseline_yaw']:.2f}" if driver["baseline_yaw"] is not None else "Not calibrated",
    )


def render_current_risk(latest_risk):
    if latest_risk >= 60:
        color, label = "#d62728", "High"
    elif latest_risk >= 30:
        color, label = "#ff7f0e", "Elevated"
    else:
        color, label = "#2ca02c", "Stable"

    st.subheader("Current Risk Display")
    st.markdown(
        f"""
        <div style="padding:1rem;border-radius:0.75rem;background-color:{color};color:white;">
            <div style="font-size:0.9rem;opacity:0.9;">Current state</div>
            <div style="font-size:2rem;font-weight:700;">{latest_risk:.1f}</div>
            <div style="font-size:1rem;">{label} risk</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_chart(risk_timeseries):
    st.subheader("Risk Over Time")
    if risk_timeseries.empty:
        st.info("No risk data available for this driver yet.")
        return

    chart_df = risk_timeseries.copy()
    chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])
    fig = px.line(
        chart_df, x="timestamp", y="risk_score", color="session_id",
        markers=True, title="Risk score over time",
    )
    fig.update_layout(xaxis_title="Timestamp", yaxis_title="Risk score")
    st.plotly_chart(fig, use_container_width=True)


def render_event_timeline(timeline_df):
    st.subheader("Event Timeline")
    if timeline_df.empty:
        st.info("No events have been logged for this driver yet.")
        return

    view_df = timeline_df.copy()
    view_df["timestamp"] = pd.to_datetime(view_df["timestamp"])
    st.dataframe(view_df, use_container_width=True, hide_index=True)


def render_session_stats(event_counts, sessions, max_risk_df):
    st.subheader("Session Statistics")

    event_lookup = {}
    if not event_counts.empty:
        event_lookup = dict(zip(event_counts["event_type"], event_counts["event_count"]))

    highest_risk = float(max_risk_df["max_risk"].iloc[0]) if not max_risk_df.empty else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Phone",        int(event_lookup.get("PHONE_DISTRACTION_EVENT", 0)))
    col2.metric("Drowsiness",   int(event_lookup.get("DROWSINESS_EVENT", 0)))
    col3.metric("Microsleep",   int(event_lookup.get("MICROSLEEP_EVENT", 0)))
    col4.metric("Gaze away",    int(event_lookup.get("GAZE_AWAY_EVENT", 0)))
    col5.metric("Highest risk", f"{highest_risk:.1f}")

    st.caption(f"Recorded sessions: {len(sessions)}")


if __name__ == "__main__":
    main()
