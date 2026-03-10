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
from db.drivers import create_driver, get_all_drivers, get_driver
from services.calibration import start_calibration


EVENT_LABELS = {
    "PHONE_DISTRACTION_EVENT": "Phone distractions",
    "DROWSINESS_EVENT": "Drowsiness events",
    "MICROSLEEP_EVENT": "Microsleep events",
    "GAZE_AWAY_EVENT": "Gaze-away events",
}


def main():
    st.set_page_config(page_title="Driver Attention Dashboard", layout="wide")
    st.title("Driver Attention Dashboard")

    drivers = get_all_drivers()
    selected_driver = render_driver_panel(drivers)

    if selected_driver is None:
        st.info("Create a driver profile to start calibration and analytics.")
        return

    driver = get_driver(selected_driver)
    if driver is None:
        st.error("Selected driver could not be loaded.")
        return

    render_driver_baseline(driver)

    metrics = build_driver_metrics(selected_driver)
    timeline_df = get_event_timeline(selected_driver)

    render_current_risk(metrics["latest_risk"])
    render_risk_chart(metrics["risk_timeseries"])
    render_event_timeline(timeline_df)
    render_session_stats(metrics["event_counts"], metrics["sessions"], metrics["max_risk"])


def render_driver_panel(drivers):
    st.sidebar.header("Driver Profile Panel")

    with st.sidebar.form("create_driver_form", clear_on_submit=True):
        new_driver_name = st.text_input("Driver name")
        submitted = st.form_submit_button("Create driver")
        if submitted:
            if new_driver_name.strip():
                created_driver = create_driver(new_driver_name.strip())
                st.sidebar.success(
                    f"Created driver #{created_driver['driver_id']} - {created_driver['name']}"
                )
                st.rerun()
            else:
                st.sidebar.error("Driver name is required.")

    if not drivers:
        return None

    driver_options = {
        f"#{driver['driver_id']} - {driver['name']}": driver["driver_id"] for driver in drivers
    }
    selected_label = st.sidebar.selectbox("Select driver", list(driver_options.keys()))
    selected_driver_id = driver_options[selected_label]

    if st.sidebar.button("Start calibration", use_container_width=True):
        with st.spinner("Capturing baseline data from the camera..."):
            try:
                result = start_calibration(selected_driver_id)
                st.sidebar.success(
                    "Calibration saved "
                    f"(EAR={result['baseline_ear']:.3f}, Yaw={result['baseline_yaw']:.2f})"
                )
                st.rerun()
            except Exception as exc:
                st.sidebar.error(str(exc))

    return selected_driver_id


def render_driver_baseline(driver):
    baseline_ear = driver["baseline_ear"]
    baseline_yaw = driver["baseline_yaw"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Driver", driver["name"])
    col2.metric(
        "Baseline EAR",
        f"{baseline_ear:.3f}" if baseline_ear is not None else "Not calibrated",
    )
    col3.metric(
        "Baseline Yaw",
        f"{baseline_yaw:.2f}" if baseline_yaw is not None else "Not calibrated",
    )


def render_current_risk(latest_risk):
    if latest_risk >= 60:
        color = "#d62728"
        label = "High"
    elif latest_risk >= 30:
        color = "#ff7f0e"
        label = "Elevated"
    else:
        color = "#2ca02c"
        label = "Stable"

    st.subheader("Current Risk Display")
    st.markdown(
        f"""
        <div style="padding: 1rem; border-radius: 0.75rem; background-color: {color}; color: white;">
            <div style="font-size: 0.9rem; opacity: 0.9;">Current state</div>
            <div style="font-size: 2rem; font-weight: 700;">{latest_risk:.1f}</div>
            <div style="font-size: 1rem;">{label} risk</div>
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
        chart_df,
        x="timestamp",
        y="risk_score",
        color="session_id",
        markers=True,
        title="Risk score over time",
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

    highest_risk = (
        float(max_risk_df["max_risk"].iloc[0]) if not max_risk_df.empty else 0.0
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(
        "Phone",
        int(event_lookup.get("PHONE_DISTRACTION_EVENT", 0)),
    )
    col2.metric(
        "Drowsiness",
        int(event_lookup.get("DROWSINESS_EVENT", 0)),
    )
    col3.metric(
        "Microsleep",
        int(event_lookup.get("MICROSLEEP_EVENT", 0)),
    )
    col4.metric(
        "Gaze away",
        int(event_lookup.get("GAZE_AWAY_EVENT", 0)),
    )
    col5.metric("Highest risk", f"{highest_risk:.1f}")

    st.caption(f"Recorded sessions: {len(sessions)}")


if __name__ == "__main__":
    main()
