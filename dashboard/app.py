from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analytics.metrics import build_driver_metrics
from analytics.queries import get_event_timeline
from db.accounts import create_account, get_account_by_email, verify_password
from db.drivers import create_driver_with_email, get_driver, get_driver_by_email


EVENT_LABELS = {
    "PHONE_DISTRACTION_EVENT": "Phone distractions",
    "DROWSINESS_EVENT":        "Drowsiness events",
    "MICROSLEEP_EVENT":        "Microsleep events",
    "GAZE_AWAY_EVENT":         "Gaze-away events",
}

# ─────────────────────────────────────────────
# Theme CSS
# ─────────────────────────────────────────────

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-base:      #1a1f2e;
    --bg-sidebar:   #141820;
    --bg-card:      #1e2535;
    --bg-card-dark: #181e2c;
    --accent-blue:  #4a9eff;
    --accent-orange:#e8872a;
    --accent-red:   #e84040;
    --accent-teal:  #2dd4bf;
    --accent-purple:#a855f7;
    --text-primary: #e8edf5;
    --text-muted:   #8899aa;
    --text-label:   #aabbcc;
    --border:       #2a3a4a;
}

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"],
[data-testid="stToolbar"] { display: none !important; }

[data-testid="stMainBlockContainer"] {
    padding: 1.5rem 2rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding: 1.25rem 1rem !important; }
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebar"] .element-container p {
    font-size: 0.78rem !important;
    color: var(--text-muted) !important;
    margin: 0 !important;
    line-height: 1.4 !important;
}
[data-testid="stSidebar"] strong {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    margin: 1rem 0 0.5rem 0 !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border) !important; margin: 0.85rem 0 !important; }
[data-testid="stSidebar"] .stButton > button {
    background-color: #252e40 !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    width: 100% !important;
    transition: background 0.18s, border-color 0.18s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #2e3a50 !important;
    border-color: var(--accent-blue) !important;
}

/* ── Page title ── */
h1 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: clamp(1.4rem, 2.5vw, 1.9rem) !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 1.1rem !important;
}
h2, h3 { font-family: 'Inter', sans-serif !important; color: var(--text-primary) !important; }

/* ── Baseline metric cards ── */
[data-testid="stMetric"] {
    background-color: var(--bg-card-dark) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.85rem 1rem !important;
    min-width: 0 !important;
}
[data-testid="stMetricLabel"] > div {
    color: var(--text-label) !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    margin-bottom: 0.25rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent-blue) !important;
    font-size: clamp(1rem, 2vw, 1.35rem) !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stMetricDelta"] { display: none !important; }

[data-testid="stHorizontalBlock"] { gap: 0.75rem !important; flex-wrap: wrap !important; }
[data-testid="stColumn"] { min-width: 0 !important; }

/* ── Auth form container ── */
[data-testid="stForm"] {
    background-color: var(--bg-card-dark) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
}

/* ── Text inputs ── */
.stTextInput > div > div > input {
    background-color: #252e40 !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 0.75rem !important;
}
.stTextInput > div > div > input::placeholder { color: #556677 !important; }
.stTextInput > div > div > input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(74,158,255,0.15) !important;
    outline: none !important;
}
.stTextInput label {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Form submit button — override the default white ── */
[data-testid="stForm"] .stButton > button {
    background-color: var(--bg-card-dark) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.5rem !important;
    width: 100% !important;
    transition: background 0.18s, color 0.18s, border-color 0.18s !important;
}
[data-testid="stForm"] .stButton > button:hover {
    background-color: var(--accent-blue) !important;
    color: #fff !important;
    border-color: var(--accent-blue) !important;
}
/* Streamlit internal primaryFormSubmit override */
button[kind="primaryFormSubmit"] {
    background-color: var(--bg-card-dark) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
}
button[kind="primaryFormSubmit"]:hover {
    background-color: var(--accent-blue) !important;
    color: #fff !important;
    border-color: var(--accent-blue) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    padding: 0.6rem 1.5rem !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent-blue) !important;
    border-bottom-color: var(--accent-blue) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.83rem !important;
}

/* ── Plotly transparent ── */
[data-testid="stPlotlyChart"] > div { background: transparent !important; }

.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
    font-size: 0.74rem !important;
    font-family: 'Inter', sans-serif !important;
}
</style>
"""

_CARD_STYLE = (
    "background:#1e2535;"
    "border:1px solid #2a3a4a;"
    "border-radius:12px;"
    "padding:1rem 0.9rem 0.75rem;"
    "margin-bottom:0.1rem;"
)


def _section_label(text: str):
    st.markdown(
        f"<p style='margin:0 0 0.35rem 0;font-size:0.85rem;font-weight:500;"
        f"color:#e8edf5;font-family:Inter,sans-serif;'>{text}</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────

def init_session_state():
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("current_user_email", None)
    st.session_state.setdefault("current_user_name", None)
    st.session_state.setdefault("current_driver_id", None)


# ─────────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────────

def _resolve_driver(email: str, name: str) -> int | None:
    driver = get_driver_by_email(email)
    if driver is None:
        driver = create_driver_with_email(name, email)
    return driver["driver_id"] if driver else None


def do_login(email: str, password: str) -> str | None:
    account = get_account_by_email(email)
    if account is None:
        return "No account found with that email."
    if not verify_password(password, account["password_hash"]):
        return "Incorrect password."
    st.session_state.update({
        "authenticated":      True,
        "current_user_email": account["email"],
        "current_user_name":  account["name"],
        "current_driver_id":  account["driver_id"],
    })
    return None


def do_signup(name: str, email: str, password: str) -> str | None:
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
    st.session_state.update({
        "authenticated":      True,
        "current_user_email": email,
        "current_user_name":  name.strip(),
        "current_driver_id":  driver_id,
    })
    return None


def do_logout():
    st.session_state.update({
        "authenticated":      False,
        "current_user_email": None,
        "current_user_name":  None,
        "current_driver_id":  None,
    })


# ─────────────────────────────────────────────
# Login / Signup UI
# ─────────────────────────────────────────────

def render_auth_page():
    st.title("Driver Safety Dashboard")
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
# Main
# ─────────────────────────────────────────────

def main():
    init_session_state()

    st.set_page_config(
        page_title="Driver Safety Dashboard",
        page_icon="🚗",
        layout="centered" if not st.session_state["authenticated"] else "wide",
    )

    st.markdown(THEME_CSS, unsafe_allow_html=True)

    if not st.session_state["authenticated"]:
        render_auth_page()
        return

    # ── Sidebar ──
    with st.sidebar:
        st.markdown(
            "<p style='color:#8899aa;font-size:0.72rem;margin:0 0 2px 0;'>Logged in as</p>"
            f"<strong>{st.session_state['current_user_name']}</strong>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Log out", use_container_width=True):
            do_logout()
            st.rerun()
        st.divider()

    st.title("Simplified Driver Safety Dashboard")

    driver_id = st.session_state["current_driver_id"]
    if driver_id is None:
        st.error("No driver profile linked. Please contact support.")
        return

    driver = get_driver(driver_id)
    if driver is None:
        st.error("Driver profile could not be loaded.")
        return

    render_driver_baseline(driver)

    metrics     = build_driver_metrics(driver_id)
    timeline_df = get_event_timeline(driver_id)

    col_risk, col_chart, col_timeline = st.columns([1.1, 1.6, 1.1], gap="medium")
    with col_risk:
        render_current_risk(metrics["latest_risk"])
    with col_chart:
        render_risk_chart(metrics["risk_timeseries"])
    with col_timeline:
        render_event_timeline(timeline_df)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    render_session_stats(metrics["event_counts"], metrics["sessions"], metrics["max_risk"])


# ─────────────────────────────────────────────
# Render helpers
# ─────────────────────────────────────────────

def render_driver_baseline(driver):
    col1, col2, col3 = st.columns(3, gap="medium")
    col1.metric("Driver:", driver["name"])
    col2.metric(
        "Baseline EAR:",
        f"{driver['baseline_ear']:.3f}" if driver["baseline_ear"] is not None else "N/A",
    )
    col3.metric(
        "Baseline Yaw:",
        f"{driver['baseline_yaw']:.2f}" if driver["baseline_yaw"] is not None else "N/A",
    )


def render_current_risk(latest_risk: float):
    if latest_risk >= 60:
        needle_color, label = "#e84040", "High risk"
    elif latest_risk >= 30:
        needle_color, label = "#e8872a", "Elevated risk"
    else:
        needle_color, label = "#2dd4bf", "Stable"

    # Card border via container
    with st.container():
        st.markdown(f"<div style='{_CARD_STYLE}'>", unsafe_allow_html=True)
        _section_label("Current Risk Display")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_risk,
            number={"font": {"size": 44, "color": "#e8edf5", "family": "Inter"}},
            title={"text": label, "font": {"size": 13, "color": "#aabbcc", "family": "Inter"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#2a3a4a",
                    "tickfont": {"color": "#8899aa", "size": 9},
                    "tickwidth": 1,
                    "nticks": 6,
                },
                "bar": {"color": needle_color, "thickness": 0.22},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  30],  "color": "#18301e"},
                    {"range": [30, 60],  "color": "#322010"},
                    {"range": [60, 100], "color": "#321010"},
                ],
                "threshold": {
                    "line": {"color": needle_color, "width": 3},
                    "thickness": 0.78,
                    "value": latest_risk,
                },
            },
        ))
        fig.update_layout(
            height=250,
            margin=dict(l=18, r=18, t=22, b=5),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter"},
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


def render_risk_chart(risk_timeseries):
    with st.container():
        st.markdown(f"<div style='{_CARD_STYLE}'>", unsafe_allow_html=True)
        _section_label("Risk Over Time")

        if risk_timeseries.empty:
            st.info("No risk data available yet.")
        else:
            chart_df = risk_timeseries.copy()
            chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])

            fig = go.Figure()
            for sid in chart_df["session_id"].unique():
                sdf = chart_df[chart_df["session_id"] == sid]
                fig.add_trace(go.Scatter(
                    x=sdf["timestamp"],
                    y=sdf["risk_score"],
                    mode="lines+markers",
                    name=f"session_id: {sid}",
                    line=dict(color="#4a9eff", width=2.5),
                    marker=dict(color="#4a9eff", size=7,
                                line=dict(color="#e8edf5", width=1)),
                    fill="tozeroy",
                    fillcolor="rgba(74,158,255,0.08)",
                ))

            fig.update_layout(
                height=230,
                margin=dict(l=5, r=5, t=8, b=5),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(
                    orientation="h", yanchor="top", y=1.18,
                    font=dict(color="#8899aa", size=10),
                    bgcolor="rgba(0,0,0,0)",
                ),
                xaxis=dict(
                    showgrid=False,
                    tickfont=dict(color="#8899aa", size=10),
                    tickformat="%b %-d, %Y",
                    linecolor="#2a3a4a",
                ),
                yaxis=dict(
                    showgrid=True, gridcolor="#202d3e",
                    tickfont=dict(color="#8899aa", size=10),
                    linecolor="#2a3a4a",
                    range=[0, None],
                ),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown("</div>", unsafe_allow_html=True)


def render_event_timeline(timeline_df):
    with st.container():
        st.markdown(
            f"<div style='{_CARD_STYLE}min-height:290px;'>",
            unsafe_allow_html=True,
        )
        _section_label("Event Timeline")

        if timeline_df.empty:
            st.info("No events logged yet.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        view_df = timeline_df[["event_type", "risk_score", "session_id"]].copy()

        rows_html = ""
        for _, row in view_df.iterrows():
            evt   = str(row["event_type"])
            score = row["risk_score"]
            sid   = int(row["session_id"])
            color = "#e84040" if ("WARNING" in evt or "DISTRACTION" in evt) else "#e8872a"
            rows_html += (
                f'<tr style="border-bottom:1px solid rgba(42,58,74,0.55);">'
                f'<td style="color:{color};font-size:0.7rem;padding:0.3rem 0.4rem;'
                f'font-weight:500;word-break:break-all;">{evt}</td>'
                f'<td style="color:#e8edf5;font-size:0.7rem;padding:0.3rem 0.4rem;'
                f'white-space:nowrap;">{score}</td>'
                f'<td style="color:#8899aa;font-size:0.7rem;padding:0.3rem 0.4rem;">{sid}</td>'
                f'</tr>'
            )

        st.markdown(
            f'<div style="max-height:245px;overflow-y:auto;border-radius:6px;">'
            f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;">'
            f'<thead><tr style="border-bottom:1px solid #2a3a4a;">'
            f'<th style="color:#8899aa;font-size:0.67rem;font-weight:500;'
            f'text-align:left;padding:0.28rem 0.4rem;">event_type</th>'
            f'<th style="color:#8899aa;font-size:0.67rem;font-weight:500;'
            f'text-align:left;padding:0.28rem 0.4rem;">risk_score</th>'
            f'<th style="color:#8899aa;font-size:0.67rem;font-weight:500;'
            f'text-align:left;padding:0.28rem 0.4rem;">session_id</th>'
            f'</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


def render_session_stats(event_counts, sessions, max_risk_df):
    event_lookup = {}
    if not event_counts.empty:
        event_lookup = dict(zip(event_counts["event_type"], event_counts["event_count"]))

    highest_risk = float(max_risk_df["max_risk"].iloc[0]) if not max_risk_df.empty else 0.0

    phone      = int(event_lookup.get("PHONE_DISTRACTION_EVENT", 0))
    drowsiness = int(event_lookup.get("DROWSINESS_EVENT", 0))
    microsleep = int(event_lookup.get("MICROSLEEP_EVENT", 0))
    gaze_away  = int(event_lookup.get("GAZE_AWAY_EVENT", 0))

    icon_phone = (
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" '
        'stroke="#e8872a" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="7" y="2" width="10" height="20" rx="2"/>'
        '<line x1="12" y1="18" x2="12.01" y2="18"/></svg>'
    )
    icon_eye = (
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" '
        'stroke="#a855f7" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>'
        '<circle cx="12" cy="12" r="3"/></svg>'
    )
    icon_zzz = (
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" '
        'stroke="#2dd4bf" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="17 6.5 23 6.5"/><line x1="23" y1="6.5" x2="17" y2="12.5"/>'
        '<polyline points="17 12.5 23 12.5"/>'
        '<polyline points="11 3 17 3"/><line x1="17" y1="3" x2="11" y2="9"/>'
        '<polyline points="11 9 17 9"/></svg>'
    )
    icon_gaze = (
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" '
        'stroke="#4a9eff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>'
        '<circle cx="12" cy="12" r="3"/>'
        '<line x1="2" y1="2" x2="22" y2="22"/></svg>'
    )
    icon_warn = (
        '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" '
        'stroke="#e84040" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/>'
        '<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
    )

    stats = [
        ("Phone",        str(phone),            icon_phone, "#e8edf5"),
        ("Drowsiness",   str(drowsiness),       icon_eye,   "#e8edf5"),
        ("Microsleep",   str(microsleep),       icon_zzz,   "#e8edf5"),
        ("Gaze away",    str(gaze_away),        icon_gaze,  "#e8edf5"),
        ("Highest risk", f"{highest_risk:.1f}", icon_warn,  "#e84040"),
    ]

    cols = st.columns(5, gap="medium")
    for col, (label, value, icon_svg, val_color) in zip(cols, stats):
        with col:
            st.markdown(
                f'<div style="background:#1e2535;border:1px solid #2a3a4a;border-radius:12px;'
                f'padding:1rem 1.1rem;display:flex;align-items:center;'
                f'justify-content:space-between;min-height:84px;min-width:0;">'
                f'<div style="min-width:0;">'
                f'<div style="font-size:clamp(1.4rem,2.2vw,2.1rem);font-weight:700;'
                f'color:{val_color};font-family:Inter,sans-serif;line-height:1.1;'
                f'white-space:nowrap;">{value}</div>'
                f'<div style="font-size:0.74rem;color:#8899aa;font-family:Inter,sans-serif;'
                f'margin-top:0.2rem;">{label}</div>'
                f'</div>'
                f'<div style="opacity:0.85;flex-shrink:0;margin-left:0.5rem;">{icon_svg}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f"<p style='color:#8899aa;font-size:0.73rem;margin-top:0.5rem;"
        f"font-family:Inter,sans-serif;'>Recorded sessions: {len(sessions)}</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
