from analytics.queries import (
    get_driver_sessions,
    get_event_counts,
    get_max_risk,
    get_risk_timeseries,
)


def build_driver_metrics(driver_id):
    risk_timeseries = get_risk_timeseries(driver_id)
    event_counts = get_event_counts(driver_id)
    sessions = get_driver_sessions(driver_id)
    max_risk = get_max_risk(driver_id)

    latest_risk = (
        float(risk_timeseries["risk_score"].iloc[-1]) if not risk_timeseries.empty else 0.0
    )

    return {
        "risk_timeseries": risk_timeseries,
        "event_counts": event_counts,
        "sessions": sessions,
        "max_risk": max_risk,
        "latest_risk": latest_risk,
    }
