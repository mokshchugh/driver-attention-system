import pandas as pd

from db.connection import get_db_connection


def get_risk_timeseries(driver_id):
    query = """
        SELECT timestamp, risk_score, event_type, session_id
        FROM risk_logs
        WHERE driver_id = %s
        ORDER BY timestamp ASC
    """
    return _read_dataframe(query, (driver_id,))


def get_event_counts(driver_id):
    query = """
        SELECT event_type, COUNT(*) AS event_count
        FROM risk_logs
        WHERE driver_id = %s
        GROUP BY event_type
        ORDER BY event_type ASC
    """
    return _read_dataframe(query, (driver_id,))


def get_driver_sessions(driver_id):
    query = """
        SELECT session_id, driver_id, start_time, end_time, max_risk
        FROM sessions
        WHERE driver_id = %s
        ORDER BY start_time DESC
    """
    return _read_dataframe(query, (driver_id,))


def get_max_risk(driver_id):
    query = """
        SELECT COALESCE(MAX(risk_score), 0) AS max_risk
        FROM risk_logs
        WHERE driver_id = %s
    """
    return _read_dataframe(query, (driver_id,))


def get_event_timeline(driver_id):
    query = """
        SELECT timestamp, event_type, risk_score, session_id
        FROM risk_logs
        WHERE driver_id = %s
        ORDER BY timestamp DESC
    """
    return _read_dataframe(query, (driver_id,))


def _read_dataframe(query, params):
    with get_db_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)
