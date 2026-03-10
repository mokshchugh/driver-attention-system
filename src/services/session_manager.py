from db.connection import get_db_connection


class SessionManager:
    def __init__(self, driver_id):
        self.driver_id = driver_id
        self.session_id = None
        self.max_risk = 0.0

    def start_session(self):
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO sessions (driver_id, start_time, max_risk)
                    VALUES (%s, CURRENT_TIMESTAMP, %s)
                    RETURNING session_id
                    """,
                    (self.driver_id, self.max_risk),
                )
                self.session_id = cursor.fetchone()[0]
        return self.session_id

    def update_max_risk(self, risk_score):
        if self.session_id is None or risk_score <= self.max_risk:
            return

        self.max_risk = float(risk_score)

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE sessions
                    SET max_risk = %s
                    WHERE session_id = %s
                    """,
                    (self.max_risk, self.session_id),
                )

    def close_session(self):
        if self.session_id is None:
            return

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE sessions
                    SET end_time = CURRENT_TIMESTAMP,
                        max_risk = %s
                    WHERE session_id = %s
                    """,
                    (self.max_risk, self.session_id),
                )
