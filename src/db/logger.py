from db.connection import get_db_connection


class DBLogger:
    def __init__(self, driver_id=None, session_id=None):
        self.conn = get_db_connection()
        self.cursor = self.conn.cursor()
        self.driver_id = driver_id
        self.session_id = session_id

    def set_context(self, driver_id=None, session_id=None):
        self.driver_id = driver_id
        self.session_id = session_id

    def log(self, score, event, driver_id=None, session_id=None):
        effective_driver_id = self.driver_id if driver_id is None else driver_id
        effective_session_id = self.session_id if session_id is None else session_id

        self.cursor.execute(
            """
            INSERT INTO risk_logs (risk_score, event_type, driver_id, session_id)
            VALUES (%s, %s, %s, %s)
            """,
            (score, event, effective_driver_id, effective_session_id)
        )
        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()
