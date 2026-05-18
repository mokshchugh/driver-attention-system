from db.connection import get_db_connection
from db.sync import sync_session_end
from db.logger import DBLogger
import threading
import queue

class SessionManager:
    def __init__(self, driver_id):
        self.driver_id = driver_id
        self.session_id = None
        self.max_risk = 0.0
        self._update_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        conn = get_db_connection()
        cursor = conn.cursor()
        while not self._stop_event.is_set() or not self._update_queue.empty():
            try:
                max_risk = self._update_queue.get(timeout=0.1)
                cursor.execute(
                    "UPDATE sessions SET max_risk = %s WHERE session_id = %s",
                    (max_risk, self.session_id)
                )
                conn.commit()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[SessionManager] Write error: {e}")
        cursor.close()
        conn.close()

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
        # Non-blocking — drop older pending updates if a newer value exists
        while not self._update_queue.empty():
            try:
                self._update_queue.get_nowait()
            except queue.Empty:
                break
        self._update_queue.put(self.max_risk)


    def close_session(self, logger: DBLogger):
        self._stop_event.set()
        self._thread.join(timeout=10)
    
        if self.session_id is None:
            return
    
        # 1. Final local write
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE sessions SET end_time = CURRENT_TIMESTAMP, max_risk = %s WHERE session_id = %s",
                    (self.max_risk, self.session_id),
                )
    
        # 2. Neon sync — driver → session → risk_logs
        sync_session_end(self.session_id, self.driver_id, logger.get_buffer())
