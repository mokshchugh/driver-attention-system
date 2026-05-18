import threading
import queue
from db.connection import get_db_connection


class DBLogger:
    def __init__(self, driver_id=None, session_id=None):
        self.driver_id = driver_id
        self.session_id = session_id
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._buffer = []                    # ← holds rows for Neon upload
        self._buffer_lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        conn = get_db_connection()
        cursor = conn.cursor()
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                row = self._queue.get(timeout=0.1)
                score, event, driver_id, session_id = row

                cursor.execute(
                    """
                    INSERT INTO risk_logs (risk_score, event_type, driver_id, session_id)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (score, event, driver_id, session_id)
                )
                conn.commit()

                with self._buffer_lock:      # ← buffer after successful local write
                    self._buffer.append(row)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[DBLogger] Write error: {e}")
        cursor.close()
        conn.close()

    def log(self, score, event, driver_id=None, session_id=None):
        self._queue.put((
            score,
            event,
            driver_id if driver_id is not None else self.driver_id,
            session_id if session_id is not None else self.session_id,
        ))

    def get_buffer(self) -> list:
        """Returns a snapshot of buffered rows for Neon upload."""
        with self._buffer_lock:
            return list(self._buffer)

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=10)
