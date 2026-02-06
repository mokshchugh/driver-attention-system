import psycopg2
from config import DB_NAME, DB_USER


class DBLogger:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER
        )
        self.cursor = self.conn.cursor()

    def log(self, score, event):
        self.cursor.execute(
            "INSERT INTO risk_logs (risk_score, event_type) VALUES (%s, %s)",
            (score, event)
        )
        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()
