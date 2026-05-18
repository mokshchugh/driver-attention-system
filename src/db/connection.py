import psycopg2
from config import (
    LOCAL_PG_HOST, LOCAL_PG_PORT, LOCAL_PG_DB, LOCAL_PG_USER, LOCAL_PG_PASSWORD,
    NEON_HOST, NEON_PORT, NEON_DB, NEON_USER, NEON_PASSWORD,
)

def get_db_connection():
    """Local PostgreSQL connection."""
    return psycopg2.connect(
        host=LOCAL_PG_HOST,
        port=LOCAL_PG_PORT,
        dbname=LOCAL_PG_DB,
        user=LOCAL_PG_USER,
        password=LOCAL_PG_PASSWORD,
    )

def get_neon_connection():
    """Neon PostgreSQL connection — used only at session end."""
    return psycopg2.connect(
        host=NEON_HOST,
        port=NEON_PORT,
        dbname=NEON_DB,
        user=NEON_USER,
        password=NEON_PASSWORD,
        sslmode="require",
    )
