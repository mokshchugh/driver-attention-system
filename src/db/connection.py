import os
import psycopg2
from config import (
    LOCAL_PG_HOST, LOCAL_PG_PORT, LOCAL_PG_DB, LOCAL_PG_USER, LOCAL_PG_PASSWORD,
    NEON_HOST, NEON_PORT, NEON_DB, NEON_USER, NEON_PASSWORD,
)

STREAMLIT_CLOUD = os.getenv("STREAMLIT_CLOUD", "false").lower() == "true"

def get_db_connection():
    """
    Active DB connection.
    Streamlit Cloud -> Neon
    Local runtime -> Local PostgreSQL
    """

    if STREAMLIT_CLOUD:
        return psycopg2.connect(
            host=NEON_HOST,
            port=NEON_PORT,
            dbname=NEON_DB,
            user=NEON_USER,
            password=NEON_PASSWORD,
            sslmode="require"
        )

    return psycopg2.connect(
        host=LOCAL_PG_HOST,
        port=LOCAL_PG_PORT,
        dbname=LOCAL_PG_DB,
        user=LOCAL_PG_USER,
        password=LOCAL_PG_PASSWORD
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
