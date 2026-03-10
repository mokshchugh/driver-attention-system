import psycopg2

from config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER


def get_db_connection():
    connection_kwargs = {
        "dbname": DB_NAME,
        "user": DB_USER,
        "host": DB_HOST,
        "port": DB_PORT,
    }

    if DB_PASSWORD:
        connection_kwargs["password"] = DB_PASSWORD

    return psycopg2.connect(**connection_kwargs)
