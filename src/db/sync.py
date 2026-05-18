import psycopg2.extras
from db.connection import get_db_connection, get_neon_connection


def _neon():
    conn = get_neon_connection()
    return conn, conn.cursor()


# ─────────────────────────────────────────────
# Generic upsert helper (reads actual columns
# from local so it never goes out of sync)
# ─────────────────────────────────────────────

def _upsert_row(table: str, pk: str, pk_val, local_query: str, local_params: tuple):
    local = get_db_connection()
    cur = local.cursor()
    cur.execute(local_query, local_params)
    row = cur.fetchone()
    cols = [d[0] for d in cur.description]
    cur.close()
    local.close()

    if not row:
        print(f"[Sync] {table} {pk}={pk_val} not found locally.")
        return False

    data = dict(zip(cols, row))
    col_names   = ", ".join(data.keys())
    placeholders = ", ".join(["%s"] * len(data))
    updates     = ", ".join(f"{c} = EXCLUDED.{c}" for c in data.keys() if c != pk)

    conn, ncur = _neon()
    ncur.execute(
        f"""
        INSERT INTO {table} ({col_names}) VALUES ({placeholders})
        ON CONFLICT ({pk}) DO UPDATE SET {updates}
        """,
        list(data.values())
    )
    conn.commit()
    ncur.close()
    conn.close()
    print(f"[Sync] {table} {pk}={pk_val} synced to Neon.")
    return True


# ─────────────────────────────────────────────
# Public sync functions
# ─────────────────────────────────────────────

def sync_account(account_id: int):
    try:
        _upsert_row(
            table="accounts",
            pk="account_id",
            pk_val=account_id,
            local_query="SELECT * FROM accounts WHERE account_id = %s",
            local_params=(account_id,),
        )
    except Exception as e:
        print(f"[Sync] Account sync error: {e}")


def sync_driver(driver_id: int):
    try:
        _upsert_row(
            table="drivers",
            pk="driver_id",
            pk_val=driver_id,
            local_query="SELECT * FROM drivers WHERE driver_id = %s",
            local_params=(driver_id,),
        )
    except Exception as e:
        print(f"[Sync] Driver sync error: {e}")


def sync_session_end(session_id: int, driver_id: int, risk_log_buffer: list):
    """
    Called once when a session closes.
    Uploads in strict FK order: driver → session → risk_logs
    """
    sync_driver(driver_id)
    _sync_session(session_id)
    _sync_risk_logs(risk_log_buffer)


def _sync_session(session_id: int):
    try:
        _upsert_row(
            table="sessions",
            pk="session_id",
            pk_val=session_id,
            local_query="SELECT * FROM sessions WHERE session_id = %s",
            local_params=(session_id,),
        )
    except Exception as e:
        print(f"[Sync] Session sync error: {e}")


def _sync_risk_logs(buffer: list):
    if not buffer:
        return
    try:
        conn, ncur = _neon()
        psycopg2.extras.execute_values(
            ncur,
            """
            INSERT INTO risk_logs (risk_score, event_type, driver_id, session_id)
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            buffer,   # list of (risk_score, event_type, driver_id, session_id)
        )
        conn.commit()
        ncur.close()
        conn.close()
        print(f"[Sync] {len(buffer)} risk_log rows synced to Neon.")
    except Exception as e:
        print(f"[Sync] risk_logs sync error: {e}")
