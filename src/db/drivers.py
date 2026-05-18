from db.connection import get_db_connection
from db.connection import get_neon_connection
from db.sync import sync_driver

def update_baseline_local(
    driver_id: int,
    ear: float,
    yaw: float,
):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:

            cursor.execute(
                """
                UPDATE drivers
                SET baseline_ear = %s,
                    baseline_yaw = %s
                WHERE driver_id = %s
                RETURNING *
                """,
                (
                    ear,
                    yaw,
                    driver_id,
                ),
            )

            row = cursor.fetchone()

            if row is None:
                return None

            cols = [
                d[0]
                for d in cursor.description
            ]

        conn.commit()

    return dict(zip(cols, row))

def update_baseline_neon(driver_id: int, ear: float, yaw: float):
    with get_neon_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE drivers
                SET baseline_ear = %s,
                    baseline_yaw = %s
                WHERE driver_id = %s
                RETURNING *
                """,
                (ear, yaw, driver_id),
            )

            row = cursor.fetchone()

            if row is None:
                return None

            cols = [d[0] for d in cursor.description]

        conn.commit()

    return dict(zip(cols, row))

def get_driver_by_email(email: str) -> dict | None:
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT driver_id, name, baseline_ear, baseline_yaw, created_at FROM drivers WHERE email = %s",
                (email,),
            )
            row = cursor.fetchone()
    return _driver_row_to_dict(row) if row else None


def create_driver_with_email(name: str, email: str) -> dict | None:
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO drivers (name, email) VALUES (%s, %s)
                RETURNING driver_id, name, baseline_ear, baseline_yaw, created_at
                """,
                (name, email),
            )
            row = cursor.fetchone()
    result = _driver_row_to_dict(row) if row else None
    if result:
        sync_driver(result["driver_id"])     # ← sync
    return result


def create_driver(name: str) -> dict | None:
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO drivers (name) VALUES (%s)
                RETURNING driver_id, name, baseline_ear, baseline_yaw, created_at
                """,
                (name,),
            )
            row = cursor.fetchone()
    result = _driver_row_to_dict(row) if row else None
    if result:
        sync_driver(result["driver_id"])     # ← sync
    return result


def get_all_drivers() -> list[dict]:         # read-only, no sync needed
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT driver_id, name, baseline_ear, baseline_yaw, created_at FROM drivers ORDER BY created_at DESC, driver_id DESC"
            )
            rows = cursor.fetchall()
    return [_driver_row_to_dict(row) for row in rows]


def get_driver(driver_id: int) -> dict | None:   # read-only
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT driver_id, name, baseline_ear, baseline_yaw, created_at FROM drivers WHERE driver_id = %s",
                (driver_id,),
            )
            row = cursor.fetchone()
    return _driver_row_to_dict(row) if row else None


def delete_driver(driver_id: int) -> bool:   # deletions intentionally not synced
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM drivers WHERE driver_id = %s", (driver_id,))
            return cursor.rowcount > 0


def update_baseline(driver_id: int, ear: float, yaw: float) -> dict | None:
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE drivers SET baseline_ear = %s, baseline_yaw = %s
                WHERE driver_id = %s
                RETURNING driver_id, name, baseline_ear, baseline_yaw, created_at
                """,
                (ear, yaw, driver_id),
            )
            row = cursor.fetchone()
    result = _driver_row_to_dict(row) if row else None
    if result:
        sync_driver(result["driver_id"])     # ← sync baseline updates too
    return result


def _driver_row_to_dict(row) -> dict:
    return {
        "driver_id":    row[0],
        "name":         row[1],
        "baseline_ear": row[2],
        "baseline_yaw": row[3],
        "created_at":   row[4],
    }
