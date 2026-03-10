from db.connection import get_db_connection


def create_driver(name):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO drivers (name)
                VALUES (%s)
                RETURNING driver_id, name, baseline_ear, baseline_yaw, created_at
                """,
                (name,),
            )
            row = cursor.fetchone()
    return _driver_row_to_dict(row) if row else None


def get_all_drivers():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT driver_id, name, baseline_ear, baseline_yaw, created_at
                FROM drivers
                ORDER BY created_at DESC, driver_id DESC
                """
            )
            rows = cursor.fetchall()
    return [_driver_row_to_dict(row) for row in rows]


def get_driver(driver_id):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT driver_id, name, baseline_ear, baseline_yaw, created_at
                FROM drivers
                WHERE driver_id = %s
                """,
                (driver_id,),
            )
            row = cursor.fetchone()
    return _driver_row_to_dict(row) if row else None


def delete_driver(driver_id):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM drivers WHERE driver_id = %s", (driver_id,))
            deleted_rows = cursor.rowcount
    return deleted_rows > 0


def update_baseline(driver_id, ear, yaw):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE drivers
                SET baseline_ear = %s,
                    baseline_yaw = %s
                WHERE driver_id = %s
                RETURNING driver_id, name, baseline_ear, baseline_yaw, created_at
                """,
                (ear, yaw, driver_id),
            )
            row = cursor.fetchone()
    return _driver_row_to_dict(row) if row else None


def _driver_row_to_dict(row):
    return {
        "driver_id": row[0],
        "name": row[1],
        "baseline_ear": row[2],
        "baseline_yaw": row[3],
        "created_at": row[4],
    }
