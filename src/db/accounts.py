import bcrypt

from db.connection import get_db_connection


# ─────────────────────────────────────────────
# Password helpers
# ─────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ─────────────────────────────────────────────
# Account helpers
# ─────────────────────────────────────────────

def get_account_by_email(email: str) -> dict | None:
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT account_id, email, name, password_hash, driver_id
                FROM accounts
                WHERE email = %s
                """,
                (email,),
            )
            row = cursor.fetchone()
    return _row_to_dict(row) if row else None


def create_account(name: str, email: str, plain_password: str, driver_id: int) -> dict | None:
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO accounts (name, email, password_hash, driver_id)
                VALUES (%s, %s, %s, %s)
                RETURNING account_id, email, name, password_hash, driver_id
                """,
                (name, email, hash_password(plain_password), driver_id),
            )
            row = cursor.fetchone()
    return _row_to_dict(row) if row else None


def _row_to_dict(row) -> dict:
    return {
        "account_id":    row[0],
        "email":         row[1],
        "name":          row[2],
        "password_hash": row[3],
        "driver_id":     row[4],
    }
