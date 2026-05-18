import bcrypt
from db.connection import get_db_connection
from db.connection import get_neon_connection
from db.sync import sync_account


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def get_neon_account_by_email(email):
    with get_neon_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM accounts
                WHERE email = %s
                """,
                (email,)
            )

            row = cursor.fetchone()

            if row is None:
                return None

            cols = [d[0] for d in cursor.description]
            return dict(zip(cols, row))

def get_account_by_email(email):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM accounts
                WHERE email = %s
                """,
                (email,),
            )

            row = cursor.fetchone()

            if row is None:
                return None

            cols = [d[0] for d in cursor.description]
            return dict(zip(cols, row))

def get_neon_driver(driver_id):
    with get_neon_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM drivers
                WHERE driver_id = %s
                """,
                (driver_id,),
            )

            row = cursor.fetchone()

            if row is None:
                return None

            cols = [d[0] for d in cursor.description]
            return dict(zip(cols, row))

def cache_account_locally(account):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO accounts (
                    account_id,
                    driver_id,
                    name,
                    email,
                    password_hash
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (account_id)
                DO NOTHING
                """,
                (
                    account["account_id"],
                    account["driver_id"],
                    account["name"],
                    account["email"],
                    account["password_hash"],
                ),
            )
        conn.commit()

def cache_driver_locally(driver):
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO drivers (
                    driver_id,
                    name,
                    email
                )
                VALUES (%s, %s, %s)
                ON CONFLICT (driver_id)
                DO NOTHING
                """,
                (
                    driver["driver_id"],
                    driver["name"],
                    driver["email"],
                ),
            )
        conn.commit()

def verify_password(password, password_hash):
    return bcrypt.checkpw(
        password.encode(),
        password_hash.encode()
	)

def hydrate_account_from_cloud(email):
    """
    Pulls account + driver from Neon
    into local PostgreSQL cache.
    """

    account = get_neon_account_by_email(email)

    if account is None:
        return None

    driver = get_neon_driver(account["driver_id"])

    if driver is None:
        return None

    cache_driver_locally(driver)
    cache_account_locally(account)

    return account

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
    result = _row_to_dict(row) if row else None
    if result:
        sync_account(result["account_id"])   # ← sync to Neon after local write
    return result


def _row_to_dict(row) -> dict:
    return {
        "account_id":    row[0],
        "email":         row[1],
        "name":          row[2],
        "password_hash": row[3],
        "driver_id":     row[4],
    }
