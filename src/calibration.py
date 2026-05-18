import getpass
import sys

from db.accounts import (
    get_account_by_email,
    hydrate_account_from_cloud,
    verify_password,
)

from calibrate.pipeline import run_calibration_pipeline


def prompt_login():
    print("=== Driver Baseline Calibration ===")

    email = input("Email: ").strip().lower()
    password = getpass.getpass("Password: ")

    account = get_account_by_email(email)

    # Local cache miss → hydrate from Neon
    if account is None:
        print("Local cache miss. Fetching from cloud...")

        account = hydrate_account_from_cloud(email)

    if account is None:
        print("No account found.")
        sys.exit(1)

    if not verify_password(password, account["password_hash"]):
        print("Incorrect password.")
        sys.exit(1)

    return account["driver_id"]


def main():
    driver_id = prompt_login()

    result = run_calibration_pipeline(driver_id)

    print("\nCalibration complete.")
    print(
        f"EAR={result['baseline_ear']:.3f}, "
        f"Yaw={result['baseline_yaw']:.2f}"
    )


if __name__ == "__main__":
    main()
