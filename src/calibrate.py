import sys
import getpass
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from db.accounts import get_account_by_email, verify_password
from services.calibration import start_calibration


def prompt_login() -> int:
    print("=== Driver Calibration – Login ===")
    email    = input("Email: ").strip().lower()
    password = getpass.getpass("Password: ")

    account = get_account_by_email(email)

    if account is None:
        print("Error: no account found for that email.")
        sys.exit(1)

    if not verify_password(password, account["password_hash"]):
        print("Error: incorrect password.")
        sys.exit(1)

    driver_id = account["driver_id"]
    if driver_id is None:
        print("Error: account has no linked driver profile.")
        sys.exit(1)

    print(f"Logged in as {account['name']} (driver #{driver_id})\n")
    return driver_id


def main():
    driver_id = prompt_login()

    print("Starting calibration — look straight ahead and keep eyes naturally open.")
    print("This will take 12 seconds...\n")

    try:
        result = start_calibration(driver_id)
        print(f"\nCalibration saved successfully:")
        print(f"  Baseline EAR : {result['baseline_ear']:.3f}")
        print(f"  Baseline Yaw : {result['baseline_yaw']:.2f}")
    except RuntimeError as exc:
        print(f"Calibration failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
