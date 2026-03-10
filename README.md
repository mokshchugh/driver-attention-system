# Driver Attention Momentum Tracking System

This project monitors a driver in real time using a webcam and combines multiple signals into a momentum-style risk score. The runtime pipeline uses MediaPipe face landmarks, EAR-based eye closure detection, YOLOv8 phone detection, head-pose yaw checks, and PostgreSQL event logging.

## Repository layout

- `src/main.py`: live runtime monitoring loop
- `src/detectors/`: face, eye, phone, and head-pose detectors
- `src/risk/`: momentum risk engine
- `src/db/`: database access, logging, and driver profile management
- `src/services/`: calibration and monitoring session lifecycle
- `src/analytics/`: read-only analytics queries and metrics
- `dashboard/app.py`: Streamlit dashboard
- `database/migrations/`: SQL schema updates for drivers and sessions

## Database setup

Run the SQL migrations in order:

1. `database/migrations/001_create_drivers_and_sessions.sql`
2. `database/migrations/002_alter_risk_logs_add_driver_and_session.sql`

By default the app connects with:

- `DB_NAME=driver_attention_db`
- `DB_USER=moksh`
- `DB_HOST=localhost`
- `DB_PORT=5432`

You can override these with environment variables. Set `DB_PASSWORD` if your PostgreSQL instance requires one.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run monitoring

For a driver-linked monitoring session:

```bash
python src/main.py --driver-id 1
```

If you omit `--driver-id`, the runtime still works but the session and event rows will not be tied to a driver profile.

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Dashboard workflow

1. Create a driver from the sidebar.
2. Select that driver profile.
3. Start calibration to capture baseline EAR and neutral yaw from the camera.
4. Run monitoring with `python src/main.py --driver-id <id>`.
5. Open the dashboard to review current risk, event history, and session analytics.

## Features

- Multiple driver profiles
- Baseline calibration per driver
- Session-based monitoring with max-risk tracking
- Event logging tied to driver and session IDs
- Streamlit dashboard backed by analytics queries
