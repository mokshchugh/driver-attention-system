ALTER TABLE risk_logs
    ADD COLUMN IF NOT EXISTS driver_id INTEGER REFERENCES drivers(driver_id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS session_id INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_risk_logs_driver_timestamp
    ON risk_logs (driver_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_risk_logs_session
    ON risk_logs (session_id);
