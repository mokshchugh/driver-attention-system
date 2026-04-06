-- Migration 004: persistent account store shared by dashboard and CLI
-- Replaces the in-memory st.session_state user_store in app.py

CREATE TABLE IF NOT EXISTS accounts (
    account_id    SERIAL PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    driver_id     INTEGER REFERENCES drivers(driver_id) ON DELETE SET NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
