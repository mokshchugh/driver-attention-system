-- Migration 003: bind each driver profile to a login account
ALTER TABLE drivers
    ADD COLUMN IF NOT EXISTS email TEXT UNIQUE;
