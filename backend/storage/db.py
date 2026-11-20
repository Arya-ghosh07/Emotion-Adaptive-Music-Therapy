"""SQLite storage. Everything a session measures is logged here, so results can
be re-analysed (scripts/evaluate.py) and the learned model can be audited."""
import os
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    resting_bpm REAL
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    mode TEXT NOT NULL,
    started_at REAL NOT NULL,
    music_started_at REAL,
    ended_at REAL,
    baseline_bpm REAL,
    heart_source TEXT,
    checkin_text TEXT,
    checkin_emotion TEXT,
    completed INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    t REAL NOT NULL,
    modality TEXT NOT NULL,
    valence REAL, arousal REAL,
    bpm REAL, quality REAL, reliable INTEGER, rmssd REAL,
    phase TEXT
);
CREATE INDEX IF NOT EXISTS readings_session ON readings(session_id, modality, t);
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    idx INTEGER NOT NULL,
    preset TEXT,
    context TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    hr_start REAL, hr_end REAL, rmssd_start REAL, rmssd_end REAL,
    thumbs INTEGER,
    reward REAL,
    valid INTEGER DEFAULT 0,
    basis TEXT
);
CREATE INDEX IF NOT EXISTS segments_user ON segments(user_id, mode, valid);
CREATE TABLE IF NOT EXISTS self_reports (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    t REAL NOT NULL,
    moment TEXT NOT NULL,
    valence_raw INTEGER, arousal_raw INTEGER,
    valence REAL, arousal REAL
);
CREATE TABLE IF NOT EXISTS modality_errors (
    user_id TEXT NOT NULL,
    modality TEXT NOT NULL,
    dim TEXT NOT NULL,
    error REAL NOT NULL,
    n INTEGER NOT NULL,
    PRIMARY KEY (user_id, modality, dim)
);
"""


class Database:
    def __init__(self, path: str):
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def execute(self, sql, params=()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.lastrowid

    def query(self, sql, params=()):
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def one(self, sql, params=()):
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # --- users -----------------------------------------------------------
    def ensure_user(self, user_id: str):
        self.execute("INSERT OR IGNORE INTO users (id, created_at) VALUES (?, ?)", (user_id, time.time()))
        return self.one("SELECT * FROM users WHERE id = ?", (user_id,))

    def set_resting_bpm(self, user_id: str, bpm: float):
        self.execute("UPDATE users SET resting_bpm = ? WHERE id = ?", (bpm, user_id))

    # --- learning state --------------------------------------------------
    def bandit_history(self, user_id: str):
        rows = self.query(
            "SELECT context, preset, reward FROM segments "
            "WHERE user_id = ? AND mode = 'adaptive' AND valid = 1 AND preset IS NOT NULL "
            "ORDER BY started_at", (user_id,))
        return [(r["context"], r["preset"], r["reward"]) for r in rows]

    def modality_errors(self, user_id: str):
        rows = self.query("SELECT modality, dim, error, n FROM modality_errors WHERE user_id = ?", (user_id,))
        return {(r["modality"], r["dim"]): (r["error"], r["n"]) for r in rows}

    def save_modality_error(self, user_id, modality, dim, error, n):
        self.execute(
            "INSERT INTO modality_errors (user_id, modality, dim, error, n) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, modality, dim) DO UPDATE SET error = excluded.error, n = excluded.n",
            (user_id, modality, dim, error, n))

    def recent_baselines(self, user_id: str, limit: int = 5):
        rows = self.query(
            "SELECT baseline_bpm FROM sessions WHERE user_id = ? AND baseline_bpm IS NOT NULL "
            "ORDER BY started_at DESC LIMIT ?", (user_id, limit))
        return [r["baseline_bpm"] for r in rows]
