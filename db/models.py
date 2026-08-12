"""
SQLite schema for FootballVision.

Tables:
    games           — one row per analysis session
    plays           — one row per detected snap
    player_tracks   — positional history per play
    fan_predictions — fan Predict-the-Play submissions
"""

CREATE_GAMES = """
CREATE TABLE IF NOT EXISTS games (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  REAL    NOT NULL,
    ended_at    REAL,
    source_type TEXT    DEFAULT 'screen'
);
"""

CREATE_PLAYS = """
CREATE TABLE IF NOT EXISTS plays (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         INTEGER NOT NULL REFERENCES games(id),
    timestamp       REAL    NOT NULL,
    formation       TEXT,
    play_type       TEXT,
    snap_frame      INTEGER,
    run_prob        REAL,
    pass_prob       REAL
);
"""

CREATE_PLAYER_TRACKS = """
CREATE TABLE IF NOT EXISTS player_tracks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    play_id         INTEGER NOT NULL REFERENCES plays(id),
    player_id       INTEGER NOT NULL,
    positions_json  TEXT    NOT NULL
);
"""

CREATE_FAN_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS fan_predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    play_id     INTEGER NOT NULL,
    session_id  TEXT    NOT NULL,
    prediction  TEXT    NOT NULL,
    correct     INTEGER NOT NULL DEFAULT 0,
    points      INTEGER NOT NULL DEFAULT 0,
    created_at  REAL    NOT NULL DEFAULT (unixepoch('now'))
);
"""

ALL_TABLES = [CREATE_GAMES, CREATE_PLAYS, CREATE_PLAYER_TRACKS, CREATE_FAN_PREDICTIONS]
