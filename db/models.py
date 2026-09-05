"""
SQLite schema for FootballVision.

Tables:
    games           — one row per analysis session
    plays           — one row per detected snap
    player_tracks   — positional history per play
    fan_predictions — fan Predict-the-Play submissions
    tactical_sessions — durable metadata for football analysis runs
    tactical_snapshots — sampled pitch states used by history and reports
    tactical_events — possession changes and conservative match events
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

CREATE_TACTICAL_SESSIONS = """
CREATE TABLE IF NOT EXISTS tactical_sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name     TEXT    NOT NULL,
    source_type     TEXT    NOT NULL,
    source_path     TEXT,
    started_at      REAL    NOT NULL,
    ended_at        REAL,
    frames          INTEGER NOT NULL DEFAULT 0,
    elapsed_s       REAL    NOT NULL DEFAULT 0,
    status          TEXT    NOT NULL DEFAULT 'running',
    error           TEXT,
    report_path     TEXT,
    tracking_path   TEXT,
    metrics_path    TEXT,
    events_path     TEXT
);
"""

CREATE_TACTICAL_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS tactical_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES tactical_sessions(id) ON DELETE CASCADE,
    frame_id        INTEGER NOT NULL,
    time_s          REAL    NOT NULL,
    source_time_s   REAL,
    players_json    TEXT    NOT NULL,
    ball_json       TEXT,
    concepts_json   TEXT,
    counts_json     TEXT,
    intelligence_json TEXT,
    created_at      REAL    NOT NULL,
    UNIQUE(session_id, frame_id)
);
"""

CREATE_TACTICAL_EVENTS = """
CREATE TABLE IF NOT EXISTS tactical_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES tactical_sessions(id) ON DELETE CASCADE,
    event_seq       INTEGER NOT NULL,
    frame_id        INTEGER NOT NULL,
    time_s          REAL    NOT NULL,
    source_time_s   REAL,
    event_type      TEXT    NOT NULL,
    label           TEXT    NOT NULL,
    team            INTEGER,
    player_id       INTEGER,
    confidence      REAL    NOT NULL,
    x               REAL,
    y               REAL,
    detail_json     TEXT    NOT NULL,
    clip_path       TEXT,
    created_at      REAL    NOT NULL,
    UNIQUE(session_id, event_seq)
);
"""

ALL_TABLES = [
    CREATE_GAMES,
    CREATE_PLAYS,
    CREATE_PLAYER_TRACKS,
    CREATE_FAN_PREDICTIONS,
    CREATE_TACTICAL_SESSIONS,
    CREATE_TACTICAL_SNAPSHOTS,
    CREATE_TACTICAL_EVENTS,
]
