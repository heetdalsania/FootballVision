"""
Async SQLite session helpers for FootballVision.

All functions are async and use aiosqlite so they can be called
from FastAPI route handlers without blocking the event loop.
"""

import json
import time
from pathlib import Path
from typing import Optional

try:
    import aiosqlite
    _AIOSQLITE = True
except ImportError:
    _AIOSQLITE = False

from db.models import ALL_TABLES

DB_PATH = Path(__file__).resolve().parent.parent / "footballvision.db"


async def init_db():
    """Create all tables if they do not exist."""
    if not _AIOSQLITE:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        for stmt in ALL_TABLES:
            await db.execute(stmt)
        # A process can be killed without running the lifespan shutdown hook.
        # Do not leave those sessions looking live forever on the next launch.
        await db.execute(
            """UPDATE tactical_sessions
               SET status = 'interrupted',
                   ended_at = COALESCE(ended_at, started_at + elapsed_s)
               WHERE status = 'running'"""
        )
        await db.commit()


async def save_game(
    source: str = "screen",
    game_id: Optional[int] = None,
    ended: bool = False,
) -> int:
    """
    Insert or update a game row.

    Args:
        source:  Video source type string.
        game_id: If provided and ended=True, marks that game as finished.
        ended:   If True, set ended_at on the existing game_id row.

    Returns:
        The game id (new or existing).
    """
    if not _AIOSQLITE:
        return 0

    async with aiosqlite.connect(DB_PATH) as db:
        if ended and game_id:
            await db.execute(
                "UPDATE games SET ended_at = ? WHERE id = ?",
                (time.time(), game_id),
            )
            await db.commit()
            return game_id

        cursor = await db.execute(
            "INSERT INTO games (started_at, source_type) VALUES (?, ?)",
            (time.time(), source),
        )
        await db.commit()
        return cursor.lastrowid


async def save_play(
    game_id: int,
    formation: str,
    play_type: str,
    snap_frame: int,
    run_prob: float,
    pass_prob: float,
) -> int:
    """Insert a play row. Returns the new play id."""
    if not _AIOSQLITE:
        return 0

    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO plays
               (game_id, timestamp, formation, play_type, snap_frame, run_prob, pass_prob)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (game_id, time.time(), formation, play_type, snap_frame, run_prob, pass_prob),
        )
        await db.commit()
        return cursor.lastrowid


async def save_prediction(
    play_id: int,
    session_id: str,
    prediction: str,
    correct: bool,
    points: int,
):
    """Insert a fan prediction row."""
    if not _AIOSQLITE:
        return

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO fan_predictions
               (play_id, session_id, prediction, correct, points)
               VALUES (?, ?, ?, ?, ?)""",
            (play_id, session_id, prediction, int(correct), points),
        )
        await db.commit()


async def create_tactical_session(
    source_name: str,
    source_type: str,
    source_path: Optional[str] = None,
) -> int:
    """Create a durable tactical-analysis session and return its id."""
    if not _AIOSQLITE:
        return 0
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO tactical_sessions
               (source_name, source_type, source_path, started_at, status)
               VALUES (?, ?, ?, ?, 'running')""",
            (source_name, source_type, source_path, time.time()),
        )
        await db.commit()
        return int(cursor.lastrowid)


async def finish_tactical_session(
    session_id: int,
    frames: int,
    elapsed_s: float,
    error: Optional[str] = None,
) -> None:
    """Mark a tactical session complete. Calling this twice is harmless."""
    if not _AIOSQLITE or not session_id:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE tactical_sessions
               SET ended_at = COALESCE(ended_at, ?), frames = ?, elapsed_s = ?,
                   status = ?, error = ?
               WHERE id = ?""",
            (time.time(), int(frames), round(float(elapsed_s), 3),
             "failed" if error else "complete", error, session_id),
        )
        await db.commit()


async def save_tactical_snapshot(session_id: int, payload: dict) -> None:
    """Persist a compact pitch state; the JPEG is deliberately not stored."""
    if not _AIOSQLITE or not session_id:
        return
    pitch = payload.get("pitch") or {}
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR REPLACE INTO tactical_snapshots
               (session_id, frame_id, time_s, players_json, ball_json,
                concepts_json, counts_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                int(payload.get("frame_id") or 0),
                float(payload.get("elapsed_s") or 0),
                json.dumps(pitch.get("players") or [], separators=(",", ":")),
                json.dumps(pitch.get("ball"), separators=(",", ":")),
                json.dumps(pitch.get("concepts"), separators=(",", ":")),
                json.dumps(pitch.get("counts") or {}, separators=(",", ":")),
                time.time(),
            ),
        )
        await db.commit()


def _decode_snapshot(row: dict) -> dict:
    out = dict(row)
    for source, target, fallback in (
        ("players_json", "players", []),
        ("ball_json", "ball", None),
        ("concepts_json", "concepts", None),
        ("counts_json", "counts", {}),
    ):
        raw = out.pop(source, None)
        try:
            out[target] = json.loads(raw) if raw else fallback
        except (TypeError, json.JSONDecodeError):
            out[target] = fallback
    return out


async def list_tactical_sessions(limit: int = 30) -> list[dict]:
    if not _AIOSQLITE:
        return []
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """SELECT s.*, COUNT(p.id) AS snapshots
               FROM tactical_sessions s
               LEFT JOIN tactical_snapshots p ON p.session_id = s.id
               GROUP BY s.id ORDER BY s.started_at DESC LIMIT ?""",
            (max(1, min(int(limit), 200)),),
        )
        return [dict(row) for row in await cur.fetchall()]


async def get_tactical_session(session_id: int) -> Optional[dict]:
    if not _AIOSQLITE:
        return None
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM tactical_sessions WHERE id = ?", (session_id,)
        )
        row = await cur.fetchone()
        return dict(row) if row else None


async def get_tactical_snapshots(session_id: int) -> list[dict]:
    if not _AIOSQLITE:
        return []
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            """SELECT frame_id, time_s, players_json, ball_json,
                      concepts_json, counts_json
               FROM tactical_snapshots WHERE session_id = ?
               ORDER BY frame_id""",
            (session_id,),
        )
        return [_decode_snapshot(dict(row)) for row in await cur.fetchall()]


async def set_tactical_artifacts(
    session_id: int,
    report_path: Optional[str] = None,
    tracking_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
) -> None:
    if not _AIOSQLITE:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE tactical_sessions
               SET report_path = COALESCE(?, report_path),
                   tracking_path = COALESCE(?, tracking_path),
                   metrics_path = COALESCE(?, metrics_path)
               WHERE id = ?""",
            (report_path, tracking_path, metrics_path, session_id),
        )
        await db.commit()
